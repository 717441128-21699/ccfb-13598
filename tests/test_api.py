"""End-to-end HTTP tests for the FastAPI upload/poll/clip workflow."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import storage
from app.main import app
from app.synth.scenarios import generate_scenario
from app.synth.signals import SR
from app.wavio import write_wav


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # isolate the SQLite DB and clip storage into a temp location
    tmp = tmp_path_factory.mktemp("api")
    storage.DB_PATH = tmp / "jobs.sqlite3"
    from app import config
    config.UPLOAD_DIR = tmp / "uploads"
    config.CLIP_DIR = tmp / "clips"
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.CLIP_DIR.mkdir(parents=True, exist_ok=True)
    # main.py imported the original path constants; rebind them for the test
    import app.main as main_mod
    main_mod.UPLOAD_DIR = config.UPLOAD_DIR
    main_mod.CLIP_DIR = config.CLIP_DIR
    storage.init_db()
    with TestClient(app) as c:
        yield c


def _wav_bytes(tmp_path, scenario="mild", minutes=10.0):
    track, gt = generate_scenario(scenario, duration=minutes * 60, seed=42)
    p = tmp_path / f"{scenario}.wav"
    write_wav(p, track, SR)
    return p.read_bytes(), gt


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_rejects_non_wav(client):
    r = client.post("/api/jobs",
                    files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_full_upload_poll_result_flow(client, tmp_path):
    wav, gt = _wav_bytes(tmp_path)
    pos = {"positions": gt.positions}
    r = client.post(
        "/api/jobs",
        files={
            "file": ("night.wav", wav, "audio/wav"),
            "position_file": ("pos.json", json.dumps(pos), "application/json"),
        },
    )
    assert r.status_code == 201
    job_id = r.json()["job_id"]

    # poll until terminal state (analysis is fast on a 10-min file)
    final = None
    for _ in range(60):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            final = job
            break
    assert final is not None and final["status"] == "done", final
    result = final["result"]
    assert result["job_id"] == job_id
    assert "ahi" in result and "envelope" in result
    assert result["duration"] >= 599.0


def test_job_listing(client, tmp_path):
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert isinstance(r.json()["jobs"], list) and r.json()["jobs"]


def test_clip_download_and_404(client, tmp_path):
    wav, _ = _wav_bytes(tmp_path, scenario="moderate", minutes=10.0)
    r = client.post("/api/jobs", files={"file": ("m.wav", wav, "audio/wav")})
    job_id = r.json()["job_id"]
    result = None
    for _ in range(60):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "done":
            result = job["result"]
            break
    assert result and result["clips"], "expected at least one apnea clip"
    clip_url = result["clips"][0]["url"]
    cr = client.get(clip_url)
    assert cr.status_code == 200
    assert cr.headers["content-type"] == "audio/wav"
    assert len(cr.content) > 1000
    # path traversal is rejected
    bad = client.get(f"/api/clips/{job_id}/..%2f..%2fsecret.wav")
    assert bad.status_code in (400, 404)


def test_missing_job_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404
