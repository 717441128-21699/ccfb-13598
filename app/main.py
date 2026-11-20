"""FastAPI application: upload overnight recordings, run screening, serve results.

Endpoints
---------
POST /api/jobs                upload a WAV (+ optional position metadata JSON)
GET  /api/jobs                list recent jobs
GET  /api/jobs/{id}           job status + full analysis result when done
GET  /api/clips/{id}/{name}   stream a flagged apnea clip WAV
GET  /health                  liveness probe
GET  /                        doctor web console (static single page)
"""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import storage
from .config import CLIP_DIR, MAX_UPLOAD_MB, UPLOAD_DIR
from .pipeline import analyze

app = FastAPI(title="睡眠呼吸暂停手机录音初筛系统", version="1.0.0")
_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
def _run_analysis(job_id: str, wav_path: Path, metadata: dict | None) -> None:
    try:
        storage.set_status(job_id, "processing")
        result = analyze(str(wav_path), job_id=job_id, metadata=metadata)
        storage.save_result(job_id, result)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as job error
        storage.set_status(job_id, "error", error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.post("/api/jobs", status_code=201)
async def create_analysis_job(background: BackgroundTasks,
                              file: UploadFile,
                              position_file: UploadFile | None = None):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "仅支持 WAV 录音文件（.wav）")
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(400, "上传文件为空")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"文件超过 {MAX_UPLOAD_MB}MB 上限")

    metadata = None
    if position_file is not None:
        import json
        try:
            metadata = json.loads((await position_file.read()).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"体位元数据 JSON 解析失败: {exc}") from exc

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    wav_path = job_dir / file.filename
    wav_path.write_bytes(raw)

    storage.create_job(job_id, file.filename)
    background.add_task(_run_analysis, job_id, wav_path, metadata)
    return {"job_id": job_id, "status": "queued",
            "poll_url": f"/api/jobs/{job_id}"}


@app.get("/api/jobs")
def get_jobs():
    return {"jobs": storage.list_jobs()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = storage.get_job(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return job


@app.get("/api/clips/{job_id}/{name}")
def get_clip(job_id: str, name: str):
    # guard against path traversal
    if "/" in name or ".." in name or not name.endswith(".wav"):
        raise HTTPException(400, "非法文件名")
    path = CLIP_DIR / job_id / name
    if not path.is_file():
        raise HTTPException(404, "音频片段不存在")
    return FileResponse(str(path), media_type="audio/wav", filename=name)


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Static doctor console
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    page = _WEB_DIR / "index.html"
    if not page.is_file():
        return JSONResponse({"detail": "网页未构建"}, status_code=404)
    return FileResponse(str(page))


app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")
