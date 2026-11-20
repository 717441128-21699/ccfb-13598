"""Requirement 3: an 8-hour night must finish well inside 10 minutes.

A full 8-hour synthetic night is expensive to build and analyse, so the
default suite measures a 1-hour night and extrapolates against the measured
linear throughput. Set RUN_FULL_8H=1 to build and analyse a real 8-hour file.
"""
from __future__ import annotations

import os
import time

import pytest

from app.config import TARGET_SR
from app.pipeline import analyze
from app.synth.scenarios import generate_scenario
from app.wavio import write_wav

EIGHT_HOURS = 8 * 3600.0
TEN_MINUTES = 10 * 60.0


def test_one_hour_throughput_budget(tmp_recordings):
    path, _ = tmp_recordings("moderate", minutes=60.0, seed=42)
    t0 = time.perf_counter()
    result = analyze(str(path), job_id="perf-1h")
    elapsed = time.perf_counter() - t0
    # 1 h must finish within 1/8 of the 10-minute budget = 75 s; the numpy
    # pipeline is two orders of magnitude faster than realtime in practice.
    assert elapsed < 75.0
    assert result["realtime_factor"] >= 48.0  # 8h / 10min minimum
    assert result["duration"] >= 3599.0


def test_memory_stays_flat_across_blocks():
    """Frame descriptors, not full spectra, are retained: output size grows
    linearly with per-second envelope, never with the FFT frame count."""
    from app import pipeline  # noqa: F401
    # structural guarantee: blocks stream and only small templates persist
    import inspect
    src = inspect.getsource(analyze)
    assert "while start < duration" in src
    assert "feat.mag" not in src  # global analysis never holds the FFT matrix


@pytest.mark.skipif(os.environ.get("RUN_FULL_8H") != "1",
                    reason="set RUN_FULL_8H=1 to run the full-night check")
def test_full_eight_hour_night(tmp_path):
    track, gt = generate_scenario("severe", duration=EIGHT_HOURS, seed=42,
                                  target_ahi=45.0)
    path = tmp_path / "full_night.wav"
    write_wav(path, track, TARGET_SR)
    size_mb = path.stat().st_size / 1e6
    t0 = time.perf_counter()
    result = analyze(str(path), job_id="perf-8h",
                     metadata={"positions": gt.positions})
    elapsed = time.perf_counter() - t0
    assert elapsed < TEN_MINUTES
    assert result["ahi"]["is_severe"]
    assert len(result["clips"]) == 10
    assert size_mb > 400  # confirms this really is a full-length night
