"""Shared pytest fixtures: synthetic recordings on demand.

Recordings are generated once per (scenario, minutes, seed) and cached in a
temporary directory, so the accuracy suite does not repeatedly synthesise the
same audio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import analyze
from app.synth.scenarios import SCENARIOS, generate_scenario
from app.synth.signals import SR
from app.wavio import write_wav


@pytest.fixture(scope="session")
def tmp_recordings(tmp_path_factory):
    base = tmp_path_factory.mktemp("recordings")
    cache: dict[tuple, tuple[Path, dict]] = {}

    def _make(scenario: str, minutes: float = 20.0, seed: int = 42,
              ahi: float | None = None):
        key = (scenario, minutes, seed, ahi)
        if key not in cache:
            track, gt = generate_scenario(
                scenario, duration=minutes * 60.0, seed=seed,
                target_ahi=ahi)
            path = base / f"{scenario}_{minutes}_{seed}_{ahi}.wav"
            write_wav(path, track, SR)
            cache[key] = (path, gt)
        return cache[key]

    return _make


@pytest.fixture(scope="session")
def analyze_cached(tmp_recordings):
    results: dict[tuple, dict] = {}

    def _run(scenario: str, minutes: float = 20.0, seed: int = 42,
             ahi: float | None = None):
        key = (scenario, minutes, seed, ahi)
        if key not in results:
            path, gt = tmp_recordings(scenario, minutes, seed, ahi)
            results[key] = analyze(
                str(path), job_id=f"t-{abs(hash(key)) % 10**8}",
                metadata={"positions": gt.positions})
        path, gt = tmp_recordings(scenario, minutes, seed, ahi)
        return results[key], gt

    return _run
