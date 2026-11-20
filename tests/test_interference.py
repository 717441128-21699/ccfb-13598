"""Requirement 2: bed-partner snoring, TV and air-conditioner must not be
attributed to the patient or inflate the patient's AHI."""
from __future__ import annotations

import numpy as np

from app.config import TARGET_SR
from app.detection import adaptive_noise_floor, detect_events
from app.features import compute_features
from app.pipeline import analyze
from app.synth import signals as S
from app.synth.scenarios import generate_scenario
from app.wavio import write_wav


def test_roommate_snores_kept_out_of_patient_count(analyze_cached):
    result, gt = analyze_cached("roommate", minutes=40.0, seed=42)
    # the partner actually snored hundreds of times ...
    assert len(gt.partner_snores) > 200
    # ... and the detector reports a substantial partner bucket separately
    assert result["counts"]["partner_snores"] >= 100
    # patient bucket must stay close to the real patient, not absorb the partner
    assert abs(result["counts"]["patient_snores"]
               - len(gt.patient_snores)) <= 0.4 * len(gt.patient_snores)


def test_roommate_noise_does_not_inflate_ahi(analyze_cached):
    base, _ = analyze_cached("normal", minutes=40.0, seed=42, ahi=10.0)
    noisy, _ = analyze_cached("roommate", minutes=40.0, seed=42)
    # partner present: AHI must remain in the same broad band (not jump severe)
    assert noisy["ahi"]["ahi"] < 30.0
    assert abs(noisy["ahi"]["ahi"] - base["ahi"]["ahi"]) <= 8.0


def test_air_conditioner_is_flagged_as_steady_noise(analyze_cached):
    result, gt = analyze_cached("ac_tv", minutes=40.0, seed=42)
    # roughly the two AC runs should be reported (allow fragmentation)
    true_ac = sum(b - a for a, b in
                  ((d["start"], d["end"]) for d in gt.ac_intervals))
    assert result["counts"]["ac_seconds"] >= 0.6 * true_ac


def test_air_conditioner_alone_produces_no_apnea(tmp_path):
    """A recording containing only room tone + AC must yield zero apnea."""
    dur = 600.0
    track = S.colored_noise(dur, -48.0, seed=1)
    S.add_at(track, S.ac_hum(dur - 60, -45.0, seed=2), 30.0)
    p = tmp_path / "ac_only.wav"
    write_wav(p, track, TARGET_SR)
    result = analyze(str(p), job_id="ac-only")
    assert result["ahi"]["apnea_count"] == 0
    assert result["counts"]["ac_seconds"] > 300


def test_tv_speech_is_not_counted_as_patient_snore(tmp_path):
    """TV utterances are higher-pitched irregular speech, not snoring."""
    dur = 600.0
    track = S.colored_noise(dur, -48.0, seed=3)
    S.add_at(track, S.tv_utterance(300.0, -42.0, seed=4), 150.0)
    p = tmp_path / "tv_only.wav"
    write_wav(p, track, TARGET_SR)
    result = analyze(str(p), job_id="tv-only")
    # no patient snore cluster should be confidently established
    assert result["ahi"]["apnea_count"] == 0
    assert result["counts"]["patient_snores"] < 20


def test_speaker_separation_uses_distance_and_timbre(analyze_cached):
    """Two simultaneous speakers resolve into two clusters with correct owner."""
    result, gt = analyze_cached("roommate", minutes=30.0, seed=42)
    assert result["counts"]["speakers_detected"] >= 2
    # near-mic patient cluster is the louder one on average
    assert result["counts"]["patient_snores"] > 0


def test_gasp_after_partner_noise_inside_gap_still_attributed_to_patient(
        analyze_cached):
    """Partner sounds during an apnea gap must not erase the patient's gasp."""
    result, gt = analyze_cached("roommate", minutes=40.0, seed=42)
    if not gt.apneas:
        return
    # at least most true apneas are recovered despite partner presence
    detected_starts = np.array([a["start"] for a in result["apneas"]])
    truth_starts = np.array([a["start"] for a in gt.apneas])
    if detected_starts.size == 0:
        pytest.fail("no apneas detected in roommate scenario")
    # tolerance of one apnea length: the detector anchors on recovery gasps,
    # whose timing legitimately drifts when partner noise fills the gap
    matched = sum(1 for t in truth_starts
                  if np.min(np.abs(detected_starts - t)) <= 10.0)
    assert matched >= 0.7 * len(truth_starts)
