"""AHI accuracy and the headline clinical requirement.

Requirement 1: for genuinely severe patients (true AHI >= 30), the screening
verdict must agree with the polysomnography-grade ground truth with a
misclassification rate no higher than 10%.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.apnea import severity_band
from app.synth.scenarios import SCENARIOS

# (scenario, target AHI, expected severity band). Targets stay within the
# physical event density the synthesiser can represent (one apnea >= 10 s,
# plus recovery, every ~12-20 s) while covering all four clinical bands.
SEVERITY_GRID = [
    ("normal", 2.0, "none"),
    ("normal", 8.0, "mild"),
    ("mild", 12.0, "mild"),
    ("moderate", 18.0, "moderate"),
    ("moderate", 25.0, "moderate"),
    ("severe", 32.0, "severe"),
    ("severe", 40.0, "severe"),
    ("severe", 45.0, "severe"),
    ("severe", 55.0, "severe"),
]


def _event_match(detected: list[dict], truth: list[dict], tol: float = 8.0):
    """Greedy one-to-one time matching of detected vs true apnea events.

    Tolerance is generous on purpose: synthesised apneas run 10-35 s and the
    detector anchors on the recovery gasp, so start times legitimately drift.
    """
    used = set()
    hits = 0
    for d in detected:
        cand = [(j, abs(d["start"] - t["start"])) for j, t in enumerate(truth)
                if j not in used and abs(d["start"] - t["start"]) <= tol]
        if cand:
            j = min(cand, key=lambda c: c[1])[0]
            used.add(j)
            hits += 1
    return hits


def test_severity_band_thresholds():
    assert severity_band(0) == "none"
    assert severity_band(6) == "mild"
    assert severity_band(17) == "moderate"
    assert severity_band(30) == "severe"
    assert severity_band(77) == "severe"


@pytest.mark.parametrize("scenario,ahi,band", SEVERITY_GRID)
def test_severity_classification(analyze_cached, scenario, ahi, band):
    # 40-minute recordings give enough events for stable hourly rates
    result, gt = analyze_cached(scenario, minutes=40.0, seed=42, ahi=ahi)
    assert result["ahi"]["severity"] == band


def test_ahi_numeric_error_within_band(analyze_cached):
    """Estimated AHI should track the target across the dynamic range."""
    errs = []
    for scenario, ahi, _band in SEVERITY_GRID:
        result, _ = analyze_cached(scenario, minutes=40.0, seed=42, ahi=ahi)
        errs.append(abs(result["ahi"]["ahi"] - ahi))
    # median absolute error well within one severity band width (15 events/h)
    assert np.median(errs) <= 10.0
    assert np.percentile(errs, 80) <= 15.0


def test_apnea_event_recall_and_precision(analyze_cached):
    result, gt = analyze_cached("severe", minutes=40.0, seed=42, ahi=45.0)
    detected = result["apneas"]
    hits = _event_match(detected, gt.apneas)
    recall = hits / max(1, len(gt.apneas))
    precision = hits / max(1, len(detected))
    assert recall >= 0.75
    assert precision >= 0.75


def test_severe_misclassification_under_10_percent(analyze_cached):
    """The headline requirement over many independent severe nights."""
    seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
    errors = 0
    details = []
    for seed in seeds:
        result, gt = analyze_cached("severe", minutes=40.0, seed=seed, ahi=45.0)
        true_severe = gt.target_ahi >= 30.0
        called_severe = result["ahi"]["is_severe"]
        if true_severe != called_severe:
            errors += 1
            details.append((seed, result["ahi"]["ahi"], result["ahi"]["severity"]))
    rate = errors / len(seeds)
    assert rate <= 0.10, f"severe misclassification {rate:.0%}, details={details}"


def test_non_severe_rarely_upgraded_to_severe(analyze_cached):
    """Guard against false alarms that would waste a prioritised bed."""
    overcalls = 0
    for seed in (12, 24, 36):
        result, _ = analyze_cached("normal", minutes=40.0, seed=seed, ahi=3.0)
        if result["ahi"]["is_severe"]:
            overcalls += 1
    assert overcalls == 0


def test_all_scenarios_return_well_formed_result(analyze_cached):
    for scen in SCENARIOS:
        result, gt = analyze_cached(scen, minutes=20.0, seed=42)
        assert set(result) >= {
            "job_id", "duration", "ahi", "envelope", "hourly",
            "positions", "apneas", "clips", "counts"}
        assert result["envelope"]["seconds"] >= int(result["duration"]) - 1
        assert len(result["clips"]) <= 10
