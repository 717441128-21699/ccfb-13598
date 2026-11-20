"""Synthetic all-night scenario generator.

Every scenario returns the float32 audio track plus a machine-readable ground
truth that the pytest suite checks the detector against. The modelled patient
breathes in rhythmic cycles; at a target AHI density a cycle is replaced by an
apnea (silent airflow gap) terminated by an explosive gasp and a loud recovery
snore. Interferers are mixed on independent schedules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import signals as S

# ---------------------------------------------------------------------------
# Levels (dBFS) — the phone sits on the patient's bedside table.
# ---------------------------------------------------------------------------
ROOM_DB = -48.0
PATIENT_SNORE_DB = -24.0
PATIENT_GASP_DB = -18.0
# The bed partner is across the bed: a bedside phone picks them up ~15 dB
# below the near-mic patient, with a distinct higher formant profile.
PARTNER_SNORE_DB = -39.0
# TV and AC sit at realistic background levels (a bedside phone sees them
# ~20 dB below a near-mic snore); they must be classified, not mistaken for
# the patient's breathing.
TV_DB = -42.0
AC_DB = -45.0

SCENARIOS = (
    "normal", "mild", "moderate", "severe",
    "roommate", "ac_tv", "mixed",
)


@dataclass
class GroundTruth:
    scenario: str
    duration: float
    target_ahi: float
    apneas: list[dict] = field(default_factory=list)
    patient_snores: list[dict] = field(default_factory=list)
    partner_snores: list[dict] = field(default_factory=list)
    gasps: list[dict] = field(default_factory=list)
    ac_intervals: list[dict] = field(default_factory=list)
    tv_intervals: list[dict] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _schedule_apneas(duration: float, target_ahi: float, rng: np.random.Generator) -> list[float]:
    """Place apnea start times at the target density with jitter, guaranteed
    to keep a clinically realistic minimum spacing even at high AHI.
    """
    n = max(0, int(round(target_ahi * duration / 3600.0)))
    if n == 0:
        return []
    # keep a short margin at both ends for sleep onset / wake-up, scaled to
    # the recording length so short test files still contain events
    margin = min(600.0, max(30.0, 0.1 * duration))
    lo, hi = margin, max(margin + 1.0, duration - margin)
    span = hi - lo
    # Denser rates get tighter spacing so high-AHI nights keep all their
    # events: cap spacing at 20 s, relax toward 12 s as density grows.
    n_requested = n
    min_spacing = float(np.clip(span / max(1, n_requested), 12.0, 20.0))
    n = min(n_requested, max(1, int(span / min_spacing)))
    base = np.linspace(lo, hi, n)
    spacing = span / max(1, n)
    # jitter stays inside half the available slack so order/spacing survive
    jitter_amp = min(0.3 * spacing, 0.45 * min_spacing)
    jitter = rng.uniform(-jitter_amp, jitter_amp, n)
    times = np.clip(base + jitter, lo, hi)
    times.sort()
    # enforce the (density-adapted) minimum spacing
    out: list[float] = []
    for t in times:
        if not out or t - out[-1] >= min_spacing:
            out.append(float(t))
    return out


def _apnea_duration(target_ahi: float, rng: np.random.Generator) -> float:
    if target_ahi >= 30:
        return float(rng.uniform(16.0, 35.0))
    if target_ahi >= 15:
        return float(rng.uniform(13.0, 26.0))
    return float(rng.uniform(10.5, 20.0))


def generate_scenario(scenario: str, duration: float = 1800.0,
                      seed: int = 42, target_ahi: float | None = None,
                      positions: list[dict] | None = None) -> tuple[np.ndarray, GroundTruth]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    if duration < 300:
        raise ValueError("scenario duration must be >= 300 s")
    rng = _rng(seed)
    n = int(duration * S.SR)
    track = S.silence(n)

    # persistent low room tone
    S.add_at(track, S.colored_noise(duration, ROOM_DB, seed=seed), 0.0)

    gt = GroundTruth(scenario=scenario, duration=duration,
                     target_ahi=float(target_ahi) if target_ahi is not None else 0.0)

    # ----- interferer schedules -----
    use_ac = scenario in ("ac_tv", "mixed")
    use_tv = scenario in ("ac_tv", "mixed")
    use_partner = scenario in ("roommate", "mixed")
    if use_ac:
        # two long steady runs with a short break
        for a, b in ((300, duration * 0.45), (duration * 0.5, duration - 120)):
            S.add_at(track, S.ac_hum(b - a, AC_DB, seed=seed + 11), a)
            gt.ac_intervals.append({"start": round(a, 2), "end": round(b, 2)})
    if use_tv:
        for a, b in ((duration * 0.12, duration * 0.3),
                     (duration * 0.62, duration * 0.82)):
            S.add_at(track, S.tv_utterance(b - a, TV_DB, seed=seed + 22), a)
            gt.tv_intervals.append({"start": round(a, 2), "end": round(b, 2)})

    # ----- patient breathing schedule with injected apneas -----
    if target_ahi is None:
        target_ahi = {
            "normal": 2.0, "mild": 8.0, "moderate": 20.0, "severe": 45.0,
            "roommate": 12.0, "ac_tv": 10.0, "mixed": 38.0,
        }[scenario]
    gt.target_ahi = float(target_ahi)
    apnea_times = _schedule_apneas(duration, target_ahi, rng)

    # partner snores on their own independent rhythm (3-5 s cycles)
    partner_times: list[float] = []
    if use_partner:
        t = 200.0
        while t < duration - 120:
            partner_times.append(t)
            t += rng.uniform(3.2, 5.2)

    # Event-driven schedule: walk the apnea markers in order and fill the
    # gaps between them with ordinary rhythmic snores, so every scheduled
    # apnea is injected exactly once regardless of breathing-cycle jitter.
    cycle_k = 0
    markers = list(apnea_times) + [duration - 5.0]
    t = 30.0
    for marker in markers:
        # ordinary breathing snores until the next apnea marker
        while t < marker - 2.0 and t < duration - 5:
            cycle = rng.uniform(3.6, 5.2)
            if rng.uniform() > 0.12:
                dur = rng.uniform(0.35, 0.8)
                sn = S.snore(dur, S.PATIENT_VOICE,
                             PATIENT_SNORE_DB + rng.uniform(-2, 2),
                             seed=seed + 3000 + cycle_k)
                S.add_at(track, sn, t)
                gt.patient_snores.append(
                    {"start": round(t, 2), "duration": round(dur, 2)})
            t += cycle + rng.uniform(-0.25, 0.25)
            cycle_k += 1

        if marker >= duration - 5.0:
            break
        # inject the apnea at the marker (snoring stops for its full length)
        t = marker
        ap_dur = _apnea_duration(target_ahi, rng)
        gasp_t = t + ap_dur
        g = S.gasp(rng.uniform(0.4, 0.65), S.PATIENT_VOICE,
                   PATIENT_GASP_DB, seed=seed + 1000 + cycle_k)
        S.add_at(track, g, gasp_t)
        gt.gasps.append({"start": round(gasp_t, 2)})
        rec = S.snore(rng.uniform(0.4, 0.7), S.PATIENT_VOICE,
                      PATIENT_SNORE_DB + 2, seed=seed + 2000 + cycle_k)
        S.add_at(track, rec, gasp_t + 0.9)
        gt.patient_snores.append({"start": round(gasp_t + 0.5, 2)})
        gt.apneas.append({
            "start": round(t, 2),
            "end": round(gasp_t, 2),
            "duration": round(ap_dur, 2),
            "gasp_time": round(gasp_t, 2),
        })
        t = gasp_t + 1.2
        cycle_k += 1

    # partner sounds go in after patient schedule so they may fall inside
    # patient apnea gaps — the detector must NOT credit them to the patient.
    for k, pt in enumerate(partner_times):
        dur = rng.uniform(0.35, 0.75)
        ps = S.snore(dur, S.PARTNER_VOICE,
                     PARTNER_SNORE_DB + rng.uniform(-2, 2), seed=seed + 5000 + k)
        S.add_at(track, ps, pt)
        gt.partner_snores.append({"start": round(pt, 2), "duration": round(dur, 2)})

    # ----- body position (phone accelerometer sidecar) -----
    if positions is None:
        positions = [
            {"start": 0, "end": int(duration * 0.45), "position": "supine"},
            {"start": int(duration * 0.45), "end": int(duration * 0.7), "position": "left"},
            {"start": int(duration * 0.7), "end": int(duration), "position": "right"},
        ]
    gt.positions = positions

    return S.normalise_track(track), gt
