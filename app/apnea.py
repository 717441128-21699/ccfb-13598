"""Apnea detection, AHI estimation and severity grading."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    AHI_MILD, AHI_MODERATE, AHI_SEVERE, APNEA_MAX_GAP_SEC, APNEA_MIN_GAP_SEC,
    FRAMES_PER_SEC, GASP_SEARCH_SEC, MIN_PATIENT_ACTIVE_FRACTION,
)
from .detection import DetectionResult, Event, find_gasp
from .features import FrameFeatures


@dataclass
class Apnea:
    start: float                    # last patient breath before the pause
    end: float                      # first recovery breath / gasp time
    duration: float
    gasp_time: float | None
    confirmed_by_gasp: bool

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "gasp_time": None if self.gasp_time is None else round(self.gasp_time, 3),
            "confirmed_by_gasp": self.confirmed_by_gasp,
        }


def severity_band(ahi: float) -> str:
    if ahi < AHI_MILD:
        return "none"
    if ahi < AHI_MODERATE:
        return "mild"
    if ahi < AHI_SEVERE:
        return "moderate"
    return "severe"


def _patient_events(det: DetectionResult) -> list[Event]:
    return [e for e in det.events
            if e.is_snore and det.is_patient_cluster(e.cluster)]


def _typical_breath_gap(patient_events) -> float:
    """Median inter-snore gap (seconds) for this patient's normal rhythm."""
    if len(patient_events) < 4:
        return 4.5
    starts = np.sort(np.array([e.start for e in patient_events], dtype=np.float64))
    gaps = np.diff(starts)
    gaps = gaps[(gaps > 0.5) & (gaps < 9.0)]      # exclude apnea gaps themselves
    if gaps.size < 3:
        return 4.5
    return float(np.median(gaps))


def detect_apneas(feat: FrameFeatures, det: DetectionResult) -> list[Apnea]:
    """Find breathing pauses between consecutive patient snores.

    A pause is accepted when (a) it spans the apnea duration window and is
    clearly longer than the patient's normal breathing cycle, (b) the room
    stays quiet in it relative to the gap's own median level (partner's
    TV/snoring does not fill the patient's airflow gap), and (c) either an
    explosive recovery gasp is detected or patient snoring resumes after an
    exceptionally quiet gap.
    """
    pat = _patient_events(det)
    apneas: list[Apnea] = []
    if len(pat) < 2:
        return apneas
    typical_gap = _typical_breath_gap(pat)

    for prev, nxt in zip(pat[:-1], pat[1:]):
        gap = nxt.start - prev.end
        if not (APNEA_MIN_GAP_SEC <= gap <= APNEA_MAX_GAP_SEC):
            continue
        # an apnea pause must clearly exceed the patient's normal breathing
        # cycle; the 10 s AASM floor is the hard clinical minimum anyway
        if gap < 2.5 * typical_gap:
            continue

        a = int(np.searchsorted(feat.times, prev.end))
        b = int(np.searchsorted(feat.times, nxt.start))
        if b <= a:
            continue
        gap_db = feat.db[a:b]
        # Quietness measured by SPREAD, not absolute level, so a constant AC
        # hum filling the gap does not look like the patient breathing. Two
        # complementary tests (percentile spread and median deviation),
        # either passing is sufficient:
        #   * natural room tone: tight cluster around its own baseline
        #   * constant hum: near-zero median absolute deviation
        baseline = float(np.percentile(gap_db, 10))
        spread_frac = float(np.mean(gap_db <= baseline + 6.0))
        med = float(np.median(gap_db))
        mad = float(np.median(np.abs(gap_db - med)))
        mad_frac = float(np.mean(np.abs(gap_db - med) < max(4.5, 2.5 * mad)))
        quiet_fraction = max(spread_frac, mad_frac)
        if quiet_fraction < 0.7:
            continue

        gasp_t = find_gasp(feat, nxt.start - 0.5, nxt.start + GASP_SEARCH_SEC,
                           pre_level=baseline, ac_mask=det.ac_mask)
        if gasp_t is not None:
            end_t = gasp_t
            confirmed = True
        else:
            # fallback: only an exceptionally quiet pause (99% of frames at
            # the gap's own floor) followed by the patient resuming counts;
            # ordinary quiet breathing gaps stay below this bar.
            if quiet_fraction >= 0.99:
                end_t = nxt.start
                confirmed = False
            else:
                continue

        apneas.append(Apnea(
            start=prev.end,
            end=float(end_t),
            duration=float(end_t - prev.end),
            gasp_time=gasp_t,
            confirmed_by_gasp=confirmed,
        ))
    return apneas


def estimate_ahi(apneas: list[Apnea], duration_hours: float,
                 det: DetectionResult, total_frames: int) -> dict:
    """AHI = number of apnea events per estimated sleep hour."""
    n = len(apneas)
    ahi = n / duration_hours if duration_hours > 0 else 0.0

    active_frames = int(det.snore_mask.sum())
    active_fraction = active_frames / max(1, total_frames)
    # Low confidence means the recording itself is unusable for attribution:
    # no stable patient voice, or near-zero patient activity. Frequent apnea
    # naturally lowers the snore fraction (long silent pauses), so a recording
    # with many detected events is NOT low confidence even when the fraction
    # looks small.
    confidence = "normal"
    notes: list[str] = []
    if det.patient_cluster < 0:
        confidence = "low"
        notes.append("未找到稳定的病人鼾声音源，AHI 仅供参考（可能录音位置不对或病人基本不打鼾）。")
    elif active_fraction < MIN_PATIENT_ACTIVE_FRACTION and n < 5:
        confidence = "low"
        notes.append("整夜病人声信号占比过低且未检出明确暂停，建议复核录音质量后再下结论。")

    gasp_confirmed = sum(1 for a in apneas if a.confirmed_by_gasp)
    return {
        "apnea_count": n,
        "gasp_confirmed": gasp_confirmed,
        "ahi": round(ahi, 1),
        "severity": severity_band(ahi),
        "is_severe": ahi >= AHI_SEVERE,
        "confidence": confidence,
        "notes": notes,
        "patient_active_fraction": round(active_fraction, 4),
        "duration_hours": round(duration_hours, 3),
    }
