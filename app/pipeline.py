"""End-to-end analysis orchestration.

The night is processed in overlapping blocks so memory stays flat regardless
of recording length; frame-level descriptors (small) are concatenated for the
global event / apnea stage, while per-block magnitude spectra (large) are
discarded after snore templates have been extracted.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

from . import wavio
from .apnea import detect_apneas, estimate_ahi
from .config import (
    BLOCK_OVERLAP_SECONDS, BLOCK_SECONDS, CLIP_DIR, CLIP_MAX_SEC,
    CLIP_PAD_SEC, FRAMES_PER_SEC, TARGET_SR, TOP_CLIPS,
)
from .detection import (
    DetectionResult, _ac_frame_mask, _extract_patches,
    adaptive_noise_floor, detect_events, frame_snore_score,
    SPEECH_CENTROID_HZ,
)
from .features import FrameFeatures, compute_features

POSITION_LABELS = {
    "supine": "仰卧", "prone": "俯卧", "left": "左侧卧", "right": "右侧卧",
    "upright": "坐起", "unknown": "未知",
}


# ---------------------------------------------------------------------------
# Streaming block processing
# ---------------------------------------------------------------------------
def _block_features(x: np.ndarray, block_start: float, block_end: float) -> dict | None:
    """Compute per-frame descriptors for one block and discard its spectra.

    Keeping every descriptor (but not the FFT matrix) makes memory flat in
    recording length regardless of night length.
    """
    feat = compute_features(x, TARGET_SR, time_offset=block_start)
    if feat.db.size == 0:
        return None
    inside = feat.times < block_end
    nf = adaptive_noise_floor(feat.db[inside])
    return {
        "times": feat.times[inside],
        "db": feat.db[inside],
        "zcr": feat.zcr[inside],
        "centroid": feat.centroid[inside],
        "flatness": feat.flatness[inside],
        "low_frac": feat.low_frac[inside],
        "mid_frac": feat.mid_frac[inside],
        "noise_floor": nf,
    }


def _extract_block_templates(x: np.ndarray, block_start: float,
                             block_end: float, global_floor: float
                             ) -> list[tuple[float, float, np.ndarray]]:
    """Second pass: extract snore templates with the NIGHT-wide noise floor.

    A single global threshold keeps templates consistent with global event
    detection, so block-local interferers cannot change the snore threshold.
    Spectra are computed, used, and released inside this one block.
    """
    from .config import (
        VAD_ABS_FLOOR_DB, VAD_MARGIN_DB, SNORE_PATCH_ACCEPT,
        EVENT_MERGE_GAP, SNORE_MIN_DUR, SNORE_MAX_DUR, SNORE_MIN_MARGIN_DB,
        SNORE_MIN_MODULATION_DB, RESEGMENT_MERGE_GAP, SNORE_MIN_LOW_FRAC,
    )
    feat = compute_features(x, TARGET_SR, time_offset=block_start)
    if feat.db.size == 0:
        return []
    nfr = feat.db.size
    vad = feat.db > np.maximum(global_floor + VAD_MARGIN_DB, VAD_ABS_FLOOR_DB)
    score = frame_snore_score(feat, global_floor)
    speech = (feat.centroid > SPEECH_CENTROID_HZ) & (feat.flatness > 0.22)
    snore_frames = (vad & ~speech
                    & (feat.low_frac >= SNORE_MIN_LOW_FRAC)
                    & (score > SNORE_PATCH_ACCEPT))
    # modulation signal for re-segmenting steady-hum-bridged long patches
    win, padw = 25, 12
    padded_db = np.pad(feat.db, (padw, padw), mode="edge")
    indexer = np.arange(nfr)[:, None] + np.arange(win)[None, :]
    local_range = padded_db[indexer].max(axis=1) - padded_db[indexer].min(axis=1)
    modulated = local_range >= SNORE_MIN_MODULATION_DB
    merge_f = int(EVENT_MERGE_GAP * FRAMES_PER_SEC)
    min_f = max(1, int(SNORE_MIN_DUR * FRAMES_PER_SEC))
    max_f = int(SNORE_MAX_DUR * FRAMES_PER_SEC)

    templates: list[tuple[float, float, np.ndarray]] = []
    bg = feat.mag.mean(axis=0) + 1e-6
    raw_patches = _extract_patches(snore_frames, merge_f)
    patch_list: list[tuple[int, int]] = []
    frag_merge = int(RESEGMENT_MERGE_GAP * FRAMES_PER_SEC)
    for a0, b0 in raw_patches:
        if b0 - a0 > max_f:
            mod_runs = _extract_patches(modulated[a0:b0], merge_gap_frames=merge_f)
            cuts = [(a0 + int(ra), a0 + int(rb)) for ra, rb in mod_runs
                    if rb - ra >= min_f]
            if cuts:
                merged = [cuts[0]]
                for fa, fb in cuts[1:]:
                    if fa - merged[-1][1] <= frag_merge:
                        merged[-1] = (merged[-1][0], fb)
                    else:
                        merged.append((fa, fb))
                patch_list.extend(merged)
            else:
                patch_list.append((int(a0), int(b0)))
        else:
            patch_list.append((int(a0), int(b0)))
    for a, b in patch_list:
        if b - a < min_f or (b - a) > max_f:
            continue
        if float(feat.db[a:b].mean()) < global_floor + SNORE_MIN_MARGIN_DB:
            continue
        if float(feat.times[(a + b) // 2]) >= block_end:
            continue
        # Full whitening removes the steady hum's spectral bias; speaker
        # identity is then carried by combining the whitened spectrum with
        # level and centroid descriptors (see _build_cluster_vectors).
        whitened = feat.mag[a:b].mean(axis=0) / bg
        norm = np.linalg.norm(whitened) + 1e-9
        templates.append((float(feat.times[a]), float(feat.times[b - 1]),
                          (whitened / norm).astype(np.float32)))
    return templates


def _merge_blocks(parts: list[dict]) -> tuple[FrameFeatures, float]:
    feat = FrameFeatures(
        times=np.concatenate([p["times"] for p in parts]),
        db=np.concatenate([p["db"] for p in parts]),
        zcr=np.concatenate([p["zcr"] for p in parts]),
        centroid=np.concatenate([p["centroid"] for p in parts]),
        flatness=np.concatenate([p["flatness"] for p in parts]),
        low_frac=np.concatenate([p["low_frac"] for p in parts]),
        mid_frac=np.concatenate([p["mid_frac"] for p in parts]),
        mag=np.zeros((0, 0), np.float32),
    )
    noise_floor = float(np.median([p["noise_floor"] for p in parts]))
    return feat, noise_floor


def _attach_templates(events: list, templates: list[tuple]) -> None:
    """Match cached block-level spectral templates onto detected snore events.

    Both lists derive from the same snore-frame patches, so a greedy
    nearest-start match is unambiguous. Events without a match (rare edge
    patches) keep a zero template and are excluded from clustering.
    """
    if not templates:
        return
    tpl_starts = np.array([t[0] for t in templates])
    used = np.zeros(len(templates), dtype=bool)
    for ev in events:
        if not ev.is_snore or ev.template.size > 0:
            continue
        order = np.argsort(np.abs(tpl_starts - ev.start))
        for k in order:
            if used[k]:
                continue
            t0, t1, spec = templates[int(k)]
            if abs(t0 - ev.start) <= 0.05 and abs(t1 - ev.end) <= 0.15:
                ev.template = spec
                used[int(k)] = True
            break


# ---------------------------------------------------------------------------
# Gasp recovery
# ---------------------------------------------------------------------------
def _recover_gasp_events(feat, det: DetectionResult, noise_floor: float) -> None:
    """Re-attribute post-apnea recovery breaths to the patient.

    The routine snore classifier can reject the broadband gasp (or park it in
    a secondary cluster). We scan every long quiet gap between consecutive
    patient snores and reclaim any gasp-shaped event in the recovery window,
    so detect_apneas sees a continuous patient breathing sequence.
    """
    from .config import (
        APNEA_MAX_GAP_SEC, APNEA_MIN_GAP_SEC, GASP_SEARCH_SEC,
    )
    from .detection import gasp_like_events

    if det.patient_cluster < 0:
        return

    from .apnea import _typical_breath_gap

    for _ in range(5):
        reclaimed = False
        candidates = gasp_like_events(feat, noise_floor, det.ac_mask)
        candidates += [e for e in det.events
                       if e.is_snore and not det.is_patient_cluster(e.cluster)]
        pat = [e for e in det.events
               if e.is_snore and det.is_patient_cluster(e.cluster)]
        if len(pat) < 2:
            break
        typical_gap = _typical_breath_gap(pat)
        for prev, nxt in zip(pat[:-1], pat[1:]):
            gap = nxt.start - prev.end
            if not (APNEA_MIN_GAP_SEC <= gap <= APNEA_MAX_GAP_SEC):
                continue
            if gap < 2.5 * typical_gap:
                continue
            a = int(np.searchsorted(feat.times, prev.end))
            b = int(np.searchsorted(feat.times, nxt.start))
            if b <= a:
                continue
            gap_db = feat.db[a:b]
            baseline = float(np.percentile(gap_db, 10))
            spread_frac = float(np.mean(gap_db <= baseline + 6.0))
            med = float(np.median(gap_db))
            mad = float(np.median(np.abs(gap_db - med)))
            mad_frac = float(np.mean(
                np.abs(gap_db - med) < max(4.5, 2.5 * mad)))
            if max(spread_frac, mad_frac) < 0.7:
                continue
            win_lo, win_hi = nxt.start - 1.0, nxt.start + GASP_SEARCH_SEC
            for ev in candidates:
                if det.is_patient_cluster(ev.cluster):
                    continue
                if not (win_lo <= ev.start <= win_hi):
                    continue
                if ev in det.events:
                    ev.cluster = det.patient_cluster
                    ev.label = "patient"
                    ev.is_snore = True
                    det.cluster_ids[det.events.index(ev)] = det.patient_cluster
                else:
                    ev.cluster = det.patient_cluster
                    ev.label = "patient"
                    ev.is_snore = True
                    det.events.append(ev)
                    det.cluster_ids = np.append(det.cluster_ids, det.patient_cluster)
                fa = int(np.searchsorted(feat.times, ev.start))
                fb = int(np.searchsorted(feat.times, ev.end))
                det.snore_mask[fa:fb] = True
                reclaimed = True
        det.events.sort(key=lambda e: e.start)
        if not reclaimed:
            break


# ---------------------------------------------------------------------------
# Position handling (phone accelerometer sidecar)
# ---------------------------------------------------------------------------
def _normalise_position(p: str) -> str:
    p = (p or "unknown").lower().strip()
    aliases = {"back": "supine", "belly": "prone", "left side": "left",
               "right side": "right"}
    return aliases.get(p, p if p in POSITION_LABELS else "unknown")


def _position_per_second(meta: dict | None, n_sec: int) -> np.ndarray:
    labels = np.full(n_sec, "unknown", dtype=object)
    if not meta or not meta.get("positions"):
        return labels
    for seg in meta["positions"]:
        a = max(0, int(seg.get("start", 0)))
        b = min(n_sec, int(seg.get("end", n_sec)))
        pos = _normalise_position(str(seg.get("position", "unknown")))
        if b > a:
            labels[a:b] = pos
    return labels


# ---------------------------------------------------------------------------
# Stats builders
# ---------------------------------------------------------------------------
def _per_second(frame_times: np.ndarray, frame_values: np.ndarray, n_sec: int,
                reduce: str) -> np.ndarray:
    sec_idx = np.clip(frame_times.astype(np.int64), 0, n_sec - 1)
    out = np.full(n_sec, np.nan, dtype=np.float64)
    if reduce == "mean":
        counts = np.bincount(sec_idx, minlength=n_sec)
        sums = np.bincount(sec_idx, weights=frame_values.astype(np.float64), minlength=n_sec)
        mask = counts > 0
        out[mask] = sums[mask] / counts[mask]
    elif reduce == "any":
        out[:] = 0.0
        np.maximum.at(out, sec_idx, frame_values.astype(np.float64))
    return out


def _build_hourly(feat, det: DetectionResult, events, apneas, duration) -> list[dict]:
    n_hours = max(1, int(np.ceil(duration / 3600)))
    rows = []
    partner = [e for e in events if e.is_snore and e.label == "partner"]
    patient = [e for e in events if e.is_snore and e.label == "patient"]
    for h in range(n_hours):
        t0, t1 = h * 3600, min(duration, (h + 1) * 3600)
        f0 = np.searchsorted(feat.times, t0)
        f1 = np.searchsorted(feat.times, t1)
        rows.append({
            "hour": h,
            "patient_snores": sum(1 for e in patient if t0 <= e.start < t1),
            "partner_snores": sum(1 for e in partner if t0 <= e.start < t1),
            "apneas": sum(1 for a in apneas if t0 <= a.start < t1),
            "mean_db": round(float(feat.db[f0:f1].mean()), 1) if f1 > f0 else None,
            "ac_seconds": round(float(det.ac_mask[f0:f1].sum() / FRAMES_PER_SEC), 1),
        })
    return rows


def _build_position_stats(pos_per_sec, pat_sec, partner_sec, apneas) -> list[dict]:
    n_sec = pos_per_sec.size
    out = []
    for pos in dict.fromkeys(pos_per_sec.tolist()):
        m = pos_per_sec == pos
        seconds = int(m.sum())
        out.append({
            "position": pos,
            "position_label": POSITION_LABELS.get(pos, pos),
            "seconds": seconds,
            "pct": round(100.0 * seconds / max(1, n_sec), 1),
            "patient_snore_seconds": int(np.nansum(np.where(m, np.nan_to_num(pat_sec), 0))),
            "partner_snore_seconds": int(np.nansum(np.where(m, np.nan_to_num(partner_sec), 0))),
            "apnea_count": sum(
                1 for a in apneas
                if m[min(n_sec - 1, max(0, int(a.start)))]
            ),
        })
    return sorted(out, key=lambda r: -r["seconds"])


def _position_intervals(pos_per_sec: np.ndarray) -> list[dict]:
    if pos_per_sec.size == 0:
        return []
    changes = np.flatnonzero(np.r_[True, pos_per_sec[1:] != pos_per_sec[:-1]])
    intervals = []
    for i, a in enumerate(changes):
        b = changes[i + 1] if i + 1 < len(changes) else pos_per_sec.size
        intervals.append({
            "start": int(a), "end": int(b),
            "position": str(pos_per_sec[a]),
            "position_label": POSITION_LABELS.get(str(pos_per_sec[a]), str(pos_per_sec[a])),
        })
    return intervals


# ---------------------------------------------------------------------------
# Clip extraction
# ---------------------------------------------------------------------------
def _cut_clips(path: str, apneas, duration: float, job_id: str) -> list[dict]:
    from . import config
    order = sorted(apneas, key=lambda a: -a.duration)[:TOP_CLIPS]
    order.sort(key=lambda a: a.start)
    out_dir = config.CLIP_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, ap in enumerate(order, 1):
        t0 = max(0.0, ap.start - CLIP_PAD_SEC)
        t1 = min(duration, ap.end + CLIP_PAD_SEC)
        if t1 - t0 > CLIP_MAX_SEC:
            t1 = t0 + CLIP_MAX_SEC
        native, sr_in = wavio.read_wav_block(path, t0, t1 - t0)
        y = wavio.resample_linear(native, sr_in, TARGET_SR)
        name = f"clip_{i:02d}.wav"
        wavio.write_wav(out_dir / name, y, TARGET_SR)
        clips.append({
            "index": i,
            "start": round(t0, 3),
            "end": round(t1, 3),
            "apnea_duration": round(ap.duration, 3),
            "confirmed_by_gasp": ap.confirmed_by_gasp,
            "url": f"/api/clips/{job_id}/{name}",
        })
    return clips


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def analyze(path: str | Path, job_id: str | None = None,
            metadata: dict | None = None) -> dict:
    t_start = time.perf_counter()
    job_id = job_id or uuid.uuid4().hex[:12]
    path = str(path)
    info = wavio.wav_info(path)
    duration = info.duration

    parts = []
    start = 0.0
    while start < duration:
        block_end = min(duration, start + BLOCK_SECONDS)
        x, sr_in = wavio.read_wav_block(
            path, start, block_end - start + BLOCK_OVERLAP_SECONDS)
        y = wavio.resample_linear(x, sr_in, TARGET_SR)
        part = _block_features(y, start, block_end)
        if part is not None:
            parts.append(part)
        if block_end >= duration:
            break
        start = block_end

    if not parts:
        raise ValueError("录音内容为空，无法分析。")

    feat, _block_floors = _merge_blocks(parts)
    # Single night-wide threshold used for BOTH event detection and templates.
    noise_floor = adaptive_noise_floor(feat.db)

    # Second lightweight pass over the file: recompute spectra only to cache
    # whitened snore templates; FFT memory is released block by block.
    templates: list[tuple] = []
    start = 0.0
    while start < duration:
        block_end = min(duration, start + BLOCK_SECONDS)
        x, sr_in = wavio.read_wav_block(
            path, start, block_end - start + BLOCK_OVERLAP_SECONDS)
        y = wavio.resample_linear(x, sr_in, TARGET_SR)
        templates.extend(_extract_block_templates(y, start, block_end, noise_floor))
        if block_end >= duration:
            break
        start = block_end

    # Global event detection on stitched frame descriptors, then inject the
    # block-cached spectral templates for speaker clustering.
    det = detect_events(feat, noise_floor_override=noise_floor)
    _attach_templates(det.events, templates)
    # clustering needs templates on every snore event; re-run attribution
    # now that templates have been attached (detect_events ran without spectra)
    det.cluster_ids = _recluster(det.events)
    det.patient_clusters, det.n_clusters = _select_patient(det.events, det.cluster_ids)
    det.patient_cluster = det.patient_clusters[0] if det.patient_clusters else -1
    for ev, cid in zip(det.events, det.cluster_ids):
        ev.cluster = int(cid)
        if ev.is_snore:
            ev.label = ("patient" if cid in det.patient_clusters else "partner")

    _recover_gasp_events(feat, det, noise_floor)
    apneas = detect_apneas(feat, det)
    ahi = estimate_ahi(apneas, duration / 3600.0, det, feat.times.size)

    # ---------------- per-second envelope ----------------
    n_sec = max(1, int(np.ceil(duration)))
    db_sec = _per_second(feat.times, feat.db, n_sec, "mean")
    pat_sec_mask = np.zeros(feat.times.size, dtype=bool)
    part_sec_mask = np.zeros(feat.times.size, dtype=bool)
    for e in det.events:
        if not e.is_snore:
            continue
        a = np.searchsorted(feat.times, e.start)
        b = np.searchsorted(feat.times, e.end)
        if e.label == "patient":
            pat_sec_mask[a:b] = True
        elif e.label == "partner":
            part_sec_mask[a:b] = True
    pat_sec = _per_second(feat.times, pat_sec_mask.astype(np.float64), n_sec, "any")
    partner_sec = _per_second(feat.times, part_sec_mask.astype(np.float64), n_sec, "any")
    ac_sec = _per_second(feat.times, det.ac_mask.astype(np.float64), n_sec, "any")
    vad_sec = _per_second(feat.times, det.vad_mask.astype(np.float64), n_sec, "any")
    db_sec = np.where(np.isnan(db_sec), noise_floor, db_sec)

    pos_per_sec = _position_per_second(metadata, n_sec)
    pos_stats = _build_position_stats(pos_per_sec, pat_sec, partner_sec, apneas)

    patient_events = [e for e in det.events if e.is_snore and e.label == "patient"]
    partner_events = [e for e in det.events if e.is_snore and e.label == "partner"]
    tv_events = [e for e in det.events if e.label == "tv"]
    other_events = [e for e in det.events if e.label == "other"]

    clips = _cut_clips(path, apneas, duration, job_id)
    elapsed = time.perf_counter() - t_start

    return {
        "job_id": job_id,
        "duration": round(duration, 2),
        "sample_rate": TARGET_SR,
        "noise_floor_db": round(noise_floor, 1),
        "processing_seconds": round(elapsed, 2),
        "realtime_factor": round(duration / max(1e-6, elapsed), 1),
        "ahi": ahi,
        "counts": {
            "patient_snores": len(patient_events),
            "partner_snores": len(partner_events),
            "tv_events": len(tv_events),
            "other_events": len(other_events),
            "ac_seconds": round(det.ac_seconds, 1),
            "speakers_detected": det.n_clusters,
        },
        "envelope": {
            "seconds": n_sec,
            "db": [None if np.isnan(v) else round(float(v), 1) for v in db_sec],
            "patient_snore": [int(v) for v in pat_sec],
            "partner_snore": [int(v) for v in partner_sec],
            "ac": [int(v) for v in ac_sec],
            "voice_activity": [int(v) for v in vad_sec],
        },
        "hourly": _build_hourly(feat, det, det.events, apneas, duration),
        "positions": pos_stats,
        "position_intervals": _position_intervals(pos_per_sec),
        "apneas": [a.to_dict() for a in sorted(apneas, key=lambda x: -x.duration)],
        "clips": clips,
        "patient_events": [
            {"start": round(e.start, 2), "end": round(e.end, 2),
             "duration": round(e.duration, 2), "peak_db": round(e.peak_db, 1)}
            for e in patient_events
        ],
        "partner_events_preview": [
            {"start": round(e.start, 2), "end": round(e.end, 2),
             "peak_db": round(e.peak_db, 1)}
            for e in partner_events[:100]
        ],
    }


# clustering wrappers (kept local so detection.py stays the single source)
def _recluster(events):
    from .detection import _cluster_and_attribute
    return _cluster_and_attribute(events)


def _select_patient(events, cluster_ids):
    from .detection import find_patient_clusters
    return find_patient_clusters(events, cluster_ids)
