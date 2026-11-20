"""Event detection and speaker attribution.

Pipeline per recording (operates on frame-level features):

1. Adaptive noise floor + voice-activity mask.
2. Frame-level snore score (linear combination of band / flatness / ZCR).
3. Connected-frame patches -> snore-shaped events; non-snore patches are kept
   as "other" events so they can explain time the patient is not snoring.
4. Steady low-band runs (air-conditioner / fan) are flagged separately.
5. K-means on spectral templates separates the patient's snore cluster from
   the bed partner's snores and the TV. The cluster nearest the microphone,
   loudest and most breathing-periodic wins as the patient.
6. Gasp detector used by the apnea stage to confirm recovery breathing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import (
    AC_HI_HZ, AC_LO_HZ, AC_MIN_RUN_SEC, AC_STABLE_DB_STD,
    EVENT_MERGE_GAP, FRAMES_PER_SEC, GASP_BAND_HI_HZ, GASP_BAND_LO_HZ,
    GASP_MAX_DUR, GASP_MIN_DUR, GASP_RISE_DB, GASP_SEARCH_SEC,
    KMEANS_MAX_ITER, MAX_CLUSTERS, MIN_PATIENT_EVENTS,
    NOISE_FLOOR_PERCENTILE, PAT_W_BAND, PAT_W_EVENTS, PAT_W_LEVEL,
    PAT_W_PERIODIC, RESEGMENT_MERGE_GAP, SCORE_W_BAND, SCORE_W_FLAT,
    SCORE_W_ZCR,
    SNORE_MAX_DUR, SNORE_MIN_DUR, SNORE_MIN_MARGIN_DB,
    SNORE_MIN_LOW_FRAC, SNORE_MIN_MODULATION_DB, SNORE_PATCH_ACCEPT,
    VAD_ABS_FLOOR_DB, VAD_MARGIN_DB,
)
from .features import FrameFeatures

# Speech-like frames are ruled out as snores when they look like TV/voices:
# high centroid and high flatness together.
SPEECH_CENTROID_HZ = 900.0
SPEECH_FLATNESS = 0.22


@dataclass
class Event:
    start: float
    end: float
    peak_db: float
    mean_db: float
    centroid: float
    low_frac: float
    mid_frac: float
    flatness: float
    zcr: float
    is_snore: bool
    label: str = "unknown"          # patient / partner / tv / ac / other
    cluster: int = -1
    template: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DetectionResult:
    events: list[Event]
    noise_floor_db: float
    ac_seconds: float
    snore_mask: np.ndarray          # boolean per frame: snore-like frame
    vad_mask: np.ndarray
    ac_mask: np.ndarray
    patient_cluster: int            # main patient cluster id (back-compat)
    patient_clusters: list[int]     # every cluster attributed to the patient
    cluster_ids: np.ndarray         # per event, -1 for non-clustered events
    n_clusters: int

    def is_patient_cluster(self, cid: int) -> bool:
        return cid in self.patient_clusters


# ---------------------------------------------------------------------------
# Frame masks
# ---------------------------------------------------------------------------
def adaptive_noise_floor(db: np.ndarray) -> float:
    if db.size == 0:
        return VAD_ABS_FLOOR_DB
    return float(np.percentile(db, NOISE_FLOOR_PERCENTILE))


def _ac_frame_mask(feat: FrameFeatures, noise_floor: float) -> np.ndarray:
    """Air-conditioner/fan: audible, low-band, extremely level frame-to-frame.

    Candidate frames are grouped into runs; short dips are bridged and a run
    is accepted only when a high fraction of its frames really satisfy the
    test, so rhythmic snoring can never be absorbed into the AC mask.
    """
    from .config import (
        AC_ABS_GATE_DB, AC_BRIDGE_SEC, AC_HI_HZ, AC_MIN_RUN_SEC,
        AC_MIN_TOTAL_SEC, AC_RUN_COVERAGE, AC_STABLE_DB_STD,
    )
    if feat.db.size == 0:
        return np.zeros(0, dtype=bool)
    active = feat.db > AC_ABS_GATE_DB
    low = feat.centroid < AC_HI_HZ
    win = 25
    kernel = np.ones(win, dtype=np.float32) / win
    pad = win // 2
    padded = np.pad(feat.db, (pad, pad), mode="edge")
    local_mean = np.convolve(padded, kernel, mode="valid")[: feat.db.size]
    stable = np.abs(feat.db - local_mean) < AC_STABLE_DB_STD
    candidate = active & low & stable

    out = np.zeros_like(candidate)
    bridge = max(1, int(AC_BRIDGE_SEC * FRAMES_PER_SEC))
    min_run = int(AC_MIN_RUN_SEC * FRAMES_PER_SEC)
    idx = np.flatnonzero(np.diff(np.r_[False, candidate, False]))
    runs = idx.reshape(-1, 2)
    # merge runs separated by gaps no longer than the bridge window
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= bridge:
            merged[-1][1] = b
        else:
            merged.append([int(a), int(b)])
    for a, b in merged:
        if b - a < min_run:
            continue
        coverage = float(candidate[a:b].mean())
        if coverage >= AC_RUN_COVERAGE and (b - a) / FRAMES_PER_SEC >= AC_MIN_TOTAL_SEC:
            out[a:b] = True
    return out


def frame_snore_score(feat: FrameFeatures, noise_floor: float) -> np.ndarray:
    """Per-frame score; >0 looks like a snore, <0 like speech/noise/quiet."""
    # standardise descriptors against sensible fixed scales so the weights
    # stay interpretable and tuning does not depend on per-file statistics.
    # low_frac baseline at 0.55: real snores sit at 0.9+ low-band energy while
    # TV/voice syllables hover around 0.7-0.8, so the band term separates them
    band = np.clip((feat.low_frac - 0.55) / 0.3, -1.0, 1.0)
    flat = np.clip((feat.flatness - 0.1) / 0.2, -1.0, 1.0)
    zcr = np.clip((feat.zcr - 0.15) / 0.15, -1.0, 1.0)
    score = SCORE_W_BAND * band + SCORE_W_FLAT * flat + SCORE_W_ZCR * zcr
    return score.astype(np.float32)


# ---------------------------------------------------------------------------
# Patches -> events
# ---------------------------------------------------------------------------
def _extract_patches(mask: np.ndarray, merge_gap_frames: int) -> np.ndarray:
    """Return (n, 2) array of [start_frame, end_frame_exclusive)."""
    if mask.size == 0 or not mask.any():
        return np.zeros((0, 2), dtype=np.int64)
    idx = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    runs = idx.reshape(-1, 2)
    if runs.shape[0] <= 1 or merge_gap_frames <= 0:
        return runs
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= merge_gap_frames:
            merged[-1][1] = b
        else:
            merged.append(np.array([a, b]))
    return np.asarray(merged, dtype=np.int64)


def _safe_mean(a: np.ndarray, default: float = 0.0) -> float:
    return float(a.mean()) if a.size else default


def _patch_to_event(feat: FrameFeatures, a: int, b: int, is_snore: bool,
                    mag: np.ndarray | None) -> Event:
    t0 = float(feat.times[a])
    t1 = float(feat.times[min(b - 1, feat.times.size - 1)]) + 1.0 / FRAMES_PER_SEC
    template = np.zeros(0, np.float32)
    if mag is not None and mag.shape[1] > 0:
        spec = mag[a:b].mean(axis=0)
        n = np.linalg.norm(spec) + 1e-9
        template = (spec / n).astype(np.float32)
    return Event(
        start=t0, end=t1,
        peak_db=float(feat.db[a:b].max()),
        mean_db=float(feat.db[a:b].mean()),
        centroid=_safe_mean(feat.centroid[a:b]),
        low_frac=_safe_mean(feat.low_frac[a:b]),
        mid_frac=_safe_mean(feat.mid_frac[a:b]),
        flatness=_safe_mean(feat.flatness[a:b]),
        zcr=_safe_mean(feat.zcr[a:b]),
        is_snore=is_snore,
        template=template,
    )


def detect_events(feat: FrameFeatures,
                  noise_floor_override: float | None = None) -> DetectionResult:
    n = feat.db.size
    if n == 0:
        empty = np.zeros(0, dtype=bool)
        return DetectionResult([], VAD_ABS_FLOOR_DB, 0.0, empty, empty, empty, -1,
                               [], np.zeros(0, dtype=np.int64), 0)

    noise_floor = (adaptive_noise_floor(feat.db)
                   if noise_floor_override is None else noise_floor_override)
    vad = feat.db > np.maximum(noise_floor + VAD_MARGIN_DB, VAD_ABS_FLOOR_DB)
    ac_mask = _ac_frame_mask(feat, noise_floor)

    # Per-frame rhythmic modulation, used only to re-segment the rare
    # over-long patches a steady hum creates (normal snores are <2.5 s).
    win = 25
    pad = win // 2
    padded_db = np.pad(feat.db, (pad, pad), mode="edge")
    indexer = np.arange(n)[:, None] + np.arange(win)[None, :]
    windowed = padded_db[indexer]
    local_range = windowed.max(axis=1) - windowed.min(axis=1)
    modulated = local_range >= SNORE_MIN_MODULATION_DB

    # Snore detection runs independently of the AC mask; over-long steady
    # patches are re-segmented below instead of being frame-gated here.
    score = frame_snore_score(feat, noise_floor)
    speech_like = (feat.centroid > SPEECH_CENTROID_HZ) & (feat.flatness > SPEECH_FLATNESS)
    # hard low-band gate rejects higher-pitched TV/voice syllables that the
    # linear score alone lets through
    low_voiced = feat.low_frac >= SNORE_MIN_LOW_FRAC
    snore_frames = vad & ~speech_like & low_voiced & (score > SNORE_PATCH_ACCEPT)

    merge_frames = int(EVENT_MERGE_GAP * FRAMES_PER_SEC)
    patches = _extract_patches(snore_frames, merge_frames)
    min_dur_f = max(1, int(SNORE_MIN_DUR * FRAMES_PER_SEC))
    max_dur_f = int(SNORE_MAX_DUR * FRAMES_PER_SEC)

    events: list[Event] = []
    keep_snore_mask = np.zeros(n, dtype=bool)
    for a0, b0 in patches:
        sub_patches = [(int(a0), int(b0))]
        # Over-long patches only happen when a steady hum bridges real snores.
        # Re-cut along the modulation signal, then rejoin fragments of what
        # is physically one snore (close internal dips) with a dedicated gap.
        if b0 - a0 > max_dur_f:
            mod_runs = _extract_patches(modulated[a0:b0], merge_gap_frames=merge_frames)
            cuts = [(a0 + int(ra), a0 + int(rb)) for ra, rb in mod_runs
                    if rb - ra >= min_dur_f]
            if cuts:
                frag_merge = int(RESEGMENT_MERGE_GAP * FRAMES_PER_SEC)
                merged = [cuts[0]]
                for fa, fb in cuts[1:]:
                    if fa - merged[-1][1] <= frag_merge:
                        merged[-1] = (merged[-1][0], fb)
                    else:
                        merged.append((fa, fb))
                sub_patches = merged
        for a, b in sub_patches:
            dur_f = b - a
            if dur_f < min_dur_f:
                continue
            # reject room-tone patches that only barely clear the VAD margin
            if float(feat.db[a:b].mean()) < noise_floor + SNORE_MIN_MARGIN_DB:
                continue
            is_snore = dur_f <= max_dur_f
            if not is_snore:
                # still over-long after re-segmentation: treat as other, never
                # as a snore (it is an uninterrupted steady-noise run)
                ev = _patch_to_event(feat, a, b, False, feat.mag)
                events.append(ev)
                continue
            ev = _patch_to_event(feat, a, b, True, feat.mag)
            keep_snore_mask[a:b] = True
            events.append(ev)

    # "Other" audible patches (TV speech, bangs, ...): explain occupied time.
    # AC-only frames are excluded so the hum is not relabelled as events.
    other_mask = vad & ~keep_snore_mask & ~ac_mask
    for a, b in _extract_patches(other_mask, merge_frames):
        if b - a < min_dur_f:
            continue
        ev = _patch_to_event(feat, int(a), int(b), False, feat.mag)
        ev.label = "tv" if ev.centroid > SPEECH_CENTROID_HZ and ev.flatness > SPEECH_FLATNESS else "other"
        events.append(ev)

    events.sort(key=lambda e: e.start)
    cluster_ids = _cluster_and_attribute(events)
    patient_clusters, n_clusters = find_patient_clusters(events, cluster_ids)
    patient_cluster = patient_clusters[0] if patient_clusters else -1

    for ev, cid in zip(events, cluster_ids):
        ev.cluster = int(cid)
        if ev.is_snore:
            ev.label = "patient" if cid in patient_clusters else "partner"
        elif ev.label == "unknown":
            ev.label = "other"

    return DetectionResult(
        events=events,
        noise_floor_db=noise_floor,
        ac_seconds=float(ac_mask.sum() / FRAMES_PER_SEC),
        snore_mask=keep_snore_mask,
        vad_mask=vad,
        ac_mask=ac_mask,
        patient_cluster=patient_cluster,
        patient_clusters=patient_clusters,
        cluster_ids=cluster_ids,
        n_clusters=n_clusters,
    )


# ---------------------------------------------------------------------------
# K-means clustering + patient attribution
# ---------------------------------------------------------------------------
def _build_cluster_vectors(events: list[Event], idx: list[int]) -> np.ndarray:
    """Unit-norm feature vectors combining whitened spectrum with acoustics.

    The spectral block (whitened, ~201 dims) carries timbre; three extra
    dimensions encode level, spectral centroid and low-band fraction so a
    near-mic patient and a far bed partner separate even when their whitened
    spectra happen to correlate highly.
    """
    specs = np.stack([events[i].template for i in idx])
    levels = np.array([events[i].mean_db for i in idx], dtype=np.float32)
    cents = np.array([events[i].centroid for i in idx], dtype=np.float32)
    lows = np.array([events[i].low_frac for i in idx], dtype=np.float32)

    def z(v: np.ndarray) -> np.ndarray:
        sd = v.std()
        return (v - v.mean()) / (sd + 1e-6)

    # acoustic block gets ~10% of total vector weight; spectrum dominates
    acoustic = np.stack([z(levels), z(cents) / 2.0, z(lows) / 2.0], axis=1) * 2.5
    X = np.concatenate([specs, acoustic], axis=1).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X


def _cluster_and_attribute(events: list[Event]) -> np.ndarray:
    ids = np.full(len(events), -1, dtype=np.int64)
    idx = [i for i, e in enumerate(events) if e.is_snore and e.template.size > 0]
    if not idx:
        return ids
    X = _build_cluster_vectors(events, idx)
    # Use the full cluster budget whenever there are enough events for each
    # cluster to be statistically viable; over-splitting one speaker is later
    # repaired by _merge_similar_clusters / find_patient_clusters, while
    # under-splitting irreversibly mixes two real speakers into one cluster.
    viable_k = max(1, len(idx) // (MIN_PATIENT_EVENTS * 2))
    k = min(MAX_CLUSTERS, viable_k)
    if k == 1 or len(idx) < 2 * MIN_PATIENT_EVENTS:
        for j in idx:
            ids[j] = 0
        return ids

    labels, centers = _kmeans_cosine(X, k)
    labels = _merge_similar_clusters(labels, centers)
    for j, lab in zip(idx, labels):
        ids[j] = int(lab)
    return ids


def _merge_similar_clusters(labels: np.ndarray, centers: np.ndarray,
                            merge_sim: float = 0.92) -> np.ndarray:
    """Greedy merge of cluster centroids that are nearly identical.

    Same-speaker snore templates correlate highly even when amplitude jitter
    splits them; bed-partner / TV templates sit in different formant regions
    and stay well below the cosine-similarity threshold.
    """
    n = centers.shape[0]
    remap = np.arange(n, dtype=np.int64)
    for a in range(n):
        ra = remap[a]
        for b in range(a + 1, n):
            rb = remap[b]
            if ra == rb:
                continue
            sim = float(np.dot(centers[a], centers[b]))
            if sim >= merge_sim:
                remap[remap == rb] = ra
    # compact label ids to 0..k-1
    uniq = {old: new for new, old in enumerate(sorted(set(remap.tolist())))}
    return np.array([uniq[int(remap[l])] for l in labels], dtype=np.int64)


def _kmeans_cosine(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """K-means on L2-normalised templates == spherical (cosine) k-means."""
    n = X.shape[0]
    # deterministic seeding: spread initial centroids through the data
    rng = np.random.default_rng(seed=20260520)
    centers = X[np.linspace(0, n - 1, k).astype(np.int64)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(KMEANS_MAX_ITER):
        sim = X @ centers.T                       # rows already unit-norm
        new_labels = sim.argmax(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = X[labels == c]
            if members.shape[0] == 0:
                centers[c] = X[rng.integers(0, n)]
                continue
            centroid = members.mean(axis=0)
            centers[c] = centroid / (np.linalg.norm(centroid) + 1e-12)
    sim = X @ centers.T
    labels = sim.argmax(axis=1)
    return labels, centers


def _periodicity_score(evs: list[Event]) -> float:
    """Breathing-like rhythm: robust coefficient of variation of inter-event
    gaps, ignoring the small number of long apnea gaps.

    Low median-based CV (regular gaps) -> high score in [0, 1].
    """
    if len(evs) < 4:
        return 0.0
    starts = np.sort(np.array([e.start for e in evs], dtype=np.float64))
    gaps = np.diff(starts)
    gaps = gaps[gaps > 0.05]
    if gaps.size < 3:
        return 0.0
    med = float(np.median(gaps))
    if med <= 0:
        return 0.0
    # robust spread: median absolute deviation, scaled to sigma-equivalent
    mad = float(np.median(np.abs(gaps - med))) * 1.4826
    rcv = mad / med
    return float(np.clip(1.1 - 2.2 * rcv, 0.0, 1.0))


def _cluster_stats(events: list[Event], cluster_ids: np.ndarray,
                   n_clusters: int) -> list[dict]:
    snore_idx = [i for i, e in enumerate(events) if e.is_snore]
    all_levels = np.array([events[i].mean_db for i in snore_idx])
    level_span = max(1e-3, float(all_levels.max() - all_levels.min()))
    level_mid = float(all_levels.mean())
    total_snores = max(1, len(snore_idx))
    stats = []
    for c in range(n_clusters):
        members = [events[i] for i in snore_idx if cluster_ids[i] == c]
        count = len(members)
        if count == 0:
            stats.append({"cluster": c, "score": -1e9, "count": 0,
                          "size_frac": 0.0, "level": -1e9, "periodic": 0.0,
                          "band": 0.0, "centroid": 1e9})
            continue
        level = float(np.mean([e.mean_db for e in members]))
        periodic = _periodicity_score(members)
        band = float(np.mean([e.low_frac for e in members]))
        centroid = float(np.mean([e.centroid for e in members]))
        level_n = (level - level_mid) / level_span
        size_frac = count / total_snores
        score = (
            PAT_W_EVENTS * size_frac
            + PAT_W_LEVEL * level_n
            + PAT_W_PERIODIC * periodic
            + PAT_W_BAND * np.clip(band - 0.35, -1.0, 1.0)
        )
        stats.append({"cluster": c, "score": float(score), "count": count,
                      "size_frac": size_frac, "level": level,
                      "periodic": periodic, "band": band, "centroid": centroid})
    return stats


def _find_patient_cluster(events: list[Event], cluster_ids: np.ndarray) -> tuple[int, int]:
    """Backward-compatible single-id selector (main patient cluster)."""
    patient, k = find_patient_clusters(events, cluster_ids)
    return (patient[0] if patient else -1), k


def find_patient_clusters(events: list[Event],
                          cluster_ids: np.ndarray) -> tuple[list[int], int]:
    """Return ALL clusters that belong to the patient.

    One speaker can be split into several spectral clusters when a steady
    interferer (AC) changes the background spectrum through the night. The
    main cluster is chosen on loudness / low-frequency content / rhythm, then
    further clusters are merged in when they look like the same near-mic
    speaker rather than a separate (farther, higher-pitched) voice.
    """
    snore_idx = [i for i, e in enumerate(events) if e.is_snore]
    if not snore_idx:
        return [], 0
    n_clusters = int(cluster_ids[snore_idx].max()) + 1
    if n_clusters == 1:
        return [0], 1

    stats = _cluster_stats(events, cluster_ids, n_clusters)
    populated = [s for s in stats if s["count"] > 0]
    if not populated:
        return [], n_clusters
    max_size = max(s["count"] for s in populated)
    eligible = [
        s for s in populated
        if s["count"] >= MIN_PATIENT_EVENTS
        and (s["size_frac"] >= 0.4 or s["count"] * 4 >= max_size)
    ]
    if not eligible:
        substantial = [s for s in populated if s["count"] >= MIN_PATIENT_EVENTS]
        if not substantial:
            return [], n_clusters
        main = max(substantial, key=lambda s: s["score"])
    else:
        main = max(eligible, key=lambda s: s["score"])

    patient = [main["cluster"]]
    chosen_members = [e for e in events
                      if e.is_snore and cluster_ids[events.index(e)] == main["cluster"]]
    for s in stats:
        c = s["cluster"]
        if c == main["cluster"] or s["count"] < MIN_PATIENT_EVENTS:
            continue
        members = [e for e in events
                   if e.is_snore and cluster_ids[events.index(e)] == c]
        mean_dur = float(np.mean([e.duration for e in members]))
        # Same near-mic speaker: not quieter than the main voice, similarly
        # low-frequency, and either rhythmic or large. A bed partner's voice
        # is farther (quieter) and higher-pitched; TV is higher still.
        same_level = s["level"] >= main["level"] - 4.0
        same_band = abs(s["centroid"] - main["centroid"]) < 350.0
        rhythmic_or_big = s["periodic"] >= 0.35 or s["count"] * 2 >= main["count"]
        # Gasp cluster: loud, short broadband recovery breaths from the same
        # speaker. They are higher-pitched than routine snores but always
        # short and at least as loud as the main voice.
        gasp_like_cluster = (
            s["level"] >= main["level"] - 2.0 and mean_dur <= 2.2
            and s["centroid"] > main["centroid"]
        )
        # Background-tinted twin: the same speaker's weak snore fragments seen
        # through an AC hum form a second cluster. The discriminator against a
        # real second speaker is microphone distance: a twin stays within ~6 dB
        # of the main voice, while a bed partner across the bed is 10+ dB fainter
        # regardless of how rhythmic either one is.
        merged_score = _periodicity_score(chosen_members + members)
        main_periodic = _periodicity_score(chosen_members)
        twin_cluster = (
            merged_score >= 0.55 * max(0.4, main_periodic)
            and s["level"] >= main["level"] - 6.0
            and s["band"] >= 0.55
        )
        if ((same_level and same_band and rhythmic_or_big)
                or gasp_like_cluster or twin_cluster):
            patient.append(c)
            chosen_members += members
    return sorted(patient), n_clusters


# ---------------------------------------------------------------------------
# Gasp detection (called by the apnea stage on specific search windows)
# ---------------------------------------------------------------------------
def find_gasp(feat: FrameFeatures, win_start: float, win_end: float,
              pre_level: float, ac_mask: np.ndarray) -> float | None:
    """Return gasp peak time inside [win_start, win_end], or None.

    A gasp is a short, relatively broadband burst whose level jumps sharply
    over the quiet pre-apnea baseline.
    """
    lo = max(0, int(np.searchsorted(feat.times, win_start)))
    hi = min(feat.times.size, int(np.searchsorted(feat.times, win_end)))
    if hi - lo < int(GASP_MIN_DUR * FRAMES_PER_SEC):
        return None
    db = feat.db[lo:hi]
    mid = feat.mid_frac[lo:hi]
    zcr = feat.zcr[lo:hi]
    ac = ac_mask[lo:hi] if ac_mask.size == feat.db.size else np.zeros(hi - lo, bool)

    burst = (db > pre_level + GASP_RISE_DB) & (mid > 0.18) & (zcr < 0.35) & ~ac
    if not burst.any():
        return None
    runs = _extract_patches(burst, merge_gap_frames=2)
    min_f = max(1, int(GASP_MIN_DUR * FRAMES_PER_SEC))
    max_f = int(GASP_MAX_DUR * FRAMES_PER_SEC)
    best_t, best_db = None, -1e9
    for a, b in runs:
        if not (min_f <= b - a <= max_f):
            continue
        peak = float(db[a:b].max())
        if peak > best_db:
            best_db = peak
            best_t = float(feat.times[lo + a + int((b - a) / 2)])
    return best_t


def gasp_search_window() -> float:
    return GASP_SEARCH_SEC


# ---------------------------------------------------------------------------
# Gasp-shaped patches (used to reclaim post-apnea recovery breaths that the
# routine snore classifier rejects because of their broadband content)
# ---------------------------------------------------------------------------
def gasp_like_events(feat: FrameFeatures, noise_floor: float,
                     ac_mask: np.ndarray) -> list[Event]:
    """Short mid-band bursts with a sharp level rise over the local floor."""
    if feat.db.size == 0:
        return []
    active = feat.db > noise_floor + GASP_RISE_DB
    burst = active & (feat.mid_frac > 0.14) & (feat.zcr < 0.4) & ~ac_mask
    merge_f = max(1, int(0.05 * FRAMES_PER_SEC))
    min_f = max(1, int(GASP_MIN_DUR * FRAMES_PER_SEC))
    max_f = int(GASP_MAX_DUR * FRAMES_PER_SEC)
    out: list[Event] = []
    for a, b in _extract_patches(burst, merge_f):
        if not (min_f <= b - a <= max_f):
            continue
        ev = _patch_to_event(feat, int(a), int(b), is_snore=False, mag=None)
        ev.label = "gasp_candidate"
        out.append(ev)
    return out
