"""Central configuration and tunable constants for the screening system.

All algorithmic thresholds live here instead of being scattered through the
pipeline, so the clinical owner can audit / retune them in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SLEEP_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
CLIP_DIR = DATA_DIR / "clips"
DB_PATH = DATA_DIR / "jobs.sqlite3"

for _d in (UPLOAD_DIR, CLIP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Audio / framing
# ---------------------------------------------------------------------------
TARGET_SR = 16_000          # phone recordings are resampled to this rate
FRAME_MS = 25.0
HOP_MS = 10.0
FRAME_LEN = int(TARGET_SR * FRAME_MS / 1000)     # 400 samples
HOP_LEN = int(TARGET_SR * HOP_MS / 1000)         # 160 samples
FRAMES_PER_SEC = 1000.0 / HOP_MS                 # 100

# Block size for streaming feature extraction (seconds of source audio per block).
# 300 s -> ~4.8M samples at 16 kHz float32 = ~19 MB per block.
BLOCK_SECONDS = 300
BLOCK_OVERLAP_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Voice activity / snore scoring
# ---------------------------------------------------------------------------
# Adaptive floor: noise estimate = this percentile of frame dB.
# Low percentile keeps a steady interferer (AC hum) from inflating the floor
# and drowning out the patient's quieter snores.
NOISE_FLOOR_PERCENTILE = 8.0
VAD_MARGIN_DB = 8.0                 # frame must be at least this above floor
VAD_ABS_FLOOR_DB = -55.0            # ...and above this absolute level
# Independent absolute gate for the steady-noise (AC) detector.
AC_ABS_GATE_DB = -52.0

# Frame-level snore classifier weights (logistic-style linear score).
SCORE_W_BAND = 1.2                  # low-frequency energy concentration
SCORE_W_FLAT = -1.0                 # spectral flatness (snore is tonal-ish)
SCORE_W_ZCR = -0.8                  # zero-crossing rate (low for snores)
SCORE_W_DUR = 0.6                   # filled-patch duration (handled at patch level)
SNORE_PATCH_ACCEPT = 0.15           # mean frame score required inside a patch
# Hard low-band gate: snore energy lives below 500 Hz (low_frac ~0.9+); TV and
# other voiced speech rarely exceed this even when they pass the shape score.
SNORE_MIN_LOW_FRAC = 0.85
# A snore must also be clearly audible above the room's noise floor; brown
# room-tone can otherwise pass the shape test and form a phantom cluster.
SNORE_MIN_MARGIN_DB = 10.0
# Frames whose level barely moves relative to their +/-0.25 s neighbourhood
# are steady interferer (AC/fan), not rhythmic snoring. A snore's modulation
# depth is ~10 dB; the hum's p99 deviation is ~5 dB.
SNORE_MIN_MODULATION_DB = 6.0

# Snore event geometry, seconds.
SNORE_MIN_DUR = 0.15
SNORE_MAX_DUR = 2.5
EVENT_MERGE_GAP = 0.12              # adjacent snore frames closer than this merge
# Fragments produced when re-segmenting a steady-hum-bridged patch are
# rejoined if separated by less than this (one physical snore's internal
# amplitude dip); stays well below the 3.6 s shortest breathing cycle.
RESEGMENT_MERGE_GAP = 0.6

# Gasp (post-apnea explosive breathing) geometry, seconds.
GASP_MIN_DUR = 0.25
GASP_MAX_DUR = 1.2
GASP_RISE_DB = 12.0                 # peak vs pre-gasp mean level
GASP_BAND_LO_HZ = 150.0
GASP_BAND_HI_HZ = 2200.0

# ---------------------------------------------------------------------------
# Steady interference (air-conditioner / fan) detector
# ---------------------------------------------------------------------------
AC_LO_HZ = 60.0
AC_HI_HZ = 600.0
AC_STABLE_DB_STD = 3.5              # frame-to-frame level stability
AC_MIN_RUN_SEC = 5.0
# Short dips inside an otherwise steady run are bridged; a run qualifies as
# AC only when at least this fraction of its frames pass the AC test.
AC_BRIDGE_SEC = 1.5
AC_RUN_COVERAGE = 0.75
AC_MIN_TOTAL_SEC = 8.0

# ---------------------------------------------------------------------------
# Speaker clustering (patient vs bed partner / TV)
# ---------------------------------------------------------------------------
MAX_CLUSTERS = 3
KMEANS_MAX_ITER = 60
# A cluster needs this many events before it can be the patient's snore cluster.
MIN_PATIENT_EVENTS = 8
# Patient cluster score weights (see classify_speakers).
PAT_W_EVENTS = 1.5
PAT_W_LEVEL = 1.5
PAT_W_PERIODIC = 1.0
PAT_W_BAND = 1.2

# ---------------------------------------------------------------------------
# Apnea / AHI
# ---------------------------------------------------------------------------
# A breathing pause is only apnea-like when it spans at least this many
# expected breathing cycles and lasts at least this long. AASM defines an
# apnea as >=10 s of airflow cessation; phone audio approximates it this way.
APNEA_MIN_GAP_SEC = 10.0
APNEA_MAX_GAP_SEC = 120.0
# Recovery window after the gap in which the gasp must appear.
GASP_SEARCH_SEC = 4.0
# Fallback: if no gasp is detected, the gap still counts when the room stays
# quiet (mean dB relative to noise floor below this) and patient snoring
# resumes afterwards.
FALLBACK_QUIET_MARGIN_DB = 6.0
# Minimum fraction of the recording that must contain patient-like activity,
# otherwise AHI is reported as low-confidence (recording problem / wrong side).
MIN_PATIENT_ACTIVE_FRACTION = 0.10

# AHI severity bands (events / hour), standard clinical thresholds.
AHI_SEVERE = 30.0
AHI_MODERATE = 15.0
AHI_MILD = 5.0

# Number of longest apnea clips to surface for the clinician.
TOP_CLIPS = 10
# Padding around each flagged gap when cutting a playable clip, seconds.
CLIP_PAD_SEC = 2.0
CLIP_MAX_SEC = 130.0

# ---------------------------------------------------------------------------
# Envelope decimation for the web UI (one sample per second).
# ---------------------------------------------------------------------------
ENVELOPE_HZ = 1

# ---------------------------------------------------------------------------
# API / jobs
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = 512
JOB_STATUSES = ("queued", "processing", "done", "error")
