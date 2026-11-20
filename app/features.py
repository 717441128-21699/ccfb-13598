"""Frame-level acoustic features, vectorised with numpy.

A frame is 25 ms with a 10 ms hop (100 frames/s). For every frame we compute
the handful of descriptors the snore/gasp/noise classifiers need:

  * RMS level (dBFS)
  * zero-crossing rate
  * spectral centroid
  * spectral flatness (geometric / arithmetic mean of the power spectrum)
  * low-band energy fraction (snore energy lives roughly 60-500 Hz)
  * mid-band fraction used for gasp detection
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FRAME_LEN, HOP_LEN, TARGET_SR

_DB_EPS = 1e-10
_RFFT_BINS = FRAME_LEN // 2 + 1
_FREQS = np.fft.rfftfreq(FRAME_LEN, d=1.0 / TARGET_SR)
_WINDOW = np.hanning(FRAME_LEN).astype(np.float32)


@dataclass
class FrameFeatures:
    times: np.ndarray          # frame centre times, seconds
    db: np.ndarray             # dBFS per frame
    zcr: np.ndarray
    centroid: np.ndarray       # Hz
    flatness: np.ndarray       # 0..1
    low_frac: np.ndarray       # 60-500 Hz power fraction
    mid_frac: np.ndarray       # 500-2200 Hz power fraction
    mag: np.ndarray            # magnitude spectra (n_frames x rfft bins), float32

    def __len__(self) -> int:
        return self.db.size


def frame_signal(x: np.ndarray, frame_len: int = FRAME_LEN, hop: int = HOP_LEN) -> np.ndarray:
    """Build overlapping frames via stride tricks (zero-copy view)."""
    if x.size < frame_len:
        pad = np.zeros(frame_len - x.size, dtype=np.float32)
        x = np.concatenate([x, pad])
    n_frames = 1 + (x.size - frame_len) // hop
    if n_frames <= 0:
        return np.zeros((0, frame_len), dtype=np.float32)
    shape = (n_frames, frame_len)
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)


def _band_fraction(power: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    mask = (_FREQS >= lo_hz) & (_FREQS < hi_hz)
    total = power.sum(axis=1) + _DB_EPS
    return (power[:, mask].sum(axis=1) / total).astype(np.float32)


def compute_features(x: np.ndarray, sr: int = TARGET_SR, time_offset: float = 0.0) -> FrameFeatures:
    if sr != TARGET_SR:
        raise ValueError(f"expected {TARGET_SR} Hz, got {sr}")
    frames = frame_signal(x)
    if frames.shape[0] == 0:
        empty = np.zeros(0, dtype=np.float32)
        return FrameFeatures(empty, empty, empty, empty, empty, empty, empty,
                             np.zeros((0, _RFFT_BINS), dtype=np.float32))

    windowed = frames * _WINDOW
    mag = np.abs(np.fft.rfft(windowed, axis=1)).astype(np.float32)
    power = mag * mag

    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + _DB_EPS)
    db = (20.0 * np.log10(rms)).astype(np.float32)

    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1).astype(np.float32)

    mag_sum = mag.sum(axis=1) + _DB_EPS
    centroid = ((mag * _FREQS).sum(axis=1) / mag_sum).astype(np.float32)

    log_power = np.log(power + _DB_EPS)
    geo = np.exp(log_power.mean(axis=1))
    arith = power.mean(axis=1) + _DB_EPS
    flatness = np.clip(geo / arith, 0.0, 1.0).astype(np.float32)

    low_frac = _band_fraction(power, 60.0, 500.0)
    mid_frac = _band_fraction(power, 500.0, 2200.0)

    n_frames = frames.shape[0]
    times = (time_offset + (np.arange(n_frames) * HOP_LEN + FRAME_LEN / 2) / TARGET_SR
             ).astype(np.float32)

    return FrameFeatures(times=times, db=db, zcr=zcr, centroid=centroid,
                         flatness=flatness, low_frac=low_frac, mid_frac=mid_frac, mag=mag)
