"""Minimal WAV I/O built on numpy alone.

Supports PCM 8/16/24/32-bit and IEEE-float 32/64-bit mono/multi-channel WAV,
streamed block reads so an 8-hour night never has to live in RAM at once.
Resampling uses vectorised linear interpolation (no scipy).
"""
from __future__ import annotations

import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import BLOCK_OVERLAP_SECONDS, BLOCK_SECONDS, TARGET_SR


@dataclass
class WavInfo:
    sample_rate: int
    n_samples: int          # per channel
    n_channels: int
    sampwidth: int          # bytes per sample
    is_float: bool
    data_offset: int        # byte offset of the data payload
    data_bytes: int

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate


def _parse_chunks(path: str | Path) -> WavInfo:
    """Parse RIFF/WAVE header without touching the (potentially huge) payload."""
    with open(path, "rb") as fh:
        riff = fh.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"{path}: not a RIFF/WAVE file")
        fmt_tag = None
        n_ch = sr = sw = None
        data_off = data_len = None
        while True:
            head = fh.read(8)
            if len(head) < 8:
                break
            cid, clen = struct.unpack("<4sI", head)
            body_off = fh.tell()
            if cid == b"fmt ":
                body = fh.read(clen)
                # WAVEFORMATEX: tag, channels, samples/sec, bytes/sec,
                # block align, bits per sample
                fmt_tag, n_ch, sr, _avg, _ba, bits = struct.unpack_from(
                    "<HHIIHH", body, 0)
                sw = bits // 8
                if fmt_tag == 0xFFFE and clen >= 24:  # EXTENSIBLE: real tag in GUID
                    fmt_tag = struct.unpack_from("<H", body, 24)[0]
                fh.seek(clen & 1, 1)
            elif cid == b"data":
                data_off = body_off
                data_len = clen
                fh.seek(clen + (clen & 1), 1)
            else:
                # chunks are word-aligned
                fh.seek(clen + (clen & 1), 1)
    if None in (fmt_tag, n_ch, sr, sw, data_off):
        raise ValueError(f"{path}: missing fmt/data chunk")
    frame_bytes = sw * n_ch
    n_samples = data_len // frame_bytes
    return WavInfo(
        sample_rate=sr,
        n_samples=n_samples,
        n_channels=n_ch,
        sampwidth=sw,
        is_float=(fmt_tag == 3),
        data_offset=data_off,
        data_bytes=n_samples * frame_bytes,
    )


def _decode_pcm(buf: bytes, sw: int, n_ch: int) -> np.ndarray:
    if sw == 1:  # 8-bit WAV is unsigned
        x = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sw == 2:
        x = np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(buf, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 3:
        raw = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        # sign-extend 24-bit two's complement
        x = np.where(x & 0x800000, x - 0x1000000, x).astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported PCM sample width: {sw} bytes")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x


def _decode_float(buf: bytes, sw: int, n_ch: int) -> np.ndarray:
    if sw == 4:
        x = np.frombuffer(buf, dtype="<f4").astype(np.float32)
    elif sw == 8:
        x = np.frombuffer(buf, dtype="<f8").astype(np.float32)
    else:
        raise ValueError(f"unsupported float sample width: {sw} bytes")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x


def read_wav_block(path: str | Path, start_sec: float, dur_sec: float) -> tuple[np.ndarray, int]:
    """Read a time range as mono float32 in [-1, 1] at the file's native rate."""
    info = _parse_chunks(path)
    s0 = max(0, int(start_sec * info.sample_rate))
    s1 = min(info.n_samples, s0 + int(dur_sec * info.sample_rate))
    if s1 <= s0:
        return np.zeros(0, dtype=np.float32), info.sample_rate
    off = info.data_offset + s0 * info.sampwidth * info.n_channels
    n_bytes = (s1 - s0) * info.sampwidth * info.n_channels
    with open(path, "rb") as fh:
        fh.seek(off)
        buf = fh.read(n_bytes)
    if info.is_float:
        x = _decode_float(buf, info.sampwidth, info.n_channels)
    else:
        x = _decode_pcm(buf, info.sampwidth, info.n_channels)
    # guard against clipped/garbage levels
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 1.5:
        x = x / peak
    return x, info.sample_rate


def wav_info(path: str | Path) -> WavInfo:
    return _parse_chunks(path)


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Vectorised linear-interpolation resampler (numpy only)."""
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32, copy=False)
    duration = x.size / sr_in
    n_out = int(round(duration * sr_out))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)
    pos = np.arange(n_out, dtype=np.float64) * (sr_in / sr_out)
    i0 = np.floor(pos).astype(np.int64)
    frac = (pos - i0).astype(np.float32)
    i0 = np.clip(i0, 0, x.size - 1)
    i1 = np.clip(i0 + 1, 0, x.size - 1)
    out = x[i0] * (1.0 - frac) + x[i1] * frac
    return out.astype(np.float32)


def stream_blocks(
    path: str | Path,
    block_sec: float = BLOCK_SECONDS,
    overlap_sec: float = BLOCK_OVERLAP_SECONDS,
    target_sr: int = TARGET_SR,
) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (mono block at target_sr, absolute start time in seconds).

    Blocks overlap so frame-level event detection never loses an event split
    across a block boundary; the analyser trims events in the overlap seam.
    """
    info = _parse_chunks(path)
    total = info.duration
    start = 0.0
    while start < total:
        end = min(total, start + block_sec)
        native, sr_in = read_wav_block(path, start, end - start + overlap_sec)
        y = resample_linear(native, sr_in, target_sr)
        yield y, start
        if end >= total:
            return
        start = end


def write_wav(path: str | Path, x: np.ndarray, sample_rate: int = TARGET_SR) -> None:
    """Write mono float32 (or int16) audio as 16-bit PCM WAV."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise ValueError("refusing to write an empty WAV")
    peak = float(np.max(np.abs(x)))
    if peak > 1.0:
        x = x / peak
    pcm = np.clip(x * 32767.0, -32768.0, 32767.0).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
