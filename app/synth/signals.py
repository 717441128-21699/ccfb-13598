"""Synthetic audio primitives built with numpy only.

These primitives model the acoustic components the screening pipeline has to
tell apart:

* coloured room-tone noise
* formant-shaped voiced bursts (snores / gasps), each "speaker" getting its
  own fixed formant pattern so clustering can separate patient from partner
* pitched TV speech-like chatter (higher centroid, irregular rhythm)
* a low, perfectly steady air-conditioner hum + rumble
"""
from __future__ import annotations

import numpy as np

SR = 16_000

# Distinctive voice profiles: (F0 Hz, formants Hz, formant gains)
PATIENT_VOICE = dict(f0=110, formants=(180, 420, 850), gains=(1.0, 0.55, 0.25))
PARTNER_VOICE = dict(f0=150, formants=(320, 700, 1300), gains=(1.0, 0.5, 0.3))
TV_VOICE = dict(f0=200, formants=(500, 1100, 2000), gains=(1.0, 0.6, 0.35))

_PHASE_CACHE: dict[str, np.ndarray] = {}


def _phase(n: int, tag: str, seed: int) -> np.ndarray:
    """Deterministic random phase spectrum per (tag, length, seed)."""
    key = (tag, n, seed)
    if key not in _PHASE_CACHE:
        rng = np.random.default_rng(seed)
        _PHASE_CACHE[key] = rng.uniform(0, 2 * np.pi, n // 2 + 1)
    return _PHASE_CACHE[key]


def silence(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def colored_noise(dur: float, level_db: float, seed: int = 0,
                  color: str = "brown") -> np.ndarray:
    """Room tone: brown/pink-ish noise shaped in the frequency domain."""
    n = int(dur * SR)
    if n <= 2:
        return silence(max(1, n))
    spec = np.exp(1j * _phase(n, f"noise-{color}", seed))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    if color == "brown":
        shape = 1.0 / np.maximum(freqs, 20.0) ** 1.2
    elif color == "pink":
        shape = 1.0 / np.maximum(freqs, 20.0) ** 0.7
    else:
        shape = np.ones_like(freqs)
    spec *= shape.astype(np.complex64)
    x = np.fft.irfft(spec, n=n).real.astype(np.float32)
    return _scale_db(x, level_db)


def formant_pulse(dur: float, voice: dict, level_db: float, seed: int,
                  breathy: bool = False) -> np.ndarray:
    """One voiced burst: harmonic source through a formant envelope.

    breathy=True widens the bandwidth and adds noise (gasp quality).
    """
    n = max(1, int(dur * SR))
    t = np.arange(n) / SR
    # gentle vibrato makes it biological rather than a pure oscillator
    jitter = 1.0 + 0.01 * np.sin(2 * np.pi * 5 * t + seed)
    phase = 2 * np.pi * voice["f0"] * np.cumsum(jitter) / SR
    src = np.sin(phase) + 0.45 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    if breathy:
        rng = np.random.default_rng(seed + 77)
        src = src * 0.5 + rng.standard_normal(n) * 0.9

    spec = np.fft.rfft(src.astype(np.float32))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    env = np.zeros_like(freqs)
    bw_mult = 1.8 if breathy else 1.0
    for fc, g in zip(voice["formants"], voice["gains"]):
        bw = fc * 0.18 * bw_mult
        env += g * np.exp(-0.5 * ((freqs - fc) / bw) ** 2)
    env = np.maximum(env, 1e-3)
    out = np.fft.irfft(spec * env, n=n).real.astype(np.float32)

    # amplitude envelope: fast attack, slower decay, slight tremolo
    attack = max(1, int(0.02 * SR))
    decay_tau = 0.09 if not breathy else 0.05
    env_t = np.ones(n, dtype=np.float32)
    env_t[:attack] = np.linspace(0, 1, attack)
    env_t *= np.exp(-t / decay_tau) * 0.7 + 0.3
    env_t *= 1.0 + 0.12 * np.sin(2 * np.pi * 6 * t)
    out *= env_t
    return _scale_db(out, level_db)


def snore(dur: float, voice: dict, level_db: float, seed: int) -> np.ndarray:
    """A snore = low formant pulse with a short nasal trailing rumble."""
    core = formant_pulse(dur * 0.8, voice, level_db, seed)
    rumble = colored_noise(dur * 0.4, level_db - 14, seed + 3, "brown")
    out = np.zeros(int(dur * SR), dtype=np.float32)
    out[: core.size] += core
    out[out.size - rumble.size:] += rumble * 0.5
    return _scale_db(out, level_db)


def gasp(dur: float, voice: dict, level_db: float, seed: int) -> np.ndarray:
    """Explosive recovery breath: broadband, mid-band, sharp attack."""
    n = int(dur * SR)
    rng = np.random.default_rng(seed + 404)
    noise = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    bp = np.exp(-0.5 * ((freqs - 700) / 500) ** 2)
    shaped = np.fft.irfft(spec * bp, n=n).real.astype(np.float32)
    t = np.arange(n) / SR
    attack = max(1, int(0.03 * SR))
    amp = np.ones(n, dtype=np.float32)
    amp[:attack] = np.linspace(0, 1, attack)
    amp *= np.exp(-t / (dur * 0.5))
    # add a voiced core so it carries the speaker's formant signature
    voiced = formant_pulse(dur, voice, level_db - 3, seed, breathy=True)
    out = shaped * amp * 0.8 + voiced * 0.6
    return _scale_db(out, level_db + 2)


def tv_utterance(dur: float, level_db: float, seed: int) -> np.ndarray:
    """Irregular string of short syllables with TV-like high formants."""
    n = int(dur * SR)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed + 909)
    t = 0.0
    k = 0
    while t < dur - 0.15:
        syl = rng.uniform(0.08, 0.28)
        gap = rng.uniform(0.05, 0.22)
        s = int(t * SR)
        e = min(n, s + int(syl * SR))
        if e > s:
            pulse = formant_pulse((e - s) / SR, TV_VOICE,
                                  level_db + rng.uniform(-4, 3), seed + k)
            m = min(e - s, pulse.size)
            out[s:s + m] += pulse[:m]
        t += syl + gap
        k += 1
    return _scale_db(out, level_db)


def ac_hum(dur: float, level_db: float, seed: int = 1) -> np.ndarray:
    """Air-conditioner: steady 120 Hz electrical hum + low rumble, no rhythm.

    Deliberately free of amplitude modulation beyond numerical jitter, so it
    models the constant-level field the AC detector is designed to catch.
    """
    n = int(dur * SR)
    t = np.arange(n) / SR
    hum = (np.sin(2 * np.pi * 120 * t)
           + 0.4 * np.sin(2 * np.pi * 240 * t)
           + 0.15 * np.sin(2 * np.pi * 360 * t)).astype(np.float32)
    rumble = colored_noise(dur, level_db - 6, seed + 5, "brown")
    out = hum * 0.5 + rumble * 0.7
    return _scale_db(out, level_db)


def add_at(track: np.ndarray, sig: np.ndarray, t_start: float) -> None:
    """Mix sig into track starting at t_start seconds (in-place, clipped)."""
    a = int(t_start * SR)
    if a >= track.size:
        return
    b = min(track.size, a + sig.size)
    track[a:b] += sig[: b - a]


def _scale_db(x: np.ndarray, level_db: float) -> np.ndarray:
    """Normalise a fragment to a target RMS level in dBFS."""
    if x.size == 0:
        return x
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    if rms < 1e-9 or not np.isfinite(rms):
        return x
    target = 10.0 ** (level_db / 20.0)
    return (x * (target / rms)).astype(np.float32)


def normalise_track(track: np.ndarray, peak_db: float = -3.0) -> np.ndarray:
    peak = float(np.max(np.abs(track)) + 1e-12)
    target = 10.0 ** (peak_db / 20.0)
    out = track * (target / peak)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
