"""Unit tests for the numpy-only audio primitives and WAV I/O."""
from __future__ import annotations

import numpy as np

from app import wavio
from app.config import TARGET_SR
from app.features import compute_features
from app.synth import signals as S
from app.synth.scenarios import SCENARIOS, generate_scenario


def test_colored_noise_shape_and_level():
    x = S.colored_noise(2.0, -30.0, seed=1)
    assert x.shape == (2 * S.SR,)
    rms_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
    assert abs(rms_db - (-30.0)) < 1.0


def test_snore_is_low_frequency_and_burst_like():
    x = S.snore(0.6, S.PATIENT_VOICE, -24.0, seed=3)
    feat = compute_features(x)
    # a snore concentrates energy below 500 Hz and is non-flat (voiced)
    loud = feat.db > feat.db.max() - 6.0
    assert loud.any()
    assert feat.low_frac[loud].mean() > 0.5
    assert feat.flatness[loud].mean() < 0.3


def test_gasp_is_broader_band_than_snore():
    sn = S.snore(0.6, S.PATIENT_VOICE, -24.0, seed=5)
    gp = S.gasp(0.55, S.PATIENT_VOICE, -18.0, seed=5)
    f_sn = compute_features(sn)
    f_gp = compute_features(gp)
    peak_sn = f_sn.db.argmax()
    peak_gp = f_gp.db.argmax()
    assert f_gp.centroid[peak_gp] > f_sn.centroid[peak_sn]
    assert f_gp.mid_frac[peak_gp] > f_sn.mid_frac[peak_sn]


def test_ac_hum_is_spectrally_steady():
    x = S.ac_hum(10.0, -40.0)
    feat = compute_features(x)
    # frame-to-frame level variation stays small across a long steady run
    assert feat.db.std() < 4.0
    assert feat.centroid.mean() < 500.0


def test_two_voices_have_distinct_formants():
    a = S.snore(0.6, S.PATIENT_VOICE, -24.0, seed=2)
    b = S.snore(0.6, S.PARTNER_VOICE, -33.0, seed=2)
    fa, fb = compute_features(a), compute_features(b)
    assert fb.centroid[fb.db.argmax()] > fa.centroid[fa.db.argmax()]


def test_wav_roundtrip_preserves_duration(tmp_path):
    x = S.colored_noise(3.0, -30.0, seed=9)
    p = tmp_path / "rt.wav"
    wavio.write_wav(p, x, S.SR)
    info = wavio.wav_info(p)
    assert abs(info.duration - 3.0) < 0.05
    y, sr = wavio.read_wav_block(p, 0.0, 3.0)
    assert sr == S.SR
    assert y.shape == x.shape


def test_resample_linear_keeps_duration():
    sr_in = 8000
    x = np.sin(2 * np.pi * 220 * np.arange(sr_in) / sr_in).astype(np.float32)
    y = wavio.resample_linear(x, sr_in, TARGET_SR)
    assert y.shape[0] == TARGET_SR
    # resynthesis stays finite and bounded for a pure tone
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.01


def test_every_scenario_builds_with_ground_truth():
    for scen in SCENARIOS:
        track, gt = generate_scenario(scen, duration=300.0, seed=7)
        assert track.ndim == 1 and track.size == int(300 * S.SR)
        assert np.all(np.isfinite(track))
        assert np.max(np.abs(track)) <= 1.0
        assert gt.scenario == scen
        assert gt.positions and all("position" in p for p in gt.positions)
