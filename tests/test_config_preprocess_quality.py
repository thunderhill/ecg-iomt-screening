import numpy as np
import pytest

from ecg.config import (
    AppConfig,
    ConfigError,
    PreprocessConfig,
    QualityConfig,
    ScreeningConfig,
)
from ecg.preprocess import StreamingFilter, preprocess, validate_signal
from ecg.quality import assess_quality
from ecg.synthetic import generate_ecg


# ---------- configuration ----------

def test_default_config_is_valid():
    cfg = AppConfig()
    assert cfg.sample_rate_hz == 250.0
    assert cfg.preprocess.fs == 250.0


@pytest.mark.parametrize("fs", [0.0, -250.0])
def test_non_positive_sample_rate_rejected(fs):
    with pytest.raises(ConfigError):
        AppConfig(sample_rate_hz=fs)


def test_bandpass_above_nyquist_rejected():
    with pytest.raises(ConfigError, match="Nyquist"):
        PreprocessConfig(fs=100.0, bandpass_high_hz=60.0).validate()


def test_bandpass_order_of_bounds_rejected():
    with pytest.raises(ConfigError):
        PreprocessConfig(bandpass_low_hz=10.0, bandpass_high_hz=5.0).validate()


def test_powerline_must_be_50_or_60():
    with pytest.raises(ConfigError):
        PreprocessConfig(powerline_hz=44.0).validate()


def test_mode_must_be_known():
    with pytest.raises(ConfigError):
        PreprocessConfig(mode="fast").validate()


def test_screening_thresholds_must_be_ordered():
    with pytest.raises(ConfigError):
        ScreeningConfig(brady_bpm=120.0, tachy_bpm=100.0).validate()


def test_buffer_must_hold_one_rolling_window():
    with pytest.raises(ConfigError, match="rolling window"):
        AppConfig(rolling_window_s=60.0, buffer_max_samples=100)


def test_with_updates_keeps_preprocess_rate_in_sync():
    cfg = AppConfig().with_updates(sample_rate_hz=500.0)
    assert cfg.preprocess.fs == 500.0


# ---------- signal validation ----------

def test_validate_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_signal(np.array([]))


def test_validate_rejects_2d():
    with pytest.raises(ValueError, match="1-D"):
        validate_signal(np.zeros((4, 4)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_validate_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="NaN/inf"):
        validate_signal(np.array([0.0, bad, 1.0]))


def test_validate_rejects_short_signal():
    with pytest.raises(ValueError, match="need at least"):
        validate_signal(np.zeros(3), min_length=10)


# ---------- preprocessing ----------

def test_causal_preserves_length_and_is_finite():
    cfg = PreprocessConfig(fs=250.0)
    rec = generate_ecg(10.0, fs=250.0, seed=1)
    y = preprocess(rec.signal, cfg)
    assert y.shape == rec.signal.shape
    assert np.all(np.isfinite(y))


def test_zero_phase_has_no_lag_on_r_peaks():
    cfg = PreprocessConfig(fs=250.0, mode="zero_phase")
    rec = generate_ecg(20.0, fs=250.0, noise_std_mv=0.005, seed=2)
    y = preprocess(rec.signal, cfg)
    # Local maxima near each ground-truth R peak should sit within a few samples.
    shifts = []
    for idx in rec.r_peak_indices[2:-2]:
        window = y[idx - 10: idx + 11]
        shifts.append(int(np.argmax(window)) - 10)
    assert abs(float(np.median(shifts))) <= 2


def test_bandpass_removes_dc_offset():
    cfg = PreprocessConfig(fs=250.0)
    t = np.arange(0, 20, 1 / 250.0)
    x = 5.0 + 0.1 * np.sin(2 * np.pi * 10 * t)  # large DC offset + 10 Hz tone
    y = preprocess(x, cfg)
    assert abs(np.mean(y[500:])) < 0.01


def test_powerline_notch_attenuates_mains_tone():
    t = np.arange(0, 20, 1 / 250.0)
    mains = np.sin(2 * np.pi * 50 * t)
    clean = preprocess(mains, PreprocessConfig(fs=250.0))
    notched = preprocess(mains, PreprocessConfig(fs=250.0, powerline_hz=50.0))
    assert np.std(notched[1000:]) < 0.3 * np.std(clean[1000:])


def test_zero_phase_short_signal_rejected():
    with pytest.raises(ValueError):
        preprocess(np.zeros(5), PreprocessConfig(fs=250.0, mode="zero_phase"))


def test_streaming_matches_batch_causal():
    cfg = PreprocessConfig(fs=250.0)
    rec = generate_ecg(8.0, fs=250.0, seed=3)
    batch = preprocess(rec.signal, cfg)
    stream = StreamingFilter(cfg)
    chunks = np.array_split(rec.signal, 37)  # irregular packet sizes
    streamed = np.concatenate([stream.process(c) for c in chunks])
    np.testing.assert_allclose(streamed, batch, atol=1e-9)


def test_streaming_reset_restarts_state():
    cfg = PreprocessConfig(fs=250.0)
    f = StreamingFilter(cfg)
    f.process(np.ones(100))
    f.reset()
    assert f._zi is None


# ---------- quality ----------

def test_insufficient_data_when_too_short():
    x = np.zeros(250)  # 1 s, below the 5 s minimum
    q = assess_quality(x, fs=250.0, cfg=QualityConfig())
    assert q.status == "insufficient_data"


def test_flatline_is_poor():
    q = assess_quality(np.full(250 * 10, 0.2), fs=250.0, cfg=QualityConfig())
    assert q.status == "poor"
    assert any("flat" in r for r in q.reasons)


def test_lead_off_flag_is_poor():
    rec = generate_ecg(10.0, fs=250.0, seed=4)
    q = assess_quality(rec.signal, fs=250.0, cfg=QualityConfig(), lead_off=True)
    assert q.status == "poor" and q.lead_off


def test_clean_synthetic_is_not_poor():
    rec = generate_ecg(10.0, fs=250.0, seed=5)
    q = assess_quality(rec.signal, fs=250.0, cfg=QualityConfig())
    assert q.status in ("good", "fair")


def test_clipping_detected():
    rec = generate_ecg(10.0, fs=250.0, seed=6).signal.copy()
    rec[::3] = rec.max()  # many samples pinned to the rail
    q = assess_quality(rec, fs=250.0, cfg=QualityConfig())
    assert q.status == "poor"
    assert any("clipping" in r for r in q.reasons)


def test_non_finite_values_are_ignored_by_quality():
    rec = generate_ecg(10.0, fs=250.0, seed=7).signal.copy()
    rec[100] = np.nan
    q = assess_quality(rec, fs=250.0, cfg=QualityConfig())
    assert np.isfinite(q.std_mv)
