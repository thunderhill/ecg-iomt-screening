import numpy as np
import pytest

from ecg.config import HRVConfig, PreprocessConfig, RPeakConfig, RRPlausibility
from ecg.detection_eval import evaluate_detection, match_peaks
from ecg.hr import FLAG_OUT_OF_RANGE, FLAG_RELATIVE_JUMP, FLAG_SPANS_GAP, heart_rate_summary, rr_intervals
from ecg.hrv import time_domain_hrv
from ecg.preprocess import preprocess
from ecg.rpeak import detect_rpeaks
from ecg.synthetic import generate_ecg


# ---------- R-peak detection ----------

def test_detection_on_synthetic_meets_basic_accuracy():
    rec = generate_ecg(60.0, fs=250.0, hr_bpm=70, hr_std_bpm=1.5, noise_std_mv=0.02, seed=10)
    filtered = preprocess(rec.signal, PreprocessConfig(fs=250.0, mode="zero_phase"))
    res = detect_rpeaks(filtered, 250.0, zero_phase=True)
    scores = evaluate_detection(rec.r_peak_indices, res.indices, fs=250.0)
    # Synthetic data is easy: this is a regression guard, not a clinical claim.
    assert scores.sensitivity > 0.95
    assert scores.ppv > 0.95


def test_causal_detection_is_close_to_zero_phase():
    rec = generate_ecg(60.0, fs=250.0, seed=11)
    filtered = preprocess(rec.signal, PreprocessConfig(fs=250.0, mode="causal"))
    res = detect_rpeaks(filtered, 250.0, zero_phase=False)
    scores = evaluate_detection(rec.r_peak_indices, res.indices, fs=250.0)
    assert scores.f1 > 0.9


def test_refractory_period_respected():
    rec = generate_ecg(30.0, fs=250.0, seed=12)
    res = detect_rpeaks(rec.signal, 250.0)
    if len(res) > 1:
        assert np.min(np.diff(res.indices)) >= int(0.2 * 250)


def test_indices_and_times_consistent():
    rec = generate_ecg(20.0, fs=250.0, seed=13)
    res = detect_rpeaks(rec.signal, 250.0)
    np.testing.assert_allclose(res.times_s, res.indices / 250.0)


def test_short_segment_returns_empty_with_note():
    res = detect_rpeaks(np.random.default_rng(0).normal(size=100), 250.0)
    assert len(res) == 0
    assert res.notes


def test_flat_signal_returns_empty():
    res = detect_rpeaks(np.zeros(5000), 250.0)
    assert len(res) == 0


def test_low_sampling_rate_rejected():
    with pytest.raises(ValueError):
        detect_rpeaks(np.zeros(1000), 20.0)


def test_non_finite_input_rejected():
    x = np.zeros(2000)
    x[5] = np.nan
    with pytest.raises(ValueError):
        detect_rpeaks(x, 250.0)


def test_confidence_in_unit_interval():
    rec = generate_ecg(30.0, fs=250.0, seed=14)
    res = detect_rpeaks(rec.signal, 250.0)
    assert np.all((res.confidence >= 0) & (res.confidence <= 1))


def test_rpeak_config_rejects_bad_searchback():
    with pytest.raises(ValueError):
        RPeakConfig(searchback_factor=0.5).validate()


# ---------- detection evaluation ----------

def test_match_is_one_to_one():
    tp, fp, fn = match_peaks(np.array([100]), np.array([98, 101]), tolerance_samples=5)
    assert (tp, fp, fn) == (1, 1, 0)


def test_scores_perfect_and_empty():
    perfect = evaluate_detection(np.array([100, 300]), np.array([100, 300]), fs=250.0)
    assert perfect.f1 == 1.0
    empty = evaluate_detection(np.array([100, 300]), np.array([]), fs=250.0)
    assert empty.sensitivity == 0.0 and empty.ppv == 0.0 and empty.f1 == 0.0


# ---------- RR intervals ----------

def test_rr_flags_out_of_range_and_jump():
    t = np.array([0.0, 0.8, 1.6, 1.6 + 0.2, 1.6 + 0.2 + 0.85])  # 0.2 s interval is too short
    s = rr_intervals(t, RRPlausibility())
    assert FLAG_OUT_OF_RANGE in s.flags[2]
    assert len(s) == 4
    assert s.nn_s.size < len(s)


def test_relative_jump_flagged():
    t = np.cumsum([0.0, 0.8, 0.8, 0.8, 1.3])  # 0.5 s after 0.8 s is a >20% jump
    s = rr_intervals(t, RRPlausibility(max_relative_change=0.2))
    assert FLAG_RELATIVE_JUMP in s.flags[-1]


def test_interval_spanning_gap_is_flagged_and_not_used_for_jumps():
    t = np.array([0.0, 0.8, 1.6, 2.4, 3.2, 6.0, 6.5, 7.0, 7.5])  # data missing between 3.5 and 5.5 s
    s = rr_intervals(t, RRPlausibility(), gaps_s=[(3.5, 5.5)])
    assert s.flags[4] == (FLAG_SPANS_GAP,)
    # After the gap the rate changed (0.8 s -> 0.5 s); those intervals are judged afresh,
    # not against the pre-gap intervals.
    assert all(f == () for f in s.flags[5:])


def test_borderline_premature_beat_does_not_stick_the_jump_reference():
    """Regression: a short interval just under the jump limit once became the
    reference and flagged every following normal interval."""
    rr = [0.77, 0.77, 0.77, 0.77, 0.616, 1.02] + [0.80] * 30
    s = rr_intervals(np.cumsum([0.0] + rr), RRPlausibility(max_relative_change=0.2))
    later = s.flags[8:]
    assert all(f == () for f in later)


def test_rr_requires_strictly_increasing_times():
    with pytest.raises(ValueError):
        rr_intervals(np.array([1.0, 0.5]), RRPlausibility())


def test_rr_empty_input():
    s = rr_intervals(np.array([1.0]), RRPlausibility())
    assert len(s) == 0


# ---------- heart rate ----------

def test_heart_rate_summary_on_regular_rhythm():
    t = np.arange(0, 60, 0.8)  # 75 bpm
    s = rr_intervals(t, RRPlausibility())
    hr = heart_rate_summary(s, now_s=60.0, window_s=30.0)
    assert hr.mean_bpm == pytest.approx(75.0, abs=0.1)
    assert hr.valid_rr_count > 0


def test_heart_rate_warns_when_no_recent_beat():
    t = np.arange(0, 10, 0.8)
    s = rr_intervals(t, RRPlausibility())
    hr = heart_rate_summary(s, now_s=30.0, window_s=20.0)
    assert any("no accepted R peak" in w for w in hr.warnings)


def test_heart_rate_empty_series():
    s = rr_intervals(np.array([]), RRPlausibility())
    hr = heart_rate_summary(s, now_s=10.0, window_s=5.0)
    assert hr.current_bpm is None


def test_heart_rate_window_must_be_positive():
    s = rr_intervals(np.arange(0, 10, 0.8), RRPlausibility())
    with pytest.raises(ValueError):
        heart_rate_summary(s, now_s=10.0, window_s=0.0)


# ---------- HRV ----------

def test_hrv_withheld_when_short():
    t = np.arange(0, 20, 0.8)
    s = rr_intervals(t, RRPlausibility())
    f = time_domain_hrv(s, HRVConfig(min_duration_s=60.0))
    assert not f.available
    assert f.sdnn_ms is None and f.rmssd_ms is None
    assert any("withheld" in w for w in f.warnings)


def test_hrv_values_on_long_regular_rhythm():
    rng = np.random.default_rng(0)
    rr = 0.8 + rng.normal(0, 0.02, size=200)
    t = np.cumsum(rr)
    s = rr_intervals(t, RRPlausibility())
    f = time_domain_hrv(s, HRVConfig(min_duration_s=60.0, min_nn_intervals=30))
    assert f.available
    assert f.mean_nn_ms == pytest.approx(800.0, abs=10.0)
    assert f.sdnn_ms == pytest.approx(20.0, abs=3.0)
    assert f.rmssd_ms is not None and f.pnn50_pct is not None
    assert 0 <= f.pnn50_pct <= 100


def test_hrv_withheld_when_too_many_flagged():
    rr = np.full(100, 0.8)
    rr[::2] = 0.1  # half the intervals implausible
    t = np.cumsum(rr)
    s = rr_intervals(t, RRPlausibility())
    f = time_domain_hrv(s, HRVConfig(min_duration_s=10.0, max_flagged_fraction=0.2))
    assert not f.available
    assert any("flagged" in w for w in f.warnings)


def test_hrv_empty_series_withheld():
    s = rr_intervals(np.array([]), RRPlausibility())
    f = time_domain_hrv(s, HRVConfig())
    assert not f.available
