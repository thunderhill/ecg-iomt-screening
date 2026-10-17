import numpy as np

from ecg.config import RRPlausibility, ScreeningConfig
from ecg.hr import rr_intervals
from ecg.ml.rules import Category, screen_heart_rate


def _series(rr_values):
    t = np.cumsum([0.0] + list(rr_values))
    return rr_intervals(t, RRPlausibility()), float(t[-1])


def test_sustained_elevated_rate_is_flagged_with_evidence():
    series, now = _series([0.5] * 60)  # 120 bpm for 30 s
    r = screen_heart_rate(series, now, ScreeningConfig())
    assert r.category == Category.ELEVATED_HR
    assert abs(r.measured_bpm - 120.0) < 0.5
    assert r.duration_s >= 10.0
    assert any("held for" in e for e in r.evidence)
    assert "does not establish" in r.interpretation


def test_short_fast_burst_is_not_flagged():
    # 6 s at 120 bpm inside an otherwise 70 bpm minute: below the 10 s duration rule.
    rr = [0.857] * 30 + [0.5] * 12 + [0.857] * 20
    series, now = _series(rr)
    r = screen_heart_rate(series, now, ScreeningConfig(min_duration_s=10.0))
    assert r.category == Category.NO_FLAG


def test_sustained_low_rate_is_flagged():
    series, now = _series([1.5] * 30)  # 40 bpm
    r = screen_heart_rate(series, now, ScreeningConfig())
    assert r.category == Category.LOW_HR
    assert r.threshold_bpm == 50.0


def test_no_flag_never_claims_normal():
    series, now = _series([0.857] * 60)
    r = screen_heart_rate(series, now, ScreeningConfig())
    assert r.category == Category.NO_FLAG
    assert "does NOT mean the rhythm is normal" in r.interpretation


def test_poor_signal_suppresses_rules():
    series, now = _series([0.5] * 60)
    r = screen_heart_rate(series, now, ScreeningConfig(), signal_status="poor")
    assert r.category == Category.POOR_SIGNAL


def test_too_few_beats_is_insufficient_data():
    series, now = _series([0.5] * 4)
    r = screen_heart_rate(series, now, ScreeningConfig(min_valid_rr=8))
    assert r.category == Category.INSUFFICIENT_DATA


def test_thresholds_are_configurable():
    series, now = _series([0.5] * 60)  # 120 bpm
    r = screen_heart_rate(series, now, ScreeningConfig(tachy_bpm=130.0))
    assert r.category == Category.NO_FLAG
