"""Model A: rule-based heart-rate screening.

Flags a SUSTAINED heart rate above ``tachy_bpm`` or below ``brady_bpm``. A
single fast or slow interval never triggers a flag.

These are preliminary research screening flags, NOT diagnoses. The thresholds
are research parameters from ``ScreeningConfig``, not universal clinical
thresholds. A recording without a flag is never described as "normal".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..config import ScreeningConfig
from ..hr import RRSeries


class Category(str, Enum):
    NO_FLAG = "no_configured_flag"
    ELEVATED_HR = "possible_elevated_heart_rate"
    LOW_HR = "possible_low_heart_rate"
    ARRHYTHMIA_PATTERN = "possible_arrhythmia_pattern"
    INSUFFICIENT_DATA = "insufficient_data"
    POOR_SIGNAL = "insufficient_signal_quality"
    MODEL_UNAVAILABLE = "model_unavailable"


@dataclass(frozen=True)
class RuleResult:
    model: str
    category: Category
    title: str
    evidence: tuple[str, ...]
    interpretation: str
    limitations: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    measured_bpm: float | None = None
    threshold_bpm: float | None = None
    duration_s: float | None = None
    valid_rr_count: int = 0


_LIMITATIONS = (
    "Motion artifacts, poor electrode contact and other signal abnormalities can change heart-rate estimates.",
    "A heart-rate threshold does not establish the underlying rhythm or its cause.",
    "Thresholds are configurable research parameters, not clinical standards.",
)


def _sustained_run(
    series: RRSeries, now_s: float, window_s: float, above: bool, threshold: float
) -> tuple[float, float | None, int]:
    """Longest contiguous run (seconds) of valid beats beyond ``threshold``.

    Returns (run_duration_s, mean_bpm_of_run, valid_count_in_window).
    A flagged or in-range interval breaks the run.
    """
    in_window = (series.times_s > now_s - window_s) & (series.times_s <= now_s)
    valid_count = int(np.count_nonzero(in_window & series.valid_mask))
    best = (0.0, None)
    run_start: float | None = None
    run_bpms: list[float] = []
    for i in np.flatnonzero(in_window):
        if not series.valid_mask[i]:
            run_start, run_bpms = None, []
            continue
        bpm = 60.0 / series.rr_s[i]
        beyond = bpm > threshold if above else bpm < threshold
        if beyond:
            if run_start is None:
                run_start = float(series.times_s[i] - series.rr_s[i])
                run_bpms = []
            run_bpms.append(float(bpm))
            duration = float(series.times_s[i]) - run_start
            if duration > best[0]:
                best = (duration, float(np.mean(run_bpms)))
        else:
            run_start, run_bpms = None, []
    return best[0], best[1], valid_count


def screen_heart_rate(
    series: RRSeries,
    now_s: float,
    cfg: ScreeningConfig,
    *,
    signal_status: str = "good",
) -> RuleResult:
    """Apply the heart-rate rules to the trailing analysis window."""
    cfg.validate()
    if signal_status == "poor":
        return RuleResult(
            model="rules",
            category=Category.POOR_SIGNAL,
            title="Insufficient signal quality",
            evidence=(),
            interpretation="Signal quality is too poor to apply heart-rate rules reliably.",
            limitations=_LIMITATIONS,
            warnings=("signal quality is poor; no heart-rate flag is reported",),
        )

    tach_dur, tach_mean, valid = _sustained_run(series, now_s, cfg.analysis_window_s, True, cfg.tachy_bpm)
    brad_dur, brad_mean, _ = _sustained_run(series, now_s, cfg.analysis_window_s, False, cfg.brady_bpm)

    if valid < cfg.min_valid_rr:
        return RuleResult(
            model="rules",
            category=Category.INSUFFICIENT_DATA,
            title="Insufficient data",
            evidence=(f"{valid} valid RR intervals in the last {cfg.analysis_window_s:.0f} s; need {cfg.min_valid_rr}",),
            interpretation="Not enough valid beats to apply the heart-rate rules.",
            limitations=_LIMITATIONS,
            valid_rr_count=valid,
        )

    common = (f"Valid RR intervals in window: {valid}", f"Signal quality: {signal_status}")
    if tach_dur >= cfg.min_duration_s:
        return RuleResult(
            model="rules",
            category=Category.ELEVATED_HR,
            title="Possible elevated heart rate",
            evidence=common
            + (
                f"Mean heart rate during the run: {tach_mean:.0f} beats/min",
                f"Threshold {cfg.tachy_bpm:.0f} beats/min held for {tach_dur:.1f} s (required {cfg.min_duration_s:.0f} s)",
            ),
            interpretation=(
                "The elevated heart rate triggered a screening rule. "
                "This observation does not establish the underlying rhythm or cause."
            ),
            limitations=_LIMITATIONS,
            measured_bpm=tach_mean,
            threshold_bpm=cfg.tachy_bpm,
            duration_s=tach_dur,
            valid_rr_count=valid,
        )
    if brad_dur >= cfg.min_duration_s:
        return RuleResult(
            model="rules",
            category=Category.LOW_HR,
            title="Possible low heart rate",
            evidence=common
            + (
                f"Mean heart rate during the run: {brad_mean:.0f} beats/min",
                f"Threshold {cfg.brady_bpm:.0f} beats/min held for {brad_dur:.1f} s (required {cfg.min_duration_s:.0f} s)",
            ),
            interpretation=(
                "The low heart rate triggered a screening rule. "
                "This observation does not establish the underlying rhythm or cause."
            ),
            limitations=_LIMITATIONS,
            measured_bpm=brad_mean,
            threshold_bpm=cfg.brady_bpm,
            duration_s=brad_dur,
            valid_rr_count=valid,
        )
    return RuleResult(
        model="rules",
        category=Category.NO_FLAG,
        title="No configured screening flag detected",
        evidence=common,
        interpretation=(
            "No sustained heart rate beyond the configured thresholds was found. "
            "This does NOT mean the rhythm is normal: this model only checks heart rate."
        ),
        limitations=_LIMITATIONS,
        valid_rr_count=valid,
    )
