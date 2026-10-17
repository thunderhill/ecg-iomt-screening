"""Time-domain HRV features from NN (normal-to-normal) intervals.

Metrics:
* Mean NN interval (ms)
* SDNN (ms): standard deviation of NN intervals
* RMSSD (ms): root mean square of successive NN differences
* pNN50 (%): share of successive NN differences greater than 50 ms

Values are WITHHELD (None) when the recording is too short, too few NN
intervals exist, or too many intervals were flagged as implausible. Warnings
always explain why. HRV features are descriptive measurements. They are not a
standalone diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HRVConfig
from .hr import RRSeries


@dataclass(frozen=True)
class HRVFeatures:
    available: bool
    mean_nn_ms: float | None
    sdnn_ms: float | None
    rmssd_ms: float | None
    pnn50_pct: float | None
    nn_count: int
    flagged_fraction: float
    warnings: tuple[str, ...]


def _withheld(reason: str, nn_count: int, flagged: float) -> HRVFeatures:
    return HRVFeatures(
        available=False,
        mean_nn_ms=None,
        sdnn_ms=None,
        rmssd_ms=None,
        pnn50_pct=None,
        nn_count=nn_count,
        flagged_fraction=flagged,
        warnings=(f"HRV withheld: {reason}",),
    )


def time_domain_hrv(series: RRSeries, cfg: HRVConfig) -> HRVFeatures:
    """Compute time-domain HRV from the NN intervals in ``series``."""
    cfg.validate()
    nn = series.nn_s
    flagged = series.flagged_fraction
    if len(series) == 0:
        return _withheld("no RR intervals", 0, flagged)

    duration = float(series.times_s[-1] - series.times_s[0]) + float(series.rr_s[0])
    if duration < cfg.min_duration_s:
        return _withheld(f"recording {duration:.0f} s, need {cfg.min_duration_s:.0f} s", nn.size, flagged)
    if nn.size < cfg.min_nn_intervals:
        return _withheld(f"{nn.size} NN intervals, need {cfg.min_nn_intervals}", nn.size, flagged)
    if flagged > cfg.max_flagged_fraction:
        return _withheld(
            f"{flagged:.0%} of intervals flagged as implausible (limit {cfg.max_flagged_fraction:.0%})",
            nn.size,
            flagged,
        )

    nn_ms = nn * 1000.0
    # Successive differences are only meaningful between consecutive NN intervals.
    # Pairs that straddle a flagged interval are excluded from RMSSD and pNN50.
    flags_mask = series.valid_mask
    pair_ok = flags_mask[:-1] & flags_mask[1:]
    successive = np.diff(series.rr_s * 1000.0)[pair_ok] if pair_ok.size else np.array([])
    warnings: list[str] = []
    if successive.size < 2:
        warnings.append("too few consecutive NN pairs; RMSSD and pNN50 withheld")
        rmssd = None
        pnn50 = None
    else:
        rmssd = float(np.sqrt(np.mean(successive**2)))
        pnn50 = float(100.0 * np.mean(np.abs(successive) > 50.0))

    if flagged > 0:
        warnings.append(f"{flagged:.0%} of RR intervals flagged and excluded from NN statistics")
    warnings.append("short-term HRV from a single recording; interpret descriptively only")
    return HRVFeatures(
        available=True,
        mean_nn_ms=float(np.mean(nn_ms)),
        sdnn_ms=float(np.std(nn_ms, ddof=1)),
        rmssd_ms=rmssd,
        pnn50_pct=pnn50,
        nn_count=int(nn.size),
        flagged_fraction=flagged,
        warnings=tuple(warnings),
    )
