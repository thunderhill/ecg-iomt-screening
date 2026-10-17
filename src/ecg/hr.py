"""RR intervals and heart-rate estimation.

Implausible intervals are FLAGGED, never silently dropped, so the dashboard can
show how much of the window was excluded and why. Only intervals with no flags
count as NN (normal-to-normal) intervals for HRV. Unfiltered RR intervals are
still available for display.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import RRPlausibility

FLAG_OUT_OF_RANGE = "out_of_range"
FLAG_RELATIVE_JUMP = "relative_jump"
FLAG_SPANS_GAP = "spans_gap"  # data was missing inside the interval; its length is unknown
STALE_BEAT_S = 3.0
JUMP_REFERENCE_BEATS = 5  # intervals in the running-median reference for the jump rule
MIN_JUMP_REFERENCE = 3  # the jump rule applies once this many in-range intervals exist


@dataclass(frozen=True)
class RRSeries:
    """RR intervals with per-interval flags.

    ``times_s`` is the time of the beat that ENDS each interval (the later R).
    ``flags[i]`` is a tuple of flag names; an empty tuple means the interval is
    valid and counts as an NN interval.
    """

    times_s: np.ndarray
    rr_s: np.ndarray
    flags: tuple[tuple[str, ...], ...]

    def __len__(self) -> int:
        return int(self.rr_s.size)

    @property
    def valid_mask(self) -> np.ndarray:
        return np.array([len(f) == 0 for f in self.flags], dtype=bool)

    @property
    def nn_s(self) -> np.ndarray:
        return self.rr_s[self.valid_mask]

    @property
    def flagged_fraction(self) -> float:
        if len(self) == 0:
            return 0.0
        return float(np.count_nonzero(~self.valid_mask)) / len(self)


def rr_intervals(
    r_times_s: np.ndarray,
    rules: RRPlausibility,
    gaps_s: list[tuple[float, float]] | None = None,
) -> RRSeries:
    """Compute RR intervals from R-peak times (seconds) and flag implausible ones.

    ``gaps_s`` lists (start, end) times of missing data. An interval that overlaps
    a gap is flagged ``spans_gap``: beats may have occurred in the missing data,
    so its length says nothing about the heart rate.
    """
    rules.validate()
    t = np.asarray(r_times_s, dtype=float).ravel()
    if t.size < 2:
        empty = np.array([], dtype=float)
        return RRSeries(times_s=empty, rr_s=empty, flags=())
    if np.any(np.diff(t) <= 0):
        raise ValueError("R-peak times must be strictly increasing")

    rr = np.diff(t)
    ends = t[1:]
    flags: list[tuple[str, ...]] = []
    # Reference for the jump rule: median of the previous in-range intervals,
    # INCLUDING ones flagged as jumps. Using only accepted intervals lets one
    # borderline premature beat become a stuck reference that flags every later
    # normal beat; a running median of all in-range intervals adapts instead.
    recent: deque[float] = deque(maxlen=JUMP_REFERENCE_BEATS)
    gaps = gaps_s or []
    for k, value in enumerate(rr):
        f: list[str] = []
        start, end = t[k], t[k + 1]
        if any(g0 < end and g1 > start for g0, g1 in gaps):
            f.append(FLAG_SPANS_GAP)
            recent.clear()  # never judge an interval against ones from before a gap
        elif not (rules.min_rr_s <= value <= rules.max_rr_s):
            f.append(FLAG_OUT_OF_RANGE)
        else:
            if len(recent) >= MIN_JUMP_REFERENCE:
                reference = float(np.median(recent))
                if abs(value - reference) / reference > rules.max_relative_change:
                    f.append(FLAG_RELATIVE_JUMP)
            recent.append(float(value))
        flags.append(tuple(f))
    return RRSeries(times_s=ends, rr_s=rr, flags=tuple(flags))


@dataclass(frozen=True)
class HeartRateSummary:
    current_bpm: float | None
    mean_bpm: float | None
    min_bpm: float | None
    max_bpm: float | None
    valid_rr_count: int
    flagged_rr_count: int
    seconds_since_last_beat: float | None
    warnings: tuple[str, ...]


def heart_rate_summary(
    series: RRSeries,
    now_s: float,
    window_s: float,
    min_valid_rr: int = 8,
) -> HeartRateSummary:
    """Summarise heart rate over the trailing ``window_s`` seconds ending at ``now_s``."""
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    warnings: list[str] = []
    if len(series) == 0:
        return HeartRateSummary(None, None, None, None, 0, 0, None, ("no RR intervals yet",))

    last_beat = float(series.times_s[-1])
    since = max(0.0, now_s - last_beat)
    in_window = (series.times_s > now_s - window_s) & (series.times_s <= now_s)
    valid = in_window & series.valid_mask
    flagged = int(np.count_nonzero(in_window & ~series.valid_mask))
    n_valid = int(np.count_nonzero(valid))

    # Confirmed beats trail the live edge by about 1 s by design (see pipeline),
    # so only a longer silence is reported as stale.
    if since > STALE_BEAT_S:
        warnings.append(f"no accepted R peak for {since:.1f} s")
    if n_valid == 0:
        warnings.append("no valid RR intervals in window")
        return HeartRateSummary(None, None, None, None, 0, flagged, since, tuple(warnings))

    bpm = 60.0 / series.rr_s[valid]
    if n_valid < min_valid_rr:
        warnings.append(f"only {n_valid} valid RR intervals; need {min_valid_rr} for a reliable average")
    if flagged:
        warnings.append(f"{flagged} implausible RR interval(s) excluded from heart-rate statistics")

    # Current HR: the most recent valid interval.
    last_valid_idx = np.flatnonzero(valid)[-1]
    current = float(60.0 / series.rr_s[last_valid_idx])
    return HeartRateSummary(
        current_bpm=current,
        mean_bpm=float(np.mean(bpm)),
        min_bpm=float(np.min(bpm)),
        max_bpm=float(np.max(bpm)),
        valid_rr_count=n_valid,
        flagged_rr_count=flagged,
        seconds_since_last_beat=since,
        warnings=tuple(warnings),
    )
