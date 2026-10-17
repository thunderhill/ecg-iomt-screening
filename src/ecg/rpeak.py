"""Pan–Tompkins-style QRS (R-peak) detection.

Stages, following Pan & Tompkins (1985):
1. Bandpass 5–15 Hz to emphasise QRS energy.
2. Five-point derivative for slope information.
3. Squaring for a non-negative energy transform.
4. Moving-window integration (about 150 ms).
5. Adaptive thresholds (signal and noise peak estimates) on the integrated signal.
6. Refractory period (200 ms) and T-wave discrimination.
7. Search-back for missed beats when the RR gap is long.
8. Refinement: the R location is the largest |amplitude| on the bandpassed signal
   near each accepted integrated peak.

Known limitations: the detector is tuned for adult ECG at moderate heart rates
and is sensitive to noise, ectopic beats and wide QRS complexes. Detected QRS
complexes are NOT assumed to be normal beats. Accuracy must be measured against
annotated recordings (see ``evaluate_detection``).

Offline use ``zero_phase=True``. Live use ``zero_phase=False`` (causal). The
live pipeline re-runs detection over a rolling window and keeps only peaks
that are safely past the window's right edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .config import ConfigError, RPeakConfig

QRS_BAND_HZ = (5.0, 15.0)


@dataclass(frozen=True)
class RPeakResult:
    indices: np.ndarray  # R-peak sample indices (int)
    times_s: np.ndarray  # R-peak times in seconds from the start of the input
    confidence: np.ndarray  # heuristic in [0, 1]; not a probability
    fs: float = 250.0
    notes: tuple[str, ...] = ()

    def __len__(self) -> int:
        return int(self.indices.size)


def _empty(fs: float, note: str) -> RPeakResult:
    return RPeakResult(
        indices=np.array([], dtype=int),
        times_s=np.array([], dtype=float),
        confidence=np.array([], dtype=float),
        fs=fs,
        notes=(note,),
    )


def _bandpass_qrs(x: np.ndarray, fs: float, zero_phase: bool) -> np.ndarray:
    nyq = fs / 2.0
    sos = sps.butter(2, [QRS_BAND_HZ[0] / nyq, QRS_BAND_HZ[1] / nyq], btype="bandpass", output="sos")
    if zero_phase:
        return sps.sosfiltfilt(sos, x)
    zi = sps.sosfilt_zi(sos) * x[0]
    y, _ = sps.sosfilt(sos, x, zi=zi)
    return y


def _derivative(x: np.ndarray, fs: float) -> np.ndarray:
    # Five-point derivative from Pan & Tompkins (scaled to ~ per-second slope).
    h = np.array([1.0, 2.0, 0.0, -2.0, -1.0]) * (fs / 8.0)
    return np.convolve(x, h, mode="same")


def _moving_window_integration(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    width = max(1, int(round(window_s * fs)))
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


def detect_rpeaks(
    x: np.ndarray,
    fs: float,
    cfg: RPeakConfig | None = None,
    *,
    zero_phase: bool = True,
) -> RPeakResult:
    """Detect R peaks in a (preferably preprocessed) ECG segment.

    Returns indices and times in seconds from the start of ``x``. Short or
    flat inputs return an empty result with a note; they never raise for
    physiologically valid but short data.
    """
    cfg = cfg or RPeakConfig()
    cfg.validate()
    if not np.isfinite(fs) or fs <= 2 * QRS_BAND_HZ[1]:
        raise ConfigError(f"fs={fs} too low for QRS detection (need > {2 * QRS_BAND_HZ[1]} Hz)")
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("R-peak detection needs finite, non-empty input")
    if arr.size < int(2.0 * fs):
        return _empty(fs, "segment shorter than 2 s; no detection attempted")
    if float(np.std(arr)) == 0.0:
        return _empty(fs, "flat signal; no QRS complexes can be detected")

    bp = _bandpass_qrs(arr, fs, zero_phase)
    diff = _derivative(bp, fs)
    squared = diff**2
    integ = _moving_window_integration(squared, fs, cfg.integration_window_s)

    refractory = int(round(cfg.refractory_s * fs))
    refine_half = int(round(cfg.refine_window_s * fs))
    tw_gap = int(round(0.36 * fs))  # T-wave discrimination window (360 ms)
    cand = sps.find_peaks(integ)[0]
    if cand.size == 0:
        return _empty(fs, "no candidate peaks in integrated signal")

    # Initial thresholds from the first 2 s (Pan & Tompkins initialisation).
    init = integ[: int(2 * fs)]
    spki = 0.25 * float(np.max(init))
    npki = 0.5 * float(np.mean(init))

    accepted: list[int] = []
    searchback: list[int] = []  # positions in the integrated signal, counted only
    accepted_amp: list[float] = []
    last_sig = -10**9
    rr_avg = None  # running average of accepted RR in samples
    last_slope = None

    def max_slope(center: int) -> float:
        lo = max(0, center - refine_half)
        hi = min(arr.size, center + refine_half + 1)
        return float(np.max(np.abs(diff[lo:hi]))) if hi > lo else 0.0

    def accept(pos: int, amp: float, from_searchback: bool) -> None:
        nonlocal spki, last_sig, rr_avg, last_slope
        accepted.append(pos)
        accepted_amp.append(amp)
        if from_searchback:
            searchback.append(pos)
        spki = 0.125 * amp + 0.875 * spki
        if last_sig > 0:
            rr = pos - last_sig
            rr_avg = rr if rr_avg is None else 0.875 * rr_avg + 0.125 * rr
        last_sig = pos
        last_slope = max_slope(pos)

    for p in cand:
        p = int(p)
        amp = float(integ[p])
        thr1 = npki + 0.25 * (spki - npki)

        # Search-back: a long gap since the last accepted beat suggests a missed
        # beat; look for the largest peak in the gap above half the threshold.
        if rr_avg is not None and last_sig > 0 and (p - last_sig) > cfg.searchback_factor * rr_avg:
            gap_lo, gap_hi = last_sig + refractory, p - refractory
            if gap_hi > gap_lo:
                window = integ[gap_lo:gap_hi]
                j = int(np.argmax(window)) + gap_lo
                if integ[j] > 0.5 * thr1:
                    accept(j, float(integ[j]), from_searchback=True)

        if amp > thr1 and (p - last_sig) >= refractory:
            # T-wave rule: a peak within 360 ms of the last beat with less than
            # half its slope is treated as a T wave (noise), not a new QRS.
            is_twave = (
                last_slope is not None
                and (p - last_sig) < tw_gap
                and max_slope(p) < 0.5 * last_slope
            )
            if is_twave:
                npki = 0.125 * amp + 0.875 * npki
            else:
                accept(p, amp, from_searchback=False)
        else:
            npki = 0.125 * amp + 0.875 * npki

    # Refinement: map each integrated peak to the R location on the CALLER'S signal.
    # The internal 5-15 Hz filter is causal when zero_phase=False and delays the QRS
    # by a few tens of ms, so locating R on it would timestamp every beat late. The
    # search window reaches back far enough to cover that delay.
    win = int(round(cfg.integration_window_s * fs))
    lag_margin = int(round(0.05 * fs))
    refined: list[int] = []
    conf: list[float] = []
    for pos, amp in zip(accepted, accepted_amp):
        lo = max(0, pos - win - lag_margin)
        hi = min(arr.size, pos + win // 2 + 1)
        if hi <= lo:
            continue
        seg = arr[lo:hi]
        r = lo + int(np.argmax(np.abs(seg - np.median(seg))))
        if refined and r - refined[-1] < refractory:
            # Two integrated peaks collapsed onto one R: keep the stronger one.
            if abs(bp[r]) > abs(bp[refined[-1]]):
                refined[-1] = r
                conf[-1] = min(1.0, amp / max(spki, 1e-12))
            continue
        refined.append(r)
        conf.append(min(1.0, amp / max(spki, 1e-12)))

    idx = np.asarray(refined, dtype=int)
    notes: list[str] = []
    if searchback:
        notes.append(f"{len(searchback)} beat(s) recovered by search-back")
    return RPeakResult(
        indices=idx,
        times_s=idx / fs,
        confidence=np.asarray(conf, dtype=float),
        fs=fs,
        notes=tuple(notes),
    )
