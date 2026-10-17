"""Per-beat feature extraction, shared by training and live inference.

Causality: the features for beat *i* use
* RR intervals that end at or before beat *i* (never later beats), and
* the causally filtered signal up to ``POST_R_S`` (0.25 s) after its R peak.
So a beat can be classified 0.25 s after it occurs. No feature uses later beats,
which would leak future information into a "real-time" claim.

Beats with fewer than ``MIN_HISTORY`` previous beats get no features. They are
reported as "insufficient history" and are never filled with guessed values.

Scaling: tree models (RF, XGBoost) use raw features. The CNN uses its own
per-beat normalisation (see ``ml/cnn.py``).

The input must be filtered with ``preprocess(..., mode="causal")`` and the
same ``PreprocessConfig`` at training and inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_HISTORY = 8  # previous beats required before a beat gets features
PRE_R_S = 0.25
POST_R_S = 0.25
TEMPLATE_HALF_S = 0.2

FEATURE_NAMES: tuple[str, ...] = (
    "rr_prev_s",  # interval ending at this beat
    "rr_prev2_s",  # interval before that
    "rr_local_mean_s",  # mean of the previous 8 intervals, excluding rr_prev
    "rr_ratio",  # rr_prev / rr_local_mean (prematurity)
    "rr_local_std_s",  # std of the last 8 intervals including rr_prev
    "rr_local_rmssd_s",  # RMSSD over the last 8 intervals
    "hr_local_bpm",  # 60 / mean of the last 8 intervals
    "r_amp_mv",  # filtered amplitude at R
    "q_amp_mv",  # minimum in the 100 ms before R
    "s_amp_mv",  # minimum in the 100 ms after R
    "qrs_width_s",  # width where |x| stays above half of |R| around R
    "qrs_energy",  # sum of squares within +/- 50 ms
    "post_r_max_mv",  # maximum 150-250 ms after R (early T-wave region)
    "template_corr",  # correlation with the running template of previous beats
    "local_noise_mv",  # std of first difference in the beat window (noise proxy)
)


@dataclass(frozen=True)
class BeatFeatures:
    beat_indices: np.ndarray  # R-peak sample indices of beats that received features
    X: np.ndarray  # shape (n_beats, n_features)
    skipped_history: int  # beats skipped for insufficient history
    skipped_edge: int  # beats skipped because the window ran past the signal


def _qrs_width(seg: np.ndarray, center: int, fs: float) -> float:
    peak = abs(seg[center])
    if peak == 0:
        return 0.0
    above = np.abs(seg) >= 0.5 * peak
    lo = center
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = center
    while hi < seg.size - 1 and above[hi + 1]:
        hi += 1
    return (hi - lo + 1) / fs


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def extract_beat_features(filtered: np.ndarray, r_indices: np.ndarray, fs: float) -> BeatFeatures:
    """Compute ``FEATURE_NAMES`` for each beat in ``r_indices`` that has enough history.

    ``filtered`` must be the causally filtered signal in millivolts.
    """
    x = np.asarray(filtered, dtype=float)
    r = np.asarray(r_indices, dtype=np.int64)
    if r.size and (np.any(np.diff(r) <= 0) or r[0] < 0 or r[-1] >= x.size):
        raise ValueError("r_indices must be strictly increasing and inside the signal")
    pre = int(round(PRE_R_S * fs))
    post = int(round(POST_R_S * fs))
    half_t = int(round(TEMPLATE_HALF_S * fs))
    w100 = int(round(0.1 * fs))
    w50 = int(round(0.05 * fs))
    t_lo, t_hi = int(round(0.15 * fs)), int(round(0.25 * fs))

    rr = np.diff(r) / fs  # rr[k] ends at beat k+1
    rows: list[list[float]] = []
    kept: list[int] = []
    templates: list[np.ndarray] = []
    skipped_history = 0
    skipped_edge = 0

    for i, ri in enumerate(r):
        lo, hi = ri - pre, ri + post + 1
        seg_ok = lo >= 0 and hi <= x.size
        tmpl_seg = x[ri - half_t : ri + half_t + 1] if (ri - half_t >= 0 and ri + half_t + 1 <= x.size) else None

        if i < MIN_HISTORY + 1:
            skipped_history += 1
        elif not seg_ok:
            skipped_edge += 1
        else:
            rr_prev = rr[i - 1]
            rr_prev2 = rr[i - 2]
            hist = rr[i - 1 - MIN_HISTORY : i - 1]  # previous 8, excluding rr_prev
            last8 = rr[i - MIN_HISTORY : i]  # last 8 including rr_prev
            local_mean = float(np.mean(hist))
            seg = x[lo:hi]
            c = pre  # index of R within seg
            template = np.mean(templates[-MIN_HISTORY:], axis=0) if templates else None
            rows.append(
                [
                    rr_prev,
                    rr_prev2,
                    local_mean,
                    rr_prev / local_mean if local_mean > 0 else np.nan,
                    float(np.std(last8)),
                    float(np.sqrt(np.mean(np.diff(last8) ** 2))),
                    60.0 / float(np.mean(last8)),
                    float(seg[c]),
                    float(np.min(seg[c - w100 : c + 1])),
                    float(np.min(seg[c : c + w100 + 1])),
                    _qrs_width(seg, c, fs),
                    float(np.sum(seg[c - w50 : c + w50 + 1] ** 2)),
                    float(np.max(seg[c + t_lo : c + t_hi + 1])),
                    _corr(tmpl_seg, template) if (template is not None and tmpl_seg is not None) else np.nan,
                    float(np.std(np.diff(seg))),
                ]
            )
            kept.append(int(ri))
        # The template is updated AFTER the beat's own features are computed,
        # so a beat is never compared with a template that contains itself.
        if tmpl_seg is not None:
            templates.append(tmpl_seg)
            if len(templates) > MIN_HISTORY:
                templates.pop(0)

    X = np.asarray(rows, dtype=float).reshape(-1, len(FEATURE_NAMES))
    return BeatFeatures(
        beat_indices=np.asarray(kept, dtype=np.int64),
        X=X,
        skipped_history=skipped_history,
        skipped_edge=skipped_edge,
    )
