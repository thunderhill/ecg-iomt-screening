"""Evaluate R-peak detection against reference annotations.

Matching uses a one-to-one greedy assignment within a tolerance (default
150 ms, the usual convention in QRS evaluation). Reports sensitivity (recall),
positive predictive value (precision) and F1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionScores:
    tp: int
    fp: int
    fn: int
    sensitivity: float
    ppv: float
    f1: float


def match_peaks(reference: np.ndarray, detected: np.ndarray, tolerance_samples: int) -> tuple[int, int, int]:
    """Return (TP, FP, FN) using one-to-one nearest matching within tolerance."""
    ref = np.sort(np.asarray(reference, dtype=int))
    det = np.sort(np.asarray(detected, dtype=int))
    used = np.zeros(det.size, dtype=bool)
    tp = 0
    for r in ref:
        lo = np.searchsorted(det, r - tolerance_samples, side="left")
        hi = np.searchsorted(det, r + tolerance_samples, side="right")
        best, best_dist = -1, None
        for j in range(lo, hi):
            if used[j]:
                continue
            dist = abs(int(det[j]) - int(r))
            if best_dist is None or dist < best_dist:
                best, best_dist = j, dist
        if best >= 0:
            used[best] = True
            tp += 1
    fp = int(det.size - tp)
    fn = int(ref.size - tp)
    return tp, fp, fn


def evaluate_detection(
    reference: np.ndarray,
    detected: np.ndarray,
    fs: float,
    tolerance_s: float = 0.150,
) -> DetectionScores:
    """Sensitivity, PPV and F1 for detected R-peak sample indices."""
    if fs <= 0 or tolerance_s <= 0:
        raise ValueError("fs and tolerance_s must be positive")
    tol = int(round(tolerance_s * fs))
    tp, fp, fn = match_peaks(reference, detected, tol)
    se = tp / (tp + fn) if (tp + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * ppv * se / (ppv + se) if (ppv + se) else 0.0
    return DetectionScores(tp=tp, fp=fp, fn=fn, sensitivity=se, ppv=ppv, f1=f1)
