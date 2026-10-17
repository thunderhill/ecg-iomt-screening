"""Reliability warnings attached to every explanation.

An explanation is only as trustworthy as the prediction it explains. These
checks are surfaced next to every model output, and they never change the
prediction itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def reliability_warnings(
    *,
    quality_status: str,
    n_beats: int,
    min_beats: int,
    bundle: dict[str, Any] | None,
    X: np.ndarray | None = None,
    predicted_classes: np.ndarray | None = None,
    source: str | None = None,
    ood_fraction_threshold: float = 0.2,
) -> list[str]:
    warnings: list[str] = []
    if bundle is None:
        warnings.append("No trained model is available; only rule-based screening is shown.")
        return warnings
    if quality_status in ("poor", "insufficient_data"):
        warnings.append(f"Signal quality is '{quality_status}': predictions and explanations may be meaningless.")
    if n_beats < min_beats:
        warnings.append(f"Only {n_beats} beats were classified (need {min_beats}); the result has little support.")
    if predicted_classes is not None and predicted_classes.size:
        unknown = set(np.unique(predicted_classes)) - set(bundle["classes"])
        if unknown:
            warnings.append(f"Predicted classes {sorted(unknown)} are outside the validated class set.")
    if X is not None and X.size:
        lo = np.asarray(bundle["feature_ranges"]["p01"])
        hi = np.asarray(bundle["feature_ranges"]["p99"])
        outside = (X < lo) | (X > hi)
        frac_beats = float(np.mean(outside.mean(axis=1) > 0.25))
        if frac_beats > ood_fraction_threshold:
            worst = np.argsort(-outside.mean(axis=0))[:3]
            names = [bundle["feature_names"][j] for j in worst]
            warnings.append(
                f"{frac_beats:.0%} of beats have many features outside the training range "
                f"(most often: {', '.join(names)}). The input may differ from the training data."
            )
    if source == "SIMULATED":
        warnings.append("Input is SIMULATED software-generated ECG. Model output on it says nothing about clinical accuracy.")
    if source == "LIVE":
        warnings.append("Live ESP32 data uses placeholder ADC calibration; amplitudes may not match the MIT-BIH training data.")
    return warnings
