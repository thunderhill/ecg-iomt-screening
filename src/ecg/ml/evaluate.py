"""Classification metrics for beat-level models.

All metrics are computed on held-out records (patient-wise). Sensitivity and
specificity are one-vs-rest per class. ROC-AUC is reported only when the model
outputs probabilities and every class appears in the test labels. Otherwise it
is omitted and the reason is recorded.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: tuple[str, ...],
    proba: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = list(classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    total = cm.sum()
    per_class = {}
    for k, c in enumerate(labels):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        tn = total - tp - fp - fn
        per_class[c] = {
            "precision": float(precision[k]),
            "sensitivity": float(recall[k]),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
            "f1": float(f1[k]),
            "support": int(support[k]),
        }

    report: dict[str, Any] = {
        "classes": labels,
        "n": int(total),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
    present = set(np.unique(y_true))
    if proba is None:
        report["roc_auc_ovr_macro"] = None
        report["roc_auc_note"] = "model does not output probabilities"
    elif not set(labels) <= present:
        report["roc_auc_ovr_macro"] = None
        report["roc_auc_note"] = "not every class is present in the test labels"
    else:
        # One-vs-rest per class; the macro value is their unweighted mean.
        per_class_auc = {c: float(roc_auc_score(y_true == c, proba[:, k])) for k, c in enumerate(labels)}
        for c, auc in per_class_auc.items():
            per_class[c]["roc_auc_ovr"] = auc
        report["roc_auc_ovr_macro"] = float(np.mean(list(per_class_auc.values())))
    return report


def measure_latency(predict: Callable[[np.ndarray], Any], X: np.ndarray, repeats: int = 200) -> dict[str, float]:
    """Median single-beat latency and batch throughput, on this machine's CPU."""
    single = []
    for i in range(min(repeats, X.shape[0])):
        row = X[i : i + 1]
        t0 = time.perf_counter()
        predict(row)
        single.append(time.perf_counter() - t0)
    batch = X[: min(5000, X.shape[0])]
    t0 = time.perf_counter()
    predict(batch)
    batch_s = time.perf_counter() - t0
    return {
        "single_beat_median_ms": float(np.median(single) * 1000.0),
        "batch_per_beat_us": float(batch_s / max(1, batch.shape[0]) * 1e6),
        "note": "single-threaded, on the training machine's CPU",
    }
