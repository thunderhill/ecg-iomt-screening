"""Train and persist beat classifiers (Model B: Random Forest, Model C: XGBoost).

Protocol:
* Train on DS1 records and test once on DS2 records. The split is record-wise,
  and a MIT-BIH record is one patient, so no patient appears in both sets.
* Report grouped (record-wise) cross-validation on DS1 for a variance estimate.
* Use balanced class weights for the heavy N-class imbalance.
* Fix every random seed.

A saved model is a "bundle" dict with the estimator, its class list, feature
names, training feature ranges (for out-of-distribution warnings), seed and
metrics. Inference refuses bundles whose feature names differ from the
current extractor.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from .. import __version__
from ..features import FEATURE_NAMES
from .beats import SUPPORTED_CLASSES, BeatDataset
from .evaluate import classification_report, measure_latency


@dataclass(frozen=True)
class RFParams:
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 2
    class_weight: str = "balanced_subsample"


@dataclass(frozen=True)
class XGBParams:
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8


@dataclass
class TrainResult:
    kind: str
    bundle: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


class _LabelEncodedXGB:
    """Wraps XGBClassifier so that it accepts and returns string class labels."""

    def __init__(self, model, classes: tuple[str, ...]):
        self.model = model
        self.classes_ = np.array(classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.model.predict(X)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def _make_estimator(kind: str, seed: int, rf: RFParams, xgb: XGBParams):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=rf.n_estimators,
            max_depth=rf.max_depth,
            min_samples_leaf=rf.min_samples_leaf,
            class_weight=rf.class_weight,
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=xgb.n_estimators,
            max_depth=xgb.max_depth,
            learning_rate=xgb.learning_rate,
            subsample=xgb.subsample,
            colsample_bytree=xgb.colsample_bytree,
            objective="multi:softprob",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown model kind {kind!r}")


def _fit(kind: str, X: np.ndarray, y: np.ndarray, seed: int, rf: RFParams, xgb: XGBParams):
    est = _make_estimator(kind, seed, rf, xgb)
    if kind == "rf":
        est.fit(X, y)
        return est
    # XGBoost needs contiguous integer labels and explicit balancing weights.
    # Encode only the classes present: a grouped CV fold may lack a rare class.
    present = tuple(c for c in SUPPORTED_CLASSES if c in set(y))
    index = {c: i for i, c in enumerate(present)}
    y_int = np.array([index[v] for v in y])
    est.fit(X, y_int, sample_weight=compute_sample_weight("balanced", y_int))
    return _LabelEncodedXGB(est, present)


def _ordered_proba(model, X: np.ndarray) -> np.ndarray:
    """Probabilities with columns in SUPPORTED_CLASSES order."""
    proba = model.predict_proba(X)
    cols = [list(model.classes_).index(c) if c in model.classes_ else None for c in SUPPORTED_CLASSES]
    out = np.zeros((X.shape[0], len(SUPPORTED_CLASSES)))
    for j, col in enumerate(cols):
        if col is not None:
            out[:, j] = proba[:, col]
    return out


def model_size_bytes(bundle: dict[str, Any]) -> int:
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    return buf.tell()


def train_and_evaluate(
    kind: str,
    train: BeatDataset,
    test: BeatDataset,
    *,
    seed: int = 42,
    cv_folds: int = 5,
    rf: RFParams = RFParams(),
    xgb: XGBParams = XGBParams(),
) -> TrainResult:
    if tuple(train.feature_names) != FEATURE_NAMES or tuple(test.feature_names) != FEATURE_NAMES:
        raise ValueError("dataset features differ from the current extractor; rebuild the dataset")
    overlap = set(np.unique(train.groups)) & set(np.unique(test.groups))
    if overlap:
        raise ValueError(f"patient leakage: records {sorted(overlap)} appear in train and test")

    # Grouped CV on the training records only (no test data is touched).
    cv_scores = []
    for tr, va in GroupKFold(n_splits=cv_folds).split(train.X, train.y, train.groups):
        m = _fit(kind, train.X[tr], train.y[tr], seed, rf, xgb)
        rep = classification_report(train.y[va], m.predict(train.X[va]), SUPPORTED_CLASSES)
        cv_scores.append(rep["macro_f1"])

    t0 = time.perf_counter()
    model = _fit(kind, train.X, train.y, seed, rf, xgb)
    fit_s = time.perf_counter() - t0

    # Measure and serve single-threaded: thread fan-out inflates single-beat latency.
    inner = getattr(model, "model", model)
    if hasattr(inner, "n_jobs"):
        inner.n_jobs = 1

    y_pred = model.predict(test.X)
    proba = _ordered_proba(model, test.X)
    report = classification_report(test.y, y_pred, SUPPORTED_CLASSES, proba)

    bundle: dict[str, Any] = {
        "kind": kind,
        "model": model,
        "classes": SUPPORTED_CLASSES,
        "feature_names": FEATURE_NAMES,
        "feature_medians": np.median(train.X, axis=0).tolist(),  # occlusion baseline
        "feature_ranges": {
            "p01": np.percentile(train.X, 1, axis=0).tolist(),
            "p99": np.percentile(train.X, 99, axis=0).tolist(),
        },
        "seed": seed,
        "train_records": sorted(np.unique(train.groups).tolist()),
        "test_records": sorted(np.unique(test.groups).tolist()),
        "params": (rf if kind == "rf" else xgb).__dict__,
        "package_version": __version__,
        "dataset": "MIT-BIH Arrhythmia Database (DS1 train / DS2 test, 250 Hz, MLII)",
        "label_level": "beat (AAMI N/S/V/F); not a rhythm or recording-level diagnosis",
    }
    metrics = {
        "test": report,
        "cv_macro_f1": {"folds": cv_scores, "mean": float(np.mean(cv_scores)), "std": float(np.std(cv_scores))},
        "fit_seconds": fit_s,
        "latency": measure_latency(model.predict, test.X),
        "size_bytes": model_size_bytes(bundle),
        "n_train": int(train.X.shape[0]),
        "n_test": int(test.X.shape[0]),
    }
    bundle["metrics"] = metrics
    return TrainResult(kind=kind, bundle=bundle, metrics=metrics)
