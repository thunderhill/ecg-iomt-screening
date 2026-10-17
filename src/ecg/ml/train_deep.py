"""Training for Model D (1D-CNN) and Model E (hybrid: CNN embedding + XGBoost).

Same protocol as the tree models: train on DS1, test once on DS2. Early stopping
uses four DS1 records held out as validation, so DS2 is never seen before the
final test. Five-fold cross-validation is NOT run for these models (CPU budget);
the metrics say so instead of reporting a single split as cross-validation.

Hybrid caveat: the CNN embedding for the hybrid's training beats comes from the
same DS1 beats the CNN was trained on. That keeps DS2 clean, but the trees may
over-trust the embedding. Treat any hybrid advantage with caution.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .. import __version__
from ..features import FEATURE_NAMES
from .beats import SUPPORTED_CLASSES, BeatDataset
from .cnn import CNNConfig, describe, state_size_bytes, train_cnn
from .evaluate import classification_report
from .train import TrainResult, XGBParams, _fit, model_size_bytes

NO_CV = {"mean": None, "std": None, "folds": [], "note": "not run for this model (CPU budget); early stopping used 4 DS1 validation records"}


def _latency(predict_one, n: int = 200) -> dict[str, Any]:
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        predict_one(i)
        times.append(time.perf_counter() - t0)
    return {"single_beat_median_ms": float(np.median(times) * 1000), "note": "single-threaded, on the training machine's CPU"}


def _common_bundle(kind: str, train: BeatDataset, test: BeatDataset, seed: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "classes": SUPPORTED_CLASSES,
        "feature_names": FEATURE_NAMES,
        "feature_medians": np.median(train.X, axis=0).tolist(),
        "feature_ranges": {"p01": np.percentile(train.X, 1, axis=0).tolist(),
                           "p99": np.percentile(train.X, 99, axis=0).tolist()},
        "seed": seed,
        "train_records": sorted(np.unique(train.groups).tolist()),
        "test_records": sorted(np.unique(test.groups).tolist()),
        "package_version": __version__,
        "dataset": "MIT-BIH Arrhythmia Database (DS1 train / DS2 test, 250 Hz, MLII)",
        "label_level": "beat (AAMI N/S/V/F); not a rhythm or recording-level diagnosis",
    }


def train_cnn_and_evaluate(train: BeatDataset, test: BeatDataset, *, seed: int = 42,
                           cfg: CNNConfig = CNNConfig()) -> TrainResult:
    import torch

    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.perf_counter()
    model = train_cnn(train.waveforms, train.X, train.y, train.groups, SUPPORTED_CLASSES, cfg=cfg, seed=seed)
    fit_s = time.perf_counter() - t0
    proba = model.predict_proba(test.waveforms, test.X)
    y_pred = np.array(SUPPORTED_CLASSES)[proba.argmax(axis=1)]
    report = classification_report(test.y, y_pred, SUPPORTED_CLASSES, proba)

    torch.set_num_threads(1)
    latency = _latency(lambda i: model.predict_proba(test.waveforms[i : i + 1], test.X[i : i + 1]))
    model._net = None  # store weights only, not the live module
    bundle = {**_common_bundle("cnn", train, test, seed), "model": model, "params": describe(cfg)}
    metrics = {
        "test": report, "cv_macro_f1": NO_CV, "fit_seconds": fit_s, "latency": latency,
        "size_bytes": state_size_bytes(model), "n_train": int(train.X.shape[0]), "n_test": int(test.X.shape[0]),
        "training_history": model.history,
    }
    bundle["metrics"] = metrics
    return TrainResult(kind="cnn", bundle=bundle, metrics=metrics)


def train_hybrid_and_evaluate(cnn_bundle: dict[str, Any], train: BeatDataset, test: BeatDataset, *,
                              seed: int = 42, xgb: XGBParams = XGBParams()) -> TrainResult:
    cnn = cnn_bundle["model"]
    z_train = np.hstack([cnn.embed(train.waveforms, train.X), train.X])
    z_test = np.hstack([cnn.embed(test.waveforms, test.X), test.X])
    t0 = time.perf_counter()
    model = _fit("xgb", z_train, train.y, seed, None, xgb)  # type: ignore[arg-type]
    fit_s = time.perf_counter() - t0
    model.model.n_jobs = 1

    raw = model.predict_proba(z_test)
    proba = np.zeros((z_test.shape[0], len(SUPPORTED_CLASSES)))
    for j, c in enumerate(SUPPORTED_CLASSES):
        if c in list(model.classes_):
            proba[:, j] = raw[:, list(model.classes_).index(c)]
    y_pred = np.array(SUPPORTED_CLASSES)[proba.argmax(axis=1)]
    report = classification_report(test.y, y_pred, SUPPORTED_CLASSES, proba)

    def one(i: int) -> None:
        z = np.hstack([cnn.embed(test.waveforms[i : i + 1], test.X[i : i + 1]), test.X[i : i + 1]])
        model.predict_proba(z)

    latency = _latency(one)
    cnn._net = None
    emb_dim = z_train.shape[1] - len(FEATURE_NAMES)
    bundle = {
        **_common_bundle("hybrid", train, test, seed),
        "model": model,
        "cnn": cnn,
        "embedding_names": [f"cnn_embedding_{i}" for i in range(emb_dim)],
        "params": {"xgb": xgb.__dict__, "cnn": describe(cnn.config)},
    }
    metrics = {
        "test": report, "cv_macro_f1": NO_CV, "fit_seconds": fit_s, "latency": latency,
        "size_bytes": model_size_bytes({"m": model}) + state_size_bytes(cnn),
        "n_train": int(train.X.shape[0]), "n_test": int(test.X.shape[0]),
        "caveat": "Embeddings for training beats come from the CNN trained on those beats; DS2 stays unseen.",
    }
    bundle["metrics"] = metrics
    return TrainResult(kind="hybrid", bundle=bundle, metrics=metrics)
