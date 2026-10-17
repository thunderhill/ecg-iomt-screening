"""Load trained model bundles and classify beats in a live or offline window.

The beat model output is aggregated into a window-level screening result:
"possible arrhythmia pattern" when at least ``min_ectopic_beats`` beats in the
window are classified as S or V. That describes beat classes the model saw. It
is not a rhythm diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..features import FEATURE_NAMES, extract_beat_features
from .beats import BEAT_WINDOW_S
from .rules import Category

ECTOPIC_CLASSES = ("S", "V")
CLASS_DESCRIPTIONS = {
    "N": "beat resembling the normal/bundle-branch group (AAMI N)",
    "S": "possible supraventricular ectopic beat (AAMI S)",
    "V": "possible ventricular ectopic beat (AAMI V)",
    "F": "possible fusion beat (AAMI F)",
}


class ModelUnavailable(RuntimeError):
    pass


def load_bundle(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ModelUnavailable(f"model file {p} not found; run scripts/train_models.py")
    if p.stem in ("cnn", "hybrid"):
        from .cnn import torch_available

        if not torch_available():
            raise ModelUnavailable("PyTorch is not installed; install it to use the CNN models")
    bundle = joblib.load(p)
    if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
        raise ModelUnavailable("model was trained with a different feature set; retrain it")
    return bundle


def available_models(models_dir: str | Path) -> dict[str, Path]:
    d = Path(models_dir)
    return {p.stem: p for p in sorted(d.glob("*.joblib"))} if d.exists() else {}


@dataclass(frozen=True)
class BeatPredictions:
    beat_indices: np.ndarray
    X: np.ndarray
    waves: np.ndarray  # beat windows, the CNN's input (same window as in training)
    classes: tuple[str, ...]
    predicted: np.ndarray
    proba: np.ndarray  # columns follow ``classes``
    skipped_history: int


def beat_windows(filtered: np.ndarray, indices: np.ndarray, fs: float) -> np.ndarray:
    """Waveform windows around each R, identical to those used for training."""
    pre = int(round(BEAT_WINDOW_S[0] * fs))
    post = int(round(BEAT_WINDOW_S[1] * fs))
    if indices.size == 0:
        return np.empty((0, pre + post), dtype=np.float32)
    return np.stack([filtered[i - pre : i + post] for i in indices]).astype(np.float32)


def bundle_proba(bundle: dict[str, Any], X: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """Class probabilities with columns in ``bundle["classes"]`` order, for any model kind."""
    classes = list(bundle["classes"])
    kind = bundle["kind"]
    if kind == "cnn":
        return bundle["model"].predict_proba(waves, X)
    inputs = np.hstack([bundle["cnn"].embed(waves, X), X]) if kind == "hybrid" else X
    model = bundle["model"]
    raw = model.predict_proba(inputs)
    proba = np.zeros((X.shape[0], len(classes)))
    for j, c in enumerate(classes):
        if c in list(model.classes_):
            proba[:, j] = raw[:, list(model.classes_).index(c)]
    return proba


def classify_beats(bundle: dict[str, Any], filtered: np.ndarray, r_indices: np.ndarray, fs: float) -> BeatPredictions:
    feats = extract_beat_features(filtered, r_indices, fs)
    classes = tuple(bundle["classes"])
    waves = beat_windows(np.asarray(filtered, dtype=float), feats.beat_indices, fs)
    if feats.X.shape[0] == 0:
        return BeatPredictions(feats.beat_indices, feats.X, waves, classes, np.array([], dtype=str),
                               np.empty((0, len(classes))), feats.skipped_history)
    proba = bundle_proba(bundle, feats.X, waves)
    predicted = np.array(classes)[np.argmax(proba, axis=1)]
    return BeatPredictions(feats.beat_indices, feats.X, waves, classes, predicted, proba, feats.skipped_history)


@dataclass(frozen=True)
class ModelScreening:
    model: str
    category: Category
    title: str
    evidence: tuple[str, ...]
    interpretation: str
    limitations: tuple[str, ...]
    class_counts: dict[str, int]
    mean_proba: dict[str, float]


_LIMITATIONS = (
    "Beat classes come from a model trained on 22 MIT-BIH recordings; other devices, leads and people may differ.",
    "On held-out MIT-BIH records, S and F beats were frequently missed (see the Evaluation page).",
    "Model probabilities express model confidence, not clinical certainty.",
    "A beat-level classification is not a rhythm diagnosis.",
)


def summarize_window(kind: str, preds: BeatPredictions, *, min_beats: int = 8, min_ectopic_beats: int = 3) -> ModelScreening:
    n = preds.predicted.size
    counts = {c: int(np.sum(preds.predicted == c)) for c in preds.classes}
    mean_proba = {c: float(preds.proba[:, j].mean()) if n else 0.0 for j, c in enumerate(preds.classes)}
    if n < min_beats:
        return ModelScreening(kind, Category.INSUFFICIENT_DATA, "Insufficient data",
                              (f"{n} beats classified; need {min_beats}",
                               f"{preds.skipped_history} beat(s) skipped: each beat needs 8 previous beats"),
                              "Not enough classified beats for a model-based screening result.",
                              _LIMITATIONS, counts, mean_proba)
    ectopic = sum(counts.get(c, 0) for c in ECTOPIC_CLASSES)
    evidence = (f"Beats classified in window: {n}",) + tuple(
        f"{c}: {counts[c]} beat(s), {CLASS_DESCRIPTIONS.get(c, c)}" for c in preds.classes if counts[c]
    )
    if ectopic >= min_ectopic_beats:
        return ModelScreening(
            kind, Category.ARRHYTHMIA_PATTERN, "Possible arrhythmia pattern",
            evidence + (f"{ectopic} beat(s) classified as S or V (flag threshold: {min_ectopic_beats})",),
            "The model classified several beats as possible ectopic beats. This is a screening signal for review, "
            "not a diagnosis, and the model's S-class sensitivity on held-out data was low.",
            _LIMITATIONS, counts, mean_proba,
        )
    return ModelScreening(
        kind, Category.NO_FLAG, "No configured screening flag detected",
        evidence,
        "Fewer ectopic beats than the flag threshold were classified. This does NOT mean the rhythm is normal: "
        "the model only recognises N/S/V/F beat classes and misses many S and F beats.",
        _LIMITATIONS, counts, mean_proba,
    )
