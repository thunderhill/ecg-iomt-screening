"""Local (single-beat) feature contributions.

Method by model:
* XGBoost: exact TreeSHAP from XGBoost itself (``pred_contribs``). Values are
  additive contributions to the class's raw score (log-odds), relative to the
  model's average output.
* Random Forest: TreeSHAP when the optional ``shap`` package is installed.
  Otherwise an OCCLUSION approximation: each feature is replaced by its
  training median, and the change in the predicted class probability is
  reported. Occlusion is not additive and ignores feature interactions.

Contributions describe how the MODEL used the inputs. They are associations,
not causes, and not evidence that the model's reasoning is medically valid.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _shap_available() -> bool:
    try:
        import shap  # noqa: F401
    except ImportError:
        return False
    return True


def _explain_cnn(bundle: dict[str, Any], x: np.ndarray, wave: np.ndarray, top_k: int) -> dict[str, Any]:
    from ..ml.cnn import RR_INPUTS

    model = bundle["model"]
    classes = list(bundle["classes"])
    proba = model.predict_proba(wave.reshape(1, -1), x.reshape(1, -1))[0]
    k = int(np.argmax(proba))
    att_w, att_r = model.integrated_gradients(wave, x, k)
    names = list(bundle["feature_names"])
    items = [{"feature": n, "value": float(x[names.index(n)]), "contribution": float(a)} for n, a in zip(RR_INPUTS, att_r)]
    items.append({"feature": "waveform_shape", "value": float("nan"), "contribution": float(att_w.sum())})
    items.sort(key=lambda d: -abs(d["contribution"]))
    for it in items:
        it["direction"] = "toward" if it["contribution"] > 0 else "away from"
        if np.isnan(it["value"]):
            it["value"] = 0.0
    weight = np.abs(att_w)
    return {
        "predicted_class": classes[k],
        "probability": float(proba[k]),
        "probabilities": {c: float(p) for c, p in zip(classes, proba)},
        "method": "Integrated gradients (64 steps; baseline: flat waveform, median RR); units: logit of the predicted class",
        "baseline": None,
        "contributions": items[:top_k],
        "saliency": (weight / weight.max()).round(4).tolist() if weight.max() > 0 else weight.tolist(),
        "note": ("Attributions show where the network's output was sensitive. Saliency maps are exploratory and do not "
                 "show that the model's reasoning is physiologically valid."),
    }


def explain_beat(bundle: dict[str, Any], x: np.ndarray, *, top_k: int = 6, wave: np.ndarray | None = None) -> dict[str, Any]:
    """Explain the predicted class for one beat (features ``x``; ``wave`` for CNN models)."""
    if bundle["kind"] == "cnn":
        if wave is None:
            raise ValueError("CNN explanations need the beat waveform")
        return _explain_cnn(bundle, np.asarray(x, dtype=float), np.asarray(wave, dtype=np.float32), top_k)
    x = np.asarray(x, dtype=float).reshape(1, -1)
    model = bundle["model"]
    names = list(bundle["feature_names"])
    if bundle["kind"] == "hybrid":
        if wave is None:
            raise ValueError("hybrid explanations need the beat waveform")
        x = np.hstack([bundle["cnn"].embed(np.asarray(wave, dtype=np.float32).reshape(1, -1), x), x])
        names = list(bundle["embedding_names"]) + names
    classes = list(model.classes_)
    proba = model.predict_proba(x)[0]
    k = int(np.argmax(proba))
    predicted = str(classes[k])

    if bundle["kind"] in ("xgb", "hybrid"):
        import xgboost

        contribs = model.model.get_booster().predict(xgboost.DMatrix(x), pred_contribs=True)[0]
        values = contribs[k, :-1]  # last column is the bias term
        method = "TreeSHAP (exact, XGBoost native); units: log-odds of the predicted class"
        baseline = float(contribs[k, -1])
        if bundle["kind"] == "hybrid":
            # SHAP values are additive, so the embedding dimensions can be summed into one input.
            n_emb = len(bundle["embedding_names"])
            values = np.concatenate([[values[:n_emb].sum()], values[n_emb:]])
            x = np.hstack([[[0.0]], x[:, n_emb:]])
            names = ["cnn_embedding"] + list(bundle["feature_names"])
            method += "; the 32 CNN embedding dimensions are summed into one contribution"
    elif _shap_available():
        import shap

        sv = shap.TreeExplainer(model).shap_values(x)
        arr = np.asarray(sv)
        values = arr[k][0] if arr.ndim == 3 and arr.shape[0] == len(classes) else arr[0, :, k]
        method = "TreeSHAP (shap package); units: probability of the predicted class"
        baseline = None
    else:
        if "feature_medians" not in bundle:
            raise ValueError("model bundle has no training medians; retrain it")
        medians = np.asarray(bundle["feature_medians"])
        values = np.zeros(len(names))
        for j in range(len(names)):
            occluded = x.copy()
            occluded[0, j] = medians[j]
            values[j] = proba[k] - model.predict_proba(occluded)[0][k]
        method = ("Occlusion approximation (feature replaced by its training median); "
                  "units: change in predicted-class probability; not additive")
        baseline = None

    order = np.argsort(-np.abs(values))[:top_k]
    return {
        "predicted_class": predicted,
        "probability": float(proba[k]),
        "probabilities": {str(c): float(p) for c, p in zip(classes, proba)},
        "method": method,
        "baseline": baseline,
        "contributions": [
            {"feature": names[j], "value": float(x[0, j]), "contribution": float(values[j]),
             "direction": "toward" if values[j] > 0 else "away from"}
            for j in order
        ],
        "note": "Contributions show how the model used each input. They are associations, not causes.",
    }
