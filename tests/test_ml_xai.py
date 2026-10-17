from pathlib import Path

import numpy as np
import pytest

from ecg.config import PreprocessConfig
from ecg.features import FEATURE_NAMES, MIN_HISTORY, POST_R_S, extract_beat_features
from ecg.ml.beats import _match_labels
from ecg.ml.evaluate import classification_report
from ecg.ml.inference import ModelUnavailable, classify_beats, load_bundle, summarize_window
from ecg.ml.rules import Category
from ecg.preprocess import preprocess
from ecg.rpeak import detect_rpeaks
from ecg.synthetic import generate_ecg
from ecg.xai.reliability import reliability_warnings

MODELS = Path(__file__).resolve().parents[1] / "models"
needs_models = pytest.mark.skipif(not (MODELS / "rf.joblib").exists(), reason="run scripts/train_models.py first")


def _beats(seed=0, duration=40.0):
    rec = generate_ecg(duration, fs=250.0, seed=seed)
    filtered = preprocess(rec.signal, PreprocessConfig(fs=250.0, mode="causal"))
    det = detect_rpeaks(filtered, 250.0, zero_phase=False)
    return filtered, det.indices


# ---------- features ----------

def test_feature_shape_and_history_rule():
    filtered, r = _beats()
    f = extract_beat_features(filtered, r, 250.0)
    assert f.X.shape == (f.beat_indices.size, len(FEATURE_NAMES))
    assert f.skipped_history == min(r.size, MIN_HISTORY + 1)
    assert np.all(np.isfinite(f.X))


def test_features_are_causal():
    """Changing the signal after beat i's window must not change beat i's features."""
    filtered, r = _beats(seed=1)
    full = extract_beat_features(filtered, r, 250.0)
    i = full.beat_indices.size // 2
    target = full.beat_indices[i]
    cut = target + int(round(POST_R_S * 250.0)) + 1
    tampered = filtered.copy()
    tampered[cut:] = np.random.default_rng(0).normal(size=tampered.size - cut) * 5
    # Later beats are dropped as well: features may only use beats up to i.
    r_trunc = r[r <= target]
    part = extract_beat_features(tampered, r_trunc, 250.0)
    j = np.flatnonzero(part.beat_indices == target)[0]
    np.testing.assert_allclose(part.X[j], full.X[i])


def test_features_reject_unsorted_peaks():
    filtered, r = _beats()
    with pytest.raises(ValueError):
        extract_beat_features(filtered, r[::-1], 250.0)


def test_too_few_beats_yields_no_features():
    filtered, r = _beats()
    f = extract_beat_features(filtered, r[:MIN_HISTORY], 250.0)
    assert f.X.shape[0] == 0


# ---------- label matching ----------

def test_match_labels_within_tolerance_only():
    labels = _match_labels(np.array([100, 300, 1000]), np.array([102, 290]), np.array(["N", "V"]), tol=20)
    assert labels.tolist() == ["N", "V", ""]


# ---------- metrics ----------

def test_classification_report_specificity_and_auc():
    y = np.array(["N", "N", "V", "V", "S", "F"])
    pred = np.array(["N", "V", "V", "V", "N", "F"])
    proba = np.eye(4)[[0, 2, 2, 2, 0, 3]]
    rep = classification_report(y, pred, ("N", "S", "V", "F"), proba)
    assert rep["per_class"]["V"]["sensitivity"] == 1.0
    assert rep["per_class"]["V"]["specificity"] == pytest.approx(3 / 4)
    assert rep["roc_auc_ovr_macro"] is not None


def test_auc_omitted_when_class_missing():
    y = np.array(["N", "V"])
    rep = classification_report(y, y, ("N", "S", "V", "F"), np.eye(4)[[0, 2]])
    assert rep["roc_auc_ovr_macro"] is None
    assert "not every class" in rep["roc_auc_note"]


# ---------- inference ----------

def test_missing_model_raises_model_unavailable(tmp_path):
    with pytest.raises(ModelUnavailable):
        load_bundle(tmp_path / "nope.joblib")


@needs_models
@pytest.mark.parametrize("kind", ["rf", "xgb", "cnn", "hybrid"])
def test_classify_and_summarize(kind):
    path = MODELS / f"{kind}.joblib"
    if not path.exists():
        pytest.skip(f"{kind} not trained")
    bundle = load_bundle(path)
    filtered, r = _beats(seed=2, duration=60.0)
    preds = classify_beats(bundle, filtered, r, 250.0)
    assert preds.proba.shape == (preds.predicted.size, len(bundle["classes"]))
    np.testing.assert_allclose(preds.proba.sum(axis=1), 1.0, atol=1e-5)
    summary = summarize_window(kind, preds)
    assert summary.category in (Category.NO_FLAG, Category.ARRHYTHMIA_PATTERN, Category.INSUFFICIENT_DATA)
    # Never claims "normal".
    assert "normal" not in summary.title.lower()


def test_summary_insufficient_with_few_beats():
    from ecg.ml.inference import BeatPredictions

    preds = BeatPredictions(np.array([1, 2]), np.zeros((2, 15)), np.zeros((2, 124)), ("N", "S", "V", "F"),
                            np.array(["N", "N"]), np.tile([1.0, 0, 0, 0], (2, 1)), 9)
    assert summarize_window("rf", preds).category == Category.INSUFFICIENT_DATA


# ---------- XAI ----------

@needs_models
def test_xgb_local_explanation_is_additive():
    import xgboost

    from ecg.xai.local import explain_beat

    path = MODELS / "xgb.joblib"
    if not path.exists():
        pytest.skip("xgb not trained")
    bundle = load_bundle(path)
    filtered, r = _beats(seed=3, duration=60.0)
    x = classify_beats(bundle, filtered, r, 250.0).X[0]
    exp = explain_beat(bundle, x, top_k=len(FEATURE_NAMES))
    margin = bundle["model"].model.get_booster().predict(xgboost.DMatrix(x.reshape(1, -1)), output_margin=True)[0]
    k = list(bundle["model"].classes_).index(exp["predicted_class"])
    total = sum(c["contribution"] for c in exp["contributions"]) + exp["baseline"]
    assert total == pytest.approx(float(margin[k]), abs=1e-3)


@needs_models
def test_rf_occlusion_explanation_runs():
    from ecg.xai.local import explain_beat

    bundle = load_bundle(MODELS / "rf.joblib")
    filtered, r = _beats(seed=4, duration=60.0)
    x = classify_beats(bundle, filtered, r, 250.0).X[0]
    exp = explain_beat(bundle, x)
    assert exp["contributions"] and "association" in exp["note"]


def test_reliability_without_model():
    w = reliability_warnings(quality_status="good", n_beats=50, min_beats=8, bundle=None)
    assert any("No trained model" in m for m in w)


def test_reliability_flags_simulated_and_poor_quality():
    bundle = {"classes": ("N",), "feature_names": FEATURE_NAMES,
              "feature_ranges": {"p01": [0.0] * 15, "p99": [1.0] * 15}}
    w = reliability_warnings(quality_status="poor", n_beats=2, min_beats=8, bundle=bundle,
                             X=np.full((5, 15), 10.0), source="SIMULATED")
    text = " ".join(w)
    assert "poor" in text and "SIMULATED" in text and "training range" in text and "Only 2 beats" in text


@pytest.mark.parametrize("kind", ["cnn", "hybrid"])
def test_deep_model_explanations(kind):
    path = MODELS / f"{kind}.joblib"
    if not path.exists():
        pytest.skip(f"{kind} not trained")
    from ecg.xai.local import explain_beat

    bundle = load_bundle(path)
    filtered, r = _beats(seed=5, duration=60.0)
    preds = classify_beats(bundle, filtered, r, 250.0)
    exp = explain_beat(bundle, preds.X[0], wave=preds.waves[0])
    assert exp["predicted_class"] == preds.predicted[0]
    assert exp["contributions"]
    if kind == "cnn":
        assert len(exp["saliency"]) == preds.waves.shape[1]
        assert max(exp["saliency"]) == pytest.approx(1.0)
    else:
        assert any(c["feature"] == "cnn_embedding" for c in exp["contributions"]) or len(exp["contributions"]) == 6
