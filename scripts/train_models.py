"""Train and evaluate the screening models on MIT-BIH (DS1 train, DS2 test).

Requires prepared data:  python scripts/prepare_mitbih.py --download

Outputs (in models/):
  rf.joblib, xgb.joblib        model bundles (XGBoost only when installed)
  metrics_<kind>.json          per-model evaluation
  evaluation_summary.json      everything the dashboard's Evaluation page shows

    python scripts/train_models.py               # RF (+ XGBoost when available)
    python scripts/train_models.py --models rf   # only RF
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg.config import AppConfig, PreprocessConfig, RRPlausibility, ScreeningConfig  # noqa: E402
from ecg.datasets import DS1, DS2, load_prepared  # noqa: E402
from ecg.detection_eval import evaluate_detection  # noqa: E402
from ecg.hr import rr_intervals  # noqa: E402
from ecg.ml.beats import SUPPORTED_CLASSES, build_dataset, load_dataset, save_dataset  # noqa: E402
from ecg.ml.rules import screen_heart_rate  # noqa: E402
from ecg.ml.cnn import torch_available  # noqa: E402
from ecg.ml.train import train_and_evaluate, xgboost_available  # noqa: E402
from ecg.ml.train_deep import train_cnn_and_evaluate, train_hybrid_and_evaluate  # noqa: E402
from ecg.preprocess import preprocess  # noqa: E402
from ecg.rpeak import detect_rpeaks  # noqa: E402
from ecg.storage import Storage  # noqa: E402
from ecg.xai.importance import global_importance  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def _dataset(name: str, records: tuple[str, ...], rebuild: bool):
    path = PROCESSED / f"beats_{name}.npz"
    if path.exists() and not rebuild:
        return load_dataset(path)
    ds = build_dataset(PROCESSED / "mitdb", records)
    save_dataset(path, ds)
    return ds


def rules_agreement(records: tuple[str, ...], window_s: float = 60.0) -> dict:
    """Model A agreement: rule on detected beats vs the same rule on reference beats.

    MIT-BIH has no tachycardia/bradycardia labels, so the reference is the SAME rule
    applied to the annotated beat times. This measures whether our detection and
    RR pipeline reproduce reference-derived flags. It is not diagnostic accuracy.
    """
    cfg = ScreeningConfig()
    rr_rules = RRPlausibility()
    cats = ("possible_elevated_heart_rate", "possible_low_heart_rate", "no_configured_flag", "insufficient_data")
    confusion = {r: {p: 0 for p in cats} for r in cats}
    for name in records:
        rec = load_prepared(PROCESSED / "mitdb" / f"{name}.npz")
        filtered = preprocess(rec.signal_mv, PreprocessConfig(fs=rec.fs, mode="causal"))
        det = detect_rpeaks(filtered, rec.fs, zero_phase=False)
        ref_series = rr_intervals(np.unique(rec.beat_samples) / rec.fs, rr_rules)
        det_series = rr_intervals(det.indices / rec.fs, rr_rules)
        duration = rec.signal_mv.size / rec.fs
        for end in np.arange(window_s, duration + 1e-9, window_s):
            ref = screen_heart_rate(ref_series, float(end), cfg).category.value
            got = screen_heart_rate(det_series, float(end), cfg).category.value
            confusion[ref][got] += 1
    total = sum(sum(row.values()) for row in confusion.values())
    agree = sum(confusion[c][c] for c in cats)
    return {
        "description": "Rule applied to detected beats vs the same rule applied to reference-annotated beats",
        "window_s": window_s,
        "thresholds": {"brady_bpm": cfg.brady_bpm, "tachy_bpm": cfg.tachy_bpm, "min_duration_s": cfg.min_duration_s},
        "records": list(records),
        "windows": total,
        "agreement": agree / total if total else None,
        "confusion": confusion,
        "note": "Reference-derived flags, not clinical tachycardia/bradycardia labels.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=["rf", "xgb", "cnn", "hybrid"], choices=["rf", "xgb", "cnn", "hybrid"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--rebuild", action="store_true", help="rebuild beat datasets from prepared records")
    args = p.parse_args()

    if not (PROCESSED / "mitdb" / "100.npz").exists():
        sys.exit("Prepared MIT-BIH data not found. Run: python scripts/prepare_mitbih.py --download")

    np.random.seed(args.seed)
    MODELS.mkdir(exist_ok=True)
    train = _dataset("DS1", DS1, args.rebuild)
    test = _dataset("DS2", DS2, args.rebuild)
    print(f"train beats {train.X.shape[0]} ({len(DS1)} records), test beats {test.X.shape[0]} ({len(DS2)} records)")

    storage = Storage(AppConfig().data_dir if Path(AppConfig().data_dir).is_absolute() else ROOT / AppConfig().data_dir)
    summary: dict = {
        "generated_at": time.time(),
        "dataset": {
            "name": "MIT-BIH Arrhythmia Database",
            "split": "DS1 (train) / DS2 (test), record-wise = patient-wise (de Chazal 2004)",
            "fs": 250.0,
            "lead": "MLII",
            "classes": list(SUPPORTED_CLASSES),
            "excluded_note": "Paced records 102/104/107/217 excluded; Q class excluded (15 beats in total).",
            "train_counts": dict(zip(*[a.tolist() for a in np.unique(train.y, return_counts=True)])),
            "test_counts": dict(zip(*[a.tolist() for a in np.unique(test.y, return_counts=True)])),
            "excluded_train": train.excluded,
            "excluded_test": test.excluded,
        },
        "models": {},
    }
    previous = MODELS / "evaluation_summary.json"
    if previous.exists():
        # Keep results of models not retrained in this run (e.g. --models cnn).
        kept = json.loads(previous.read_text()).get("models", {})
        summary["models"] = {k: v for k, v in kept.items() if k not in args.models}
        if "seed_variance" in json.loads(previous.read_text()):
            summary["seed_variance"] = json.loads(previous.read_text())["seed_variance"]

    for split_name, ds in (("DS1", train), ("DS2", test)):
        tp = sum(d["tp"] for d in ds.detection.values())
        fp = sum(d["fp"] for d in ds.detection.values())
        fn = sum(d["fn"] for d in ds.detection.values())
        summary.setdefault("rpeak_detection", {})[split_name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "sensitivity": tp / (tp + fn), "ppv": tp / (tp + fp),
            "f1": 2 * tp / (2 * tp + fp + fn),
            "per_record": ds.detection,
            "mode": "causal (live) filtering and detection, 150 ms tolerance",
        }

    cnn_bundle = None
    for kind in args.models:
        if kind in ("xgb", "hybrid") and not xgboost_available():
            print(f"XGBoost is not installed; skipping {kind}.")
            summary["models"][kind] = {"available": False, "reason": "xgboost not installed"}
            continue
        if kind in ("cnn", "hybrid") and not torch_available():
            print(f"PyTorch is not installed; skipping {kind}.")
            summary["models"][kind] = {"available": False, "reason": "PyTorch not installed"}
            continue
        if kind == "hybrid" and cnn_bundle is None:
            print("The hybrid needs the CNN trained in the same run; skipping it.")
            summary["models"][kind] = {"available": False, "reason": "CNN not trained in this run"}
            continue
        print(f"training {kind} ...", flush=True)
        if kind == "cnn":
            result = train_cnn_and_evaluate(train, test, seed=args.seed)
            cnn_bundle = result.bundle
        elif kind == "hybrid":
            result = train_hybrid_and_evaluate(cnn_bundle, train, test, seed=args.seed)
        else:
            result = train_and_evaluate(kind, train, test, seed=args.seed, cv_folds=args.cv_folds)
        path = MODELS / f"{kind}.joblib"
        joblib.dump(result.bundle, path, compress=3)
        (MODELS / f"metrics_{kind}.json").write_text(json.dumps(result.metrics, indent=2))
        storage.register_model(
            model_id=kind, kind=kind, path=str(path), size_bytes=path.stat().st_size,
            metrics=result.metrics, seed=args.seed, dataset="mitdb DS1/DS2",
        )
        t = result.metrics["test"]
        cv = result.metrics["cv_macro_f1"]
        cv_text = f"{cv['mean']:.3f} ± {cv['std']:.3f}" if cv["mean"] is not None else "not run"
        print(f"  {kind}: DS2 macro-F1 {t['macro_f1']:.3f}  accuracy {t['accuracy']:.3f}  CV macro-F1 {cv_text}")
        for c, m in t["per_class"].items():
            print(f"     {c}: Se {m['sensitivity']:.3f}  Sp {m['specificity']:.3f}  PPV {m['precision']:.3f}  n={m['support']}")
        summary["models"][kind] = {"available": True, **result.metrics, "params": result.bundle["params"]}
        if kind in ("rf", "xgb"):
            print(f"  permutation importance for {kind} on held-out DS2 beats ...", flush=True)
            imp = global_importance(result.bundle, test.X, test.y, seed=args.seed)
            (MODELS / f"importance_{kind}.json").write_text(json.dumps(imp, indent=2))
            summary["models"][kind]["importance"] = imp

    print("evaluating rule-based screening agreement (Model A) ...", flush=True)
    summary["rules"] = rules_agreement(DS2)
    print(f"  agreement {summary['rules']['agreement']:.3f} over {summary['rules']['windows']} windows")
    (MODELS / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {MODELS / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
