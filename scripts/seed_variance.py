"""Seed variance of the CNN and hybrid models on DS2.

The deep models have no cross-validation (CPU budget), so a single DS2 number
could be luck. This retrains them with several seeds and records the spread.
It writes models/seed_variance.json and adds it to evaluation_summary.json.
The saved cnn/hybrid models are not changed.

    python scripts/seed_variance.py --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg.ml.beats import load_dataset  # noqa: E402
from ecg.ml.train_deep import train_cnn_and_evaluate, train_hybrid_and_evaluate  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = p.parse_args()
    train = load_dataset(PROCESSED / "beats_DS1.npz")
    test = load_dataset(PROCESSED / "beats_DS2.npz")
    runs: dict[str, list[dict]] = {"cnn": [], "hybrid": []}
    for seed in args.seeds:
        cnn = train_cnn_and_evaluate(train, test, seed=seed)
        hyb = train_hybrid_and_evaluate(cnn.bundle, train, test, seed=seed)
        for kind, res in (("cnn", cnn), ("hybrid", hyb)):
            t = res.metrics["test"]
            runs[kind].append({"seed": seed, "macro_f1": t["macro_f1"],
                               **{f"{c}_sensitivity": m["sensitivity"] for c, m in t["per_class"].items()}})
            print(f"seed {seed} {kind}: macro-F1 {t['macro_f1']:.3f}", flush=True)
    result = {"seeds": args.seeds, "note": "DS2 metrics per training seed; spread shows run-to-run variance.", "models": {}}
    for kind, rows in runs.items():
        f1 = np.array([r["macro_f1"] for r in rows])
        result["models"][kind] = {"runs": rows, "macro_f1_mean": float(f1.mean()), "macro_f1_std": float(f1.std(ddof=1)) if f1.size > 1 else 0.0}
        print(f"{kind}: macro-F1 {f1.mean():.3f} ± {result['models'][kind]['macro_f1_std']:.3f} over seeds {args.seeds}")
    (MODELS / "seed_variance.json").write_text(json.dumps(result, indent=2))
    summary_path = MODELS / "evaluation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["seed_variance"] = result
        summary_path.write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
