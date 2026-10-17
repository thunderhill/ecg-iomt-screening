"""Global feature importance.

* Permutation importance (preferred): the drop in macro-F1 on HELD-OUT beats
  when one feature is shuffled. It is model-agnostic and measured on data the
  model did not train on.
* Impurity importance (RF/XGBoost built-in): reported for comparison only. It
  is computed on training data and is biased toward high-cardinality features.

Importance shows which features the model relies on. It is an association, not
a causal or physiological explanation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def global_importance(
    bundle: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int = 6000,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    # Stratified subsample keeps rare classes represented while bounding runtime.
    idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        take = min(members.size, max(50, int(max_samples * members.size / y.size)))
        idx.extend(rng.choice(members, size=take, replace=False))
    idx = np.array(sorted(idx))
    Xs, ys = X[idx], y[idx]
    model = bundle["model"]
    labels = list(bundle["classes"])

    def score(X_):
        return f1_score(ys, model.predict(X_), labels=labels, average="macro", zero_division=0)

    # Permutation importance: shuffle one column at a time and measure the score drop.
    # Implemented directly so it works with any object that has ``predict``.
    base = score(Xs)
    names = list(bundle["feature_names"])
    drops = np.zeros((len(names), n_repeats))
    for j in range(len(names)):
        for r in range(n_repeats):
            shuffled = Xs.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            drops[j, r] = base - score(shuffled)
    perm = sorted(
        ({"feature": n, "mean": float(drops[j].mean()), "std": float(drops[j].std())} for j, n in enumerate(names)),
        key=lambda d: -d["mean"],
    )
    impurity = None
    inner = getattr(model, "model", model)
    if hasattr(inner, "feature_importances_"):
        impurity = sorted(
            ({"feature": n, "value": float(v)} for n, v in zip(names, inner.feature_importances_)),
            key=lambda d: -d["value"],
        )
    return {
        "method": "permutation importance (drop in macro-F1 on held-out DS2 beats)",
        "n_samples": int(idx.size),
        "n_repeats": n_repeats,
        "baseline_macro_f1": float(base),
        "permutation": perm,
        "impurity": impurity,
        "note": "Importance reflects model reliance on a feature, not a causal or physiological mechanism.",
    }
