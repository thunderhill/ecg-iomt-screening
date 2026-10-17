"""Whole-recording (OFFLINE) analysis, used by the History page and imports.

Two filtered versions of the signal are computed on purpose:
* zero-phase filtering for R-peak detection, display, HR and HRV. Offline, the
  whole recording is available, so a phase-free result is preferable.
* causal filtering for the beat-model features, because the models were
  trained on causally filtered signals. Feeding them zero-phase features would
  silently change the input distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import AppConfig, PreprocessConfig
from .hr import heart_rate_summary, rr_intervals
from .hrv import time_domain_hrv
from .ml.rules import screen_heart_rate
from .preprocess import preprocess
from .quality import assess_quality
from .rpeak import detect_rpeaks

QUALITY_SEGMENT_S = 10.0


def analyze_recording(mv: np.ndarray, fs: float, cfg: AppConfig, *, bundle: dict | None = None,
                      model_kind: str | None = None, source: str = "OFFLINE") -> dict[str, Any]:
    x = np.asarray(mv, dtype=float)
    if x.size < int(2 * fs):
        raise ValueError("recording shorter than 2 s cannot be analysed")
    zp_cfg = PreprocessConfig(**{**cfg.preprocess.__dict__, "fs": fs, "mode": "zero_phase"})
    filtered = preprocess(x, zp_cfg)
    det = detect_rpeaks(filtered, fs, cfg.rpeak, zero_phase=True)
    series = rr_intervals(det.times_s, cfg.rr)
    duration = x.size / fs
    hr = heart_rate_summary(series, duration, duration, cfg.screening.min_valid_rr)
    hrv = time_domain_hrv(series, cfg.hrv)

    seg = int(QUALITY_SEGMENT_S * fs)
    quality_segments = []
    for start in range(0, x.size - seg + 1, seg):
        q = assess_quality(filtered[start : start + seg], fs, cfg.quality)
        quality_segments.append({"start_s": start / fs, "status": q.status, "reasons": list(q.reasons)})
    statuses = [q["status"] for q in quality_segments]
    overall_quality = "poor" if statuses.count("poor") > len(statuses) / 2 else ("good" if statuses and all(s == "good" for s in statuses) else "fair")
    rules = screen_heart_rate(series, duration, cfg.screening, signal_status=overall_quality)

    result: dict[str, Any] = {
        "source": source,
        "fs": fs,
        "duration_s": duration,
        "filtering": "zero-phase (offline) for display, R peaks, HR and HRV; causal for model features",
        "r_peaks_s": det.times_s.round(4).tolist(),
        "detector_notes": list(det.notes),
        "rr": {
            "times_s": series.times_s.round(4).tolist(),
            "rr_s": series.rr_s.round(4).tolist(),
            "flags": [list(f) for f in series.flags],
            "flagged_fraction": series.flagged_fraction,
        },
        "heart_rate": {**{k: getattr(hr, k) for k in ("mean_bpm", "min_bpm", "max_bpm", "valid_rr_count", "flagged_rr_count")},
                       "warnings": list(hr.warnings)},
        "hrv": {**{k: getattr(hrv, k) for k in ("available", "mean_nn_ms", "sdnn_ms", "rmssd_ms", "pnn50_pct", "nn_count")},
                "warnings": list(hrv.warnings)},
        "quality": {"overall": overall_quality, "segments": quality_segments},
        "rules": {"category": rules.category.value, "title": rules.title, "evidence": list(rules.evidence),
                  "interpretation": rules.interpretation, "limitations": list(rules.limitations),
                  "note": "Rule evaluated over the final analysis window of the recording."},
    }
    if bundle is not None:
        from .ml.inference import classify_beats

        causal = preprocess(x, PreprocessConfig(**{**cfg.preprocess.__dict__, "fs": fs, "mode": "causal"}))
        causal_peaks = detect_rpeaks(causal, fs, cfg.rpeak, zero_phase=False).indices
        preds = classify_beats(bundle, causal, causal_peaks, fs)
        result["model"] = {
            "kind": model_kind,
            "beat_times_s": (preds.beat_indices / fs).round(4).tolist(),
            "predicted": preds.predicted.tolist(),
            "class_counts": {c: int(np.sum(preds.predicted == c)) for c in preds.classes},
            "skipped_history": preds.skipped_history,
        }
    return result
