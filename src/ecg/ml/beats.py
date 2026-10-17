"""Build a beat-level dataset from prepared MIT-BIH records.

End-to-end, like the live system: the signal is filtered causally, beats come
from OUR detector (not the reference annotations), and each detected beat takes
the label of the reference beat it matches within 150 ms. Detections with no
matching reference beat are excluded and counted, and so are reference beats
the detector missed. Both counts are reported. Their errors are not hidden by
training on perfect beat locations.

Labels are beat-level AAMI classes. They are not rhythm or recording-level
diagnoses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import PreprocessConfig, RPeakConfig
from ..datasets import load_prepared
from ..detection_eval import evaluate_detection
from ..features import FEATURE_NAMES, extract_beat_features
from ..preprocess import preprocess
from ..rpeak import detect_rpeaks

SUPPORTED_CLASSES: tuple[str, ...] = ("N", "S", "V", "F")  # Q excluded: 15 beats in all of MIT-BIH
MATCH_TOLERANCE_S = 0.150
BEAT_WINDOW_S = (0.25, 0.25)  # waveform kept per beat for the CNN (before, after R)


@dataclass
class BeatDataset:
    X: np.ndarray
    y: np.ndarray  # class labels (str)
    groups: np.ndarray  # record name per beat (= patient for MIT-BIH)
    beat_index: np.ndarray  # R index within its record
    waveforms: np.ndarray  # (n, window) causally filtered beat segments for the CNN
    feature_names: tuple[str, ...] = FEATURE_NAMES
    detection: dict = field(default_factory=dict)  # per-record detection scores
    excluded: dict = field(default_factory=dict)  # counts of excluded beats by reason


def _match_labels(detected: np.ndarray, ref_idx: np.ndarray, ref_lab: np.ndarray, tol: int) -> np.ndarray:
    """Label for each detected beat: the nearest reference beat within ``tol``, else ""."""
    out = np.full(detected.size, "", dtype=object)
    if ref_idx.size == 0:
        return out
    pos = np.searchsorted(ref_idx, detected)
    for k, d in enumerate(detected):
        best, best_dist = -1, tol + 1
        for j in (pos[k] - 1, pos[k]):
            if 0 <= j < ref_idx.size:
                dist = abs(int(ref_idx[j]) - int(d))
                if dist < best_dist:
                    best, best_dist = j, dist
        if best >= 0 and best_dist <= tol:
            out[k] = ref_lab[best]
    return out


def build_dataset(
    processed_dir: str | Path,
    records: tuple[str, ...],
    pre_cfg: PreprocessConfig | None = None,
    rpeak_cfg: RPeakConfig | None = None,
) -> BeatDataset:
    pre_cfg = pre_cfg or PreprocessConfig(fs=250.0, mode="causal")
    if pre_cfg.mode != "causal":
        raise ValueError("training features must use causal filtering to match live inference")
    rpeak_cfg = rpeak_cfg or RPeakConfig()
    Xs, ys, gs, idxs, waves = [], [], [], [], []
    detection: dict[str, dict] = {}
    excluded = {"unmatched_detection": 0, "unsupported_label": 0, "insufficient_history": 0, "edge": 0}

    for name in records:
        rec = load_prepared(Path(processed_dir) / f"{name}.npz")
        if rec.fs != pre_cfg.fs:
            raise ValueError(f"record {name} is {rec.fs} Hz, preprocessing expects {pre_cfg.fs} Hz")
        filtered = preprocess(rec.signal_mv, pre_cfg)
        det = detect_rpeaks(filtered, rec.fs, rpeak_cfg, zero_phase=False)
        scores = evaluate_detection(rec.beat_samples, det.indices, rec.fs, MATCH_TOLERANCE_S)
        detection[name] = scores.__dict__

        feats = extract_beat_features(filtered, det.indices, rec.fs)
        excluded["insufficient_history"] += feats.skipped_history
        excluded["edge"] += feats.skipped_edge
        tol = int(round(MATCH_TOLERANCE_S * rec.fs))
        labels = _match_labels(feats.beat_indices, rec.beat_samples, rec.beat_labels, tol)

        unmatched = labels == ""
        unsupported = ~unmatched & ~np.isin(labels, SUPPORTED_CLASSES)
        excluded["unmatched_detection"] += int(unmatched.sum())
        excluded["unsupported_label"] += int(unsupported.sum())
        keep = ~unmatched & ~unsupported

        pre = int(round(BEAT_WINDOW_S[0] * rec.fs))
        post = int(round(BEAT_WINDOW_S[1] * rec.fs))
        kept_idx = feats.beat_indices[keep]
        Xs.append(feats.X[keep])
        ys.append(labels[keep].astype(str))
        gs.append(np.full(kept_idx.size, name))
        idxs.append(kept_idx)
        waves.append(np.stack([filtered[i - pre : i + post] for i in kept_idx]) if kept_idx.size else
                     np.empty((0, pre + post)))

    return BeatDataset(
        X=np.vstack(Xs),
        y=np.concatenate(ys),
        groups=np.concatenate(gs),
        beat_index=np.concatenate(idxs),
        waveforms=np.vstack(waves).astype(np.float32),
        detection=detection,
        excluded=excluded,
    )


def save_dataset(path: str | Path, ds: BeatDataset) -> None:
    import json

    np.savez_compressed(
        path, X=ds.X, y=ds.y, groups=ds.groups, beat_index=ds.beat_index, waveforms=ds.waveforms,
        feature_names=np.array(ds.feature_names), detection=json.dumps(ds.detection), excluded=json.dumps(ds.excluded),
    )


def load_dataset(path: str | Path) -> BeatDataset:
    import json

    d = np.load(path, allow_pickle=False)
    return BeatDataset(
        X=d["X"], y=d["y"], groups=d["groups"], beat_index=d["beat_index"], waveforms=d["waveforms"],
        feature_names=tuple(d["feature_names"].tolist()),
        detection=json.loads(str(d["detection"])), excluded=json.loads(str(d["excluded"])),
    )
