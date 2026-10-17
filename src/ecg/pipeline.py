"""Live processing pipeline: packets in, analysis snapshot out.

Flow for each accepted packet:
  counts -> mV (calibration) -> causal filter (stateful) -> current segment
  -> periodic R-peak detection on the trailing window
  -> confirmed R peaks (kept only once safely behind the live edge)
  -> RR intervals (flagged, not dropped) -> heart-rate summary -> HRV
  -> signal quality -> rule-based and model screening.

Timeline and gaps: every sample has an index on the DEVICE timeline. When
packets are lost, the timeline advances by the number of lost samples (known
exactly, because every packet carries the same number of samples), and the gap
is recorded. Samples are NEVER fabricated to fill it. Filtering, detection, the
displayed waveform and saved recordings use only the current contiguous segment
(the samples since the last gap). Confirmed beats and RR history survive the
gap; any RR interval that spans it is flagged and excluded. A device restart
starts a new timeline, so everything is reset.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Callable

import numpy as np

from .config import AppConfig
from .hr import heart_rate_summary, rr_intervals
from .hrv import time_domain_hrv
from .ingest import StreamReceiver
from .ml.rules import screen_heart_rate
from .preprocess import StreamingFilter
from .quality import assess_quality
from .rpeak import detect_rpeaks

# Peaks closer than this to the live edge are not yet confirmed: the detector
# sees the window edge and its estimates there are less reliable.
CONFIRM_MARGIN_S = 1.0
DETECT_EVERY_S = 1.0
PEAK_HISTORY_S = 600.0
MAX_GAPS = 1000


class LivePipeline:
    """Owns one device stream. Not thread-safe: call from a single task or lock."""

    def __init__(self, cfg: AppConfig, *, source: str, clock: Callable[[], float] = time.monotonic):
        if source not in ("LIVE", "SIMULATED", "OFFLINE"):
            raise ValueError(f"unknown source {source!r}")
        self.cfg = cfg
        self.source = source
        self.receiver = StreamReceiver(fs=cfg.sample_rate_hz, max_samples=cfg.buffer_max_samples, clock=clock)
        self._filter = StreamingFilter(cfg.preprocess)
        self._window_len = int(round(cfg.rolling_window_s * cfg.sample_rate_hz))
        # Current contiguous segment (bounded): filtered mV and the raw counts behind it.
        self._filtered: deque[float] = deque(maxlen=cfg.buffer_max_samples)
        self._raw_counts: deque[int] = deque(maxlen=cfg.buffer_max_samples)
        self._next_index = 0  # device-timeline index of the next sample
        # Confirmed R-peak indices on the device timeline; bounded by time.
        self._peaks: deque[int] = deque(maxlen=int(PEAK_HISTORY_S * cfg.sample_rate_hz / 0.2))
        self._gaps: deque[tuple[int, int]] = deque(maxlen=MAX_GAPS)  # (start, end) indices of missing data
        self._last_detect_at = -math.inf
        self._last_counters = self.receiver.counters.to_dict()
        self.gap_count = 0
        self.restarts = 0
        self.last_error: str | None = None
        self.model_kind: str | None = None
        self._bundle: dict | None = None
        self._model_cache: tuple[tuple, dict] | None = None

    def set_model(self, kind: str | None, bundle: dict | None) -> None:
        """Select the beat model used for screening (None = rules only)."""
        self.model_kind = kind
        self._bundle = bundle
        self._model_cache = None

    # ----- ingest -----

    def ingest_text(self, text: str | bytes) -> bool:
        before = self._last_counters
        accepted = self.receiver.ingest_text(text)
        now = self.receiver.counters.to_dict()
        if now["session_restarts"] > before["session_restarts"]:
            self._reset_timeline()
        elif now["lost_packets"] > before["lost_packets"]:
            self._record_gap(now["lost_samples_estimated"] - before["lost_samples_estimated"])
        if accepted and self.receiver.last_accepted is not None:
            self._process_packet_samples(self.receiver.last_accepted.samples)
        self.last_error = self.receiver.last_error
        self._last_counters = now
        return accepted

    def _start_new_segment(self) -> None:
        self._filter.reset()
        self._filtered.clear()
        self._raw_counts.clear()
        self._last_detect_at = -math.inf
        self._model_cache = None

    def _record_gap(self, lost_samples: int) -> None:
        start = self._next_index
        self._next_index += lost_samples
        self._gaps.append((start, self._next_index))
        self.gap_count += 1
        self._start_new_segment()

    def _reset_timeline(self) -> None:
        self._start_new_segment()
        self._peaks.clear()
        self._gaps.clear()
        self.restarts += 1

    def _process_packet_samples(self, counts: np.ndarray) -> None:
        mv = self.cfg.adc.counts_to_mv(counts)
        self._filtered.extend(float(v) for v in self._filter.process(mv))
        self._raw_counts.extend(int(c) for c in counts)
        self._next_index += int(counts.size)
        self._maybe_detect()

    def _maybe_detect(self) -> None:
        fs = self.cfg.sample_rate_hz
        end = self._next_index
        if end - self._last_detect_at < DETECT_EVERY_S * fs:
            return
        self._last_detect_at = end
        if len(self._filtered) < int(2.0 * fs):
            return
        window = np.asarray(self._filtered, dtype=float)[-self._window_len :]
        result = detect_rpeaks(window, fs, self.cfg.rpeak, zero_phase=False)
        base = end - window.size
        confirm_before = end - CONFIRM_MARGIN_S * fs
        for idx in result.indices:
            absolute = base + int(idx)
            if absolute >= confirm_before:
                continue  # not confirmed yet; it will be re-detected on a later pass
            if self._peaks and absolute - self._peaks[-1] < self.cfg.rpeak.refractory_s * fs:
                continue  # already confirmed, or too close to the last confirmed beat
            self._peaks.append(absolute)

    # ----- snapshot -----

    def _segment_start(self) -> int:
        return self._next_index - len(self._filtered)

    def snapshot(self, *, recent_waveform_s: float = 10.0) -> dict:
        fs = self.cfg.sample_rate_hz
        now_s = self._next_index / fs
        peaks_abs = np.asarray(self._peaks, dtype=np.int64)
        gaps_s = [(a / fs, b / fs) for a, b in self._gaps]
        series = rr_intervals(peaks_abs / fs, self.cfg.rr, gaps_s)
        hr = heart_rate_summary(series, now_s, self.cfg.screening.analysis_window_s, self.cfg.screening.min_valid_rr)
        hrv = time_domain_hrv(series, self.cfg.hrv)

        segment = np.asarray(self._filtered, dtype=float)
        quality = assess_quality(segment[-self._window_len :], fs, self.cfg.quality, lead_off=any(self.receiver.lead_off))
        screening = screen_heart_rate(series, now_s, self.cfg.screening, signal_status=quality.status)

        n_recent = int(recent_waveform_s * fs)
        filt_recent = segment[-n_recent:]
        raw_recent = np.asarray(self._raw_counts, dtype=np.int64)[-n_recent:]
        raw_mv = self.cfg.adc.counts_to_mv(raw_recent) if raw_recent.size else np.array([])
        recent_start = self._next_index - filt_recent.size
        peak_offsets = [int(p - recent_start) for p in peaks_abs if p >= recent_start]

        return {
            "source": self.source,
            "device_id": self.receiver.device_id,
            "simulated": self.receiver.simulated,
            "fs": fs,
            "connected": self._connected(),
            "seconds_since_last_packet": self.receiver.seconds_since_last_packet(),
            "total_samples": self.receiver.total_samples,
            "timeline_s": now_s,
            "segment_s": segment.size / fs,
            "gaps": self.gap_count,
            "analysis_restarts": self.restarts,
            "lead_off": {"lo_plus": self.receiver.lead_off[0], "lo_minus": self.receiver.lead_off[1]},
            "packet_stats": self.receiver.counters.to_dict(),
            "last_error": self.last_error,
            "heart_rate": {
                "current_bpm": hr.current_bpm,
                "mean_bpm": hr.mean_bpm,
                "min_bpm": hr.min_bpm,
                "max_bpm": hr.max_bpm,
                "valid_rr_count": hr.valid_rr_count,
                "flagged_rr_count": hr.flagged_rr_count,
                "seconds_since_last_beat": hr.seconds_since_last_beat,
                "warnings": list(hr.warnings),
            },
            "rr_recent_s": series.rr_s[series.valid_mask][-10:].tolist(),
            "hrv": {
                "available": hrv.available,
                "mean_nn_ms": hrv.mean_nn_ms,
                "sdnn_ms": hrv.sdnn_ms,
                "rmssd_ms": hrv.rmssd_ms,
                "pnn50_pct": hrv.pnn50_pct,
                "nn_count": hrv.nn_count,
                "warnings": list(hrv.warnings),
            },
            "quality": quality.to_dict(),
            "screening": {
                "model": screening.model,
                "category": screening.category.value,
                "title": screening.title,
                "evidence": list(screening.evidence),
                "interpretation": screening.interpretation,
                "limitations": list(screening.limitations),
                "warnings": list(screening.warnings),
            },
            "model": self._model_analysis(segment, peaks_abs, quality.status),
            "waveform": {
                "time_s": (np.arange(filt_recent.size) / fs).round(4).tolist(),
                "raw_mv": np.round(raw_mv, 4).tolist(),
                "filtered_mv": filt_recent.round(4).tolist(),
                "r_peak_offsets": peak_offsets,
            },
        }

    def _model_analysis(self, segment: np.ndarray, peaks_abs: np.ndarray, quality_status: str) -> dict:
        """Beat-model screening over the analysis window, cached per confirmed beat.

        Beats are classified on the current contiguous segment only, because the
        features need an unbroken signal and eight previous beats.
        """
        from .ml.inference import classify_beats, summarize_window
        from .xai.local import explain_beat
        from .xai.reliability import reliability_warnings

        fs = self.cfg.sample_rate_hz
        if self._bundle is None:
            return {"kind": self.model_kind, "available": False,
                    "warnings": reliability_warnings(quality_status=quality_status, n_beats=0, min_beats=8, bundle=None)}
        key = (self.model_kind, int(peaks_abs[-1]) if peaks_abs.size else -1, quality_status, self._segment_start())
        if self._model_cache is not None and self._model_cache[0] == key:
            return self._model_cache[1]

        n_win = min(segment.size, int(self.cfg.screening.analysis_window_s * fs))
        seg = segment[segment.size - n_win :]
        seg_start = self._next_index - seg.size
        local = np.array([p - seg_start for p in peaks_abs if p >= seg_start], dtype=np.int64)
        preds = classify_beats(self._bundle, seg, local, fs)
        summary = summarize_window(self.model_kind or "model", preds)
        warnings = reliability_warnings(quality_status=quality_status, n_beats=preds.predicted.size, min_beats=8,
                                        bundle=self._bundle, X=preds.X, predicted_classes=preds.predicted,
                                        source=self.source)
        if self.gap_count and seg_start == self._segment_start():
            warnings.append("Data was lost recently; only beats since the last gap were classified.")
        explanation = None
        if preds.predicted.size:
            ectopic = np.flatnonzero(np.isin(preds.predicted, ("S", "V")))
            j = int(ectopic[-1]) if ectopic.size else preds.predicted.size - 1
            explanation = explain_beat(self._bundle, preds.X[j], wave=preds.waves[j])
            r = int(preds.beat_indices[j])
            lo, hi = max(0, r - int(0.4 * fs)), min(seg.size, r + int(0.6 * fs))
            explanation["beat"] = {
                "seconds_ago": (seg.size - r) / fs,
                "reason_selected": "most recent beat classified S or V" if ectopic.size else "most recent classified beat",
                "waveform_mv": seg[lo:hi].round(4).tolist(),
                "r_offset": r - lo,
                "fs": fs,
            }
            if "saliency" in explanation:
                # The saliency covers the model's own input window (0.25 s either side of R).
                explanation["saliency_window"] = {
                    "mv": preds.waves[j].round(4).tolist(),
                    "weight": explanation.pop("saliency"),
                    "r_offset": int(round(0.25 * fs)),
                    "fs": fs,
                }
        result = {
            "kind": self.model_kind,
            "available": True,
            "category": summary.category.value,
            "title": summary.title,
            "evidence": list(summary.evidence),
            "interpretation": summary.interpretation,
            "limitations": list(summary.limitations),
            "class_counts": summary.class_counts,
            "mean_probability": summary.mean_proba,
            "probability_note": "Mean model probability per class over the window: model confidence, not clinical certainty.",
            "warnings": warnings,
            "explanation": explanation,
        }
        self._model_cache = (key, result)
        return result

    def _connected(self) -> bool:
        age = self.receiver.seconds_since_last_packet()
        return age is not None and age < 5.0

    # ----- helpers for recording -----

    def buffered_counts(self) -> np.ndarray:
        """Raw counts of the current contiguous segment (no gaps inside), for saving."""
        return np.asarray(self._raw_counts, dtype=np.int64)
