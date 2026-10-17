"""ECG preprocessing: bandpass, optional mains notch, baseline handling.

Two filtering modes, with different trade-offs:

* ``causal`` (live monitoring): each output sample depends only on current and
  past inputs, so it is usable in real time. It introduces a phase delay
  (group delay of a few tens of ms at 0.5 Hz high-pass edges). Use
  ``StreamingFilter`` to keep state across packets.
* ``zero_phase`` (offline analysis): forward-backward filtering with
  ``sosfiltfilt`` gives no phase distortion, but it needs the whole recording
  and uses future samples, so it must never be used to claim real-time
  performance.

Filtering reduces noise and drift. It does not remove all artifacts and does
not make the signal clinically interpretable on its own.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .config import ConfigError, PreprocessConfig


def validate_signal(x: np.ndarray, min_length: int = 1) -> np.ndarray:
    """Return a 1-D float64 copy of ``x`` or raise ``ValueError``.

    Rejects empty input, wrong dimensionality, non-finite values and inputs
    shorter than ``min_length``. Non-finite values are an error here on purpose:
    the ingest layer is responsible for counting and handling them upstream.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D signal, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("signal is empty")
    if not np.all(np.isfinite(arr)):
        bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
        raise ValueError(f"signal contains {bad} NaN/inf value(s)")
    if arr.size < min_length:
        raise ValueError(f"signal has {arr.size} samples, need at least {min_length}")
    return arr


def bandpass_sos(cfg: PreprocessConfig) -> np.ndarray:
    """Butterworth bandpass as second-order sections (numerically stable)."""
    cfg.validate()
    nyq = cfg.fs / 2.0
    return sps.butter(
        cfg.filter_order,
        [cfg.bandpass_low_hz / nyq, cfg.bandpass_high_hz / nyq],
        btype="bandpass",
        output="sos",
    )


def notch_sos(cfg: PreprocessConfig) -> np.ndarray | None:
    """Mains notch as SOS, or None when power-line filtering is disabled."""
    if cfg.powerline_hz is None:
        return None
    cfg.validate()
    b, a = sps.iirnotch(cfg.powerline_hz, cfg.notch_q, fs=cfg.fs)
    return sps.tf2sos(b, a)


def _min_length_for_filtfilt(sos: np.ndarray) -> int:
    # Matches the padding requirement of sosfiltfilt (default padlen).
    n_sections = sos.shape[0]
    return 3 * (2 * n_sections + 1)


def _sections(cfg: PreprocessConfig) -> np.ndarray:
    parts = [bandpass_sos(cfg)]
    notch = notch_sos(cfg)
    if notch is not None:
        parts.append(notch)
    return np.vstack(parts)


def _baseline_median(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """Centered running median used to estimate slow baseline wander (offline only)."""
    width = max(3, int(round(window_s * fs)))
    if width % 2 == 0:
        width += 1
    return sps.medfilt(x, kernel_size=width) if x.size >= width else np.full_like(x, np.median(x))


def preprocess(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Filter a full recording according to ``cfg.mode``.

    Returns an array of the same length as the input.
    """
    cfg.validate()
    sos = _sections(cfg)
    x = validate_signal(x, min_length=_min_length_for_filtfilt(sos) if cfg.mode == "zero_phase" else 1)
    if cfg.mode == "causal":
        # Single pass; the startup transient is suppressed by initial conditions.
        zi = sps.sosfilt_zi(sos) * x[0]
        y, _ = sps.sosfilt(sos, x, zi=zi)
        return y
    # zero_phase: remove slow baseline first (non-causal, offline only), then
    # filter forward and backward so that the output has no phase shift.
    detrended = x - _baseline_median(x, cfg.fs, cfg.baseline_window_s)
    return sps.sosfiltfilt(sos, detrended)


class StreamingFilter:
    """Stateful causal filter for live packets.

    Keeps filter state across ``process`` calls so that chunk boundaries do not
    introduce discontinuities. Only causal filtering is supported here because
    zero-phase filtering needs future samples.
    """

    def __init__(self, cfg: PreprocessConfig):
        cfg.validate()
        self._cfg = cfg
        self._sos = _sections(cfg)
        self._zi: np.ndarray | None = None

    @property
    def config(self) -> PreprocessConfig:
        return self._cfg

    def reset(self) -> None:
        self._zi = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        x = validate_signal(chunk)
        if self._zi is None:
            self._zi = sps.sosfilt_zi(self._sos) * x[0]
        y, self._zi = sps.sosfilt(self._sos, x, zi=self._zi)
        return y


def ensure_fs_valid(fs: float) -> float:
    """Validate a sampling rate used for ADC/packet timing."""
    if not np.isfinite(fs) or fs <= 0:
        raise ConfigError(f"invalid sampling rate: {fs}")
    return float(fs)
