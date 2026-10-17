"""Signal-quality assessment over a window.

The checks are heuristics. They detect obvious lead-off, flatlines, rail
clipping and high-frequency noise. A "good" status does not mean the signal is
clinically valid, and the reasons list is returned so the dashboard can show
exactly why a window was judged poor.

Signal units are assumed to be millivolts after acquisition calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import signal as sps

from .config import QualityConfig


@dataclass(frozen=True)
class QualityReport:
    status: str  # "good" | "fair" | "poor" | "insufficient_data"
    reasons: tuple[str, ...] = field(default_factory=tuple)
    std_mv: float = 0.0
    clip_fraction: float = 0.0
    noise_ratio: float = 0.0
    lead_off: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


def _noise_ratio(x: np.ndarray, fs: float, low: float, high: float) -> float:
    """Ratio of power above the passband to power inside it (Welch estimate)."""
    nperseg = min(x.size, max(64, int(fs * 2)))
    freqs, psd = sps.welch(x, fs=fs, nperseg=nperseg)
    in_band = (freqs >= low) & (freqs <= high)
    out_band = freqs > high
    p_in = float(np.trapezoid(psd[in_band], freqs[in_band])) if np.any(in_band) else 0.0
    p_out = float(np.trapezoid(psd[out_band], freqs[out_band])) if np.any(out_band) else 0.0
    if p_in <= 0:
        return float("inf")
    return p_out / p_in


def assess_quality(
    x: np.ndarray,
    fs: float,
    cfg: QualityConfig,
    *,
    band_low_hz: float = 0.5,
    band_high_hz: float = 40.0,
    lead_off: bool = False,
) -> QualityReport:
    """Assess a window of (preferably already-filtered or raw) ECG samples.

    ``lead_off`` carries the AD8232 LO+/LO- status when hardware reports it.
    """
    cfg.validate()
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    duration = arr.size / fs if fs > 0 else 0.0
    if arr.size < 2 or duration < cfg.min_window_s:
        return QualityReport(
            status="insufficient_data",
            reasons=(f"only {duration:.1f} s of valid data; need {cfg.min_window_s:.1f} s",),
            lead_off=lead_off,
        )

    reasons: list[str] = []
    severity = 0  # 0 good, 1 fair, 2 poor

    if lead_off:
        reasons.append("electrode lead-off reported (LO+ or LO- asserted)")
        severity = 2

    std = float(np.std(arr))
    if std < cfg.flatline_std_mv:
        reasons.append(f"near-flat signal (std {std:.4f} mV): possible lead-off")
        severity = 2

    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi > lo:
        at_rail = np.count_nonzero((arr >= hi) | (arr <= lo)) / arr.size
    else:
        at_rail = 1.0
    if at_rail > cfg.clip_fraction_max:
        reasons.append(f"{at_rail:.1%} of samples at extreme values (possible clipping)")
        severity = max(severity, 2)

    noise = _noise_ratio(arr, fs, band_low_hz, band_high_hz)
    if noise > cfg.noise_ratio_max:
        reasons.append(f"high-frequency noise ratio {noise:.2f} exceeds {cfg.noise_ratio_max:.2f}")
        severity = max(severity, 1)

    status = {0: "good", 1: "fair", 2: "poor"}[severity]
    if not reasons:
        reasons.append("no quality problems detected by heuristic checks")
    return QualityReport(
        status=status,
        reasons=tuple(reasons),
        std_mv=std,
        clip_fraction=float(at_rail),
        noise_ratio=float(noise) if np.isfinite(noise) else float("inf"),
        lead_off=lead_off,
    )
