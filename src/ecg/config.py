"""Configuration for acquisition, preprocessing, detection and screening.

All thresholds here are RESEARCH PARAMETERS, not universal clinical
thresholds. They are validated before use so that a bad setting fails loudly
instead of producing silent nonsense.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


MODEL_KINDS = ("rules", "rf", "xgb", "cnn", "hybrid")


@dataclass(frozen=True)
class PreprocessConfig:
    fs: float = 250.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    filter_order: int = 2
    # Baseline wander is estimated with a median filter of this length (s).
    baseline_window_s: float = 0.6
    powerline_hz: float | None = None  # 50 or 60 when enabled
    notch_q: float = 30.0
    # "causal" for live monitoring (adds phase delay), "zero_phase" for offline
    # analysis (no delay, but needs the full recording and is not real-time).
    mode: str = "causal"

    def validate(self) -> None:
        if not (self.fs > 0):
            raise ConfigError("fs must be a positive sampling rate")
        nyquist = self.fs / 2.0
        if not (0 < self.bandpass_low_hz < self.bandpass_high_hz):
            raise ConfigError("need 0 < bandpass_low_hz < bandpass_high_hz")
        if self.bandpass_high_hz >= nyquist:
            raise ConfigError(
                f"bandpass_high_hz={self.bandpass_high_hz} must be below Nyquist ({nyquist} Hz)"
            )
        if not (1 <= self.filter_order <= 6):
            raise ConfigError("filter_order must be between 1 and 6")
        if self.baseline_window_s <= 0:
            raise ConfigError("baseline_window_s must be positive")
        if self.powerline_hz is not None:
            if self.powerline_hz not in (50.0, 60.0):
                raise ConfigError("powerline_hz must be 50 or 60 (or None)")
            if self.powerline_hz >= nyquist:
                raise ConfigError("powerline_hz must be below Nyquist")
        if self.mode not in ("causal", "zero_phase"):
            raise ConfigError("mode must be 'causal' or 'zero_phase'")


@dataclass(frozen=True)
class RPeakConfig:
    # Refractory period: no two R-peaks closer than this (physiological bound).
    refractory_s: float = 0.2
    # Window for integrating the squared derivative (Pan-Tompkins uses ~150 ms).
    integration_window_s: float = 0.15
    # Search-back is triggered when no beat is found for this multiple of the
    # running RR average.
    searchback_factor: float = 1.66
    # Peak refinement searches +/- this window around the candidate on the
    # bandpassed signal.
    refine_window_s: float = 0.1

    def validate(self) -> None:
        if self.refractory_s <= 0 or self.integration_window_s <= 0:
            raise ConfigError("R-peak windows must be positive")
        if self.searchback_factor <= 1.0:
            raise ConfigError("searchback_factor must be > 1")


@dataclass(frozen=True)
class RRPlausibility:
    """Rules that FLAG (never silently drop) implausible RR intervals."""

    min_rr_s: float = 0.3  # ~200 bpm
    max_rr_s: float = 2.0  # ~30 bpm
    # Relative change versus the previous accepted RR beyond which an interval
    # is flagged as a possible ectopic/artifact-related jump (for NN selection).
    max_relative_change: float = 0.2

    def validate(self) -> None:
        if not (0 < self.min_rr_s < self.max_rr_s):
            raise ConfigError("need 0 < min_rr_s < max_rr_s")
        if not (0 < self.max_relative_change <= 1.0):
            raise ConfigError("max_relative_change must be in (0, 1]")


@dataclass(frozen=True)
class HRVConfig:
    min_duration_s: float = 60.0  # below this, time-domain HRV is withheld
    min_nn_intervals: int = 30
    max_flagged_fraction: float = 0.2  # above this, HRV is withheld

    def validate(self) -> None:
        if self.min_duration_s <= 0 or self.min_nn_intervals < 2:
            raise ConfigError("invalid HRV minimums")
        if not (0 < self.max_flagged_fraction <= 1):
            raise ConfigError("max_flagged_fraction must be in (0, 1]")


@dataclass(frozen=True)
class QualityConfig:
    flatline_std_mv: float = 0.005  # std below this across the window => lead-off
    clip_fraction_max: float = 0.02  # fraction of samples at the rail
    noise_ratio_max: float = 0.5  # high-frequency residual vs in-band power
    min_window_s: float = 5.0  # minimum data before quality is judged

    def validate(self) -> None:
        if self.min_window_s <= 0:
            raise ConfigError("min_window_s must be positive")
        if not (0 <= self.clip_fraction_max <= 1):
            raise ConfigError("clip_fraction_max must be in [0, 1]")


@dataclass(frozen=True)
class ScreeningConfig:
    """Heart-rate screening rules. Research parameters, not clinical thresholds."""

    brady_bpm: float = 50.0  # flag if sustained HR below this
    tachy_bpm: float = 100.0  # flag if sustained HR above this
    min_duration_s: float = 10.0  # the threshold must hold for this long
    min_valid_rr: int = 8  # below this, the rule reports insufficient data
    analysis_window_s: float = 60.0

    def validate(self) -> None:
        if not (0 < self.brady_bpm < self.tachy_bpm):
            raise ConfigError("need 0 < brady_bpm < tachy_bpm")
        if self.min_duration_s <= 0 or self.analysis_window_s <= 0:
            raise ConfigError("durations must be positive")
        if self.min_valid_rr < 2:
            raise ConfigError("min_valid_rr must be >= 2")


@dataclass(frozen=True)
class AdcCalibration:
    """Maps ESP32 ADC counts to millivolts.

    These are PLACEHOLDER values for the ESP32 12-bit ADC at 11 dB attenuation
    and the AD8232 mid-rail bias. The ESP32 ADC is non-linear and needs
    per-device calibration and validation before any measurement is trusted.
    """

    adc_counts_max: int = 4095
    adc_vref_mv: float = 3300.0
    mid_rail_mv: float = 1650.0  # AD8232 output bias (approximately mid-supply)
    analog_gain: float = 100.0  # AD8232 gain; verify against your board

    def validate(self) -> None:
        if self.adc_counts_max <= 0 or self.adc_vref_mv <= 0 or self.analog_gain <= 0:
            raise ConfigError("ADC calibration values must be positive")

    def counts_to_mv(self, counts):
        """Vectorised conversion; accepts scalars or arrays of counts."""
        import numpy as np

        c = np.asarray(counts, dtype=float)
        return (c * self.adc_vref_mv / self.adc_counts_max - self.mid_rail_mv) / self.analog_gain

    def mv_to_counts(self, mv):
        import numpy as np

        v = np.asarray(mv, dtype=float) * self.analog_gain + self.mid_rail_mv
        return np.clip(np.round(v * self.adc_counts_max / self.adc_vref_mv), 0, self.adc_counts_max).astype(int)


@dataclass(frozen=True)
class SecurityConfig:
    """Deployment security settings (see docs/security.md). Not editable from the dashboard."""

    tls: bool = False  # set by scripts/serve.py; controls Secure cookies and HSTS
    secrets_dir: str = "secrets"
    session_ttl_s: float = 12 * 3600
    cookie_name: str = "ecg_session"

    def validate(self) -> None:
        if self.session_ttl_s <= 0:
            raise ConfigError("session_ttl_s must be positive")
        if not self.cookie_name.isidentifier():
            raise ConfigError("cookie_name must be a simple identifier")


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration. Use `load_config` / `replace` to derive variants."""

    device_id: str = "ecg-sim-01"
    sample_rate_hz: float = 250.0
    rolling_window_s: float = 30.0  # bounded live window shown in the dashboard
    buffer_max_samples: int = 250 * 600  # hard cap on in-memory samples
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    rpeak: RPeakConfig = field(default_factory=RPeakConfig)
    rr: RRPlausibility = field(default_factory=RRPlausibility)
    hrv: HRVConfig = field(default_factory=HRVConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    adc: AdcCalibration = field(default_factory=AdcCalibration)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    selected_model: str = "rules"  # rules | rf | xgb | cnn | hybrid
    models_dir: str = "models"
    api_host: str = "127.0.0.1"  # network binds require TLS (docs/security.md)
    api_port: int = 8000
    data_dir: str = "data"
    retention_days: int = 30

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ConfigError("sample_rate_hz must be positive")
        if self.preprocess.fs != self.sample_rate_hz:
            # Keep the preprocessing sample rate in lock-step with acquisition.
            object.__setattr__(self, "preprocess", replace(self.preprocess, fs=self.sample_rate_hz))
        if self.rolling_window_s <= 0:
            raise ConfigError("rolling_window_s must be positive")
        if self.buffer_max_samples < int(self.rolling_window_s * self.sample_rate_hz):
            raise ConfigError("buffer_max_samples must hold at least one rolling window")
        for part in (self.preprocess, self.rpeak, self.rr, self.hrv, self.quality, self.screening, self.adc,
                     self.security):
            part.validate()
        if self.retention_days < 1:
            raise ConfigError("retention_days must be >= 1")
        if self.selected_model not in MODEL_KINDS:
            raise ConfigError(f"selected_model must be one of {MODEL_KINDS}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Rebuild a validated config from ``to_dict`` output. Unknown keys are ignored."""
        nested = {"preprocess": PreprocessConfig, "rpeak": RPeakConfig, "rr": RRPlausibility, "hrv": HRVConfig,
                  "quality": QualityConfig, "screening": ScreeningConfig, "adc": AdcCalibration,
                  "security": SecurityConfig}
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            if f.name in nested and isinstance(value, dict):
                sub = nested[f.name]
                names = {sf.name for sf in fields(sub)}
                value = sub(**{k: v for k, v in value.items() if k in names})
            kwargs[f.name] = value
        return cls(**kwargs)

    def with_updates(self, **changes: Any) -> "AppConfig":
        """Return a validated copy. Raises ConfigError on invalid values."""
        return replace(self, **changes)


def default_config() -> AppConfig:
    return AppConfig()
