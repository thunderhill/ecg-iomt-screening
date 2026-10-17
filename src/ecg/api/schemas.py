"""Pydantic models for request validation and documented responses."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PreprocessPatch(BaseModel):
    bandpass_low_hz: Optional[float] = Field(None, gt=0)
    bandpass_high_hz: Optional[float] = Field(None, gt=0)
    filter_order: Optional[int] = Field(None, ge=1, le=6)
    powerline_hz: Optional[Literal[0, 50, 60]] = None  # 0 switches the notch off
    baseline_window_s: Optional[float] = Field(None, gt=0)


class ScreeningPatch(BaseModel):
    brady_bpm: Optional[float] = Field(None, gt=0, lt=300)
    tachy_bpm: Optional[float] = Field(None, gt=0, lt=300)
    min_duration_s: Optional[float] = Field(None, gt=0)
    min_valid_rr: Optional[int] = Field(None, ge=2)
    analysis_window_s: Optional[float] = Field(None, gt=0)


class QualityPatch(BaseModel):
    flatline_std_mv: Optional[float] = Field(None, gt=0)
    clip_fraction_max: Optional[float] = Field(None, ge=0, le=1)
    noise_ratio_max: Optional[float] = Field(None, gt=0)
    min_window_s: Optional[float] = Field(None, gt=0)


class ConfigPatch(BaseModel):
    """Partial update. Every value is validated again by ``AppConfig`` before use."""

    sample_rate_hz: Optional[float] = Field(None, gt=0, le=2000)
    rolling_window_s: Optional[float] = Field(None, gt=0, le=600)
    retention_days: Optional[int] = Field(None, ge=1, le=3650)
    selected_model: Optional[Literal["rules", "rf", "xgb", "cnn", "hybrid"]] = None
    preprocess: Optional[PreprocessPatch] = None
    screening: Optional[ScreeningPatch] = None
    quality: Optional[QualityPatch] = None


class SaveRecordingRequest(BaseModel):
    notes: Optional[str] = Field(None, max_length=500)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ErrorResponse(BaseModel):
    detail: str


class LoginRequest(BaseModel):
    token: str = Field(..., min_length=1, max_length=200)
