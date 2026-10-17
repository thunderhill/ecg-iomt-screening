"""Local storage: SQLite for metadata, NumPy files for waveforms.

Raw ADC counts are stored separately from any processed signal, so a
recording is never lost because of a processing choice. Patient identifiers
are not stored. Each recording only has a generated id, a source label and
timing and device metadata.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,          -- LIVE | SIMULATED | OFFLINE
    device_id TEXT,
    fs REAL NOT NULL,
    n_samples INTEGER NOT NULL,
    waveform_path TEXT NOT NULL,
    config_json TEXT NOT NULL,
    model_version TEXT,
    notes TEXT,
    units TEXT NOT NULL DEFAULT 'adc_counts'  -- adc_counts | mv
);
CREATE TABLE IF NOT EXISTS screening_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    model TEXT NOT NULL,
    category TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS config_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    seed INTEGER,
    dataset TEXT
);
"""


@dataclass(frozen=True)
class RecordingInfo:
    id: str
    created_at: float
    source: str
    device_id: str | None
    fs: float
    n_samples: int
    waveform_path: str
    model_version: str | None
    notes: str | None
    units: str = "adc_counts"

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs


class Storage:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.waveforms = self.root / "waveforms"
        self.waveforms.mkdir(parents=True, exist_ok=True)
        # Health data: readable by the owner only (no app-level encryption; see docs/security.md).
        for d in (self.root, self.waveforms):
            d.chmod(0o700)
        self.db_path = self.root / "ecg.sqlite3"
        with self._connect() as con:
            con.executescript(SCHEMA)
            self._migrate(con)

    @staticmethod
    def _migrate(con: sqlite3.Connection) -> None:
        """Bring databases created by older versions up to the current schema."""
        cols = {row[1] for row in con.execute("PRAGMA table_info(recordings)")}
        if "units" not in cols:
            con.execute("ALTER TABLE recordings ADD COLUMN units TEXT NOT NULL DEFAULT 'adc_counts'")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    # ----- recordings -----

    def save_recording(
        self,
        counts: np.ndarray,
        *,
        source: str,
        fs: float,
        device_id: str | None,
        config: dict[str, Any],
        model_version: str | None = None,
        notes: str | None = None,
        units: str = "adc_counts",
    ) -> RecordingInfo:
        """Save a waveform. ``units`` is "adc_counts" (device data) or "mv" (imported data)."""
        if source not in ("LIVE", "SIMULATED", "OFFLINE"):
            raise ValueError(f"unknown source label {source!r}")
        if units not in ("adc_counts", "mv"):
            raise ValueError(f"unknown units {units!r}")
        arr = np.asarray(counts, dtype=np.int64 if units == "adc_counts" else np.float32)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("recording must be a non-empty 1-D array")
        if units == "mv" and not np.all(np.isfinite(arr)):
            raise ValueError("recording contains NaN/inf values")
        rec_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        path = self.waveforms / f"{rec_id}.npy"
        np.save(path, arr)  # raw samples, saved regardless of processing outcome
        info = RecordingInfo(
            id=rec_id,
            created_at=time.time(),
            source=source,
            device_id=device_id,
            fs=float(fs),
            n_samples=int(arr.size),
            waveform_path=str(path),
            model_version=model_version,
            notes=notes,
            units=units,
        )
        with self._connect() as con:
            con.execute(
                "INSERT INTO recordings VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    info.id,
                    info.created_at,
                    info.source,
                    info.device_id,
                    info.fs,
                    info.n_samples,
                    info.waveform_path,
                    json.dumps(config, sort_keys=True, default=str),
                    info.model_version,
                    info.notes,
                    info.units,
                ),
            )
        return info

    def list_recordings(self, limit: int = 100) -> list[RecordingInfo]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, created_at, source, device_id, fs, n_samples, waveform_path, model_version, notes, units "
                "FROM recordings ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [RecordingInfo(**dict(r)) for r in rows]

    def get_recording(self, rec_id: str) -> RecordingInfo | None:
        with self._connect() as con:
            r = con.execute(
                "SELECT id, created_at, source, device_id, fs, n_samples, waveform_path, model_version, notes, units "
                "FROM recordings WHERE id = ?",
                (rec_id,),
            ).fetchone()
        return RecordingInfo(**dict(r)) if r else None

    def load_raw(self, rec_id: str) -> tuple[RecordingInfo, np.ndarray]:
        """The stored samples, in the recording's own units (see ``RecordingInfo.units``)."""
        info = self.get_recording(rec_id)
        if info is None:
            raise KeyError(rec_id)
        return info, np.load(info.waveform_path)

    def load_mv(self, rec_id: str, adc) -> tuple[RecordingInfo, np.ndarray]:
        """Samples in millivolts. Device recordings are converted with ``adc`` calibration."""
        info, raw = self.load_raw(rec_id)
        mv = adc.counts_to_mv(raw) if info.units == "adc_counts" else raw.astype(float)
        return info, mv

    def delete_older_than(self, days: int) -> int:
        """Apply retention: remove recordings and their waveform files older than ``days``."""
        if days < 1:
            raise ValueError("retention must be at least 1 day")
        cutoff = time.time() - days * 86400
        with self._connect() as con:
            rows = con.execute("SELECT id, waveform_path FROM recordings WHERE created_at < ?", (cutoff,)).fetchall()
            for r in rows:
                Path(r["waveform_path"]).unlink(missing_ok=True)
            con.execute("DELETE FROM recordings WHERE created_at < ?", (cutoff,))
        return len(rows)

    # ----- model registry -----

    def list_models(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, created_at, kind, path, size_bytes, metrics_json, seed, dataset "
                "FROM models ORDER BY created_at DESC"
            ).fetchall()
        return [{**{k: r[k] for k in ("id", "created_at", "kind", "size_bytes", "seed", "dataset")},
                 "metrics": json.loads(r["metrics_json"])} for r in rows]

    def register_model(self, *, model_id: str, kind: str, path: str, size_bytes: int,
                       metrics: dict, seed: int | None, dataset: str | None) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO models (id, created_at, kind, path, size_bytes, metrics_json, seed, dataset) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (model_id, time.time(), kind, path, size_bytes, json.dumps(metrics, default=str), seed, dataset),
            )

    # ----- events and configuration -----

    def log_screening(self, recording_id: str | None, source: str, model: str, category: str, payload: dict) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO screening_events (recording_id, created_at, source, model, category, payload_json) "
                "VALUES (?,?,?,?,?,?)",
                (recording_id, time.time(), source, model, category, json.dumps(payload, default=str)),
            )

    def recent_screening(self, limit: int = 50) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT created_at, source, model, category, payload_json FROM screening_events "
                "ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload_json"])} for r in rows]

    def latest_config(self) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute("SELECT config_json FROM config_history ORDER BY id DESC LIMIT 1").fetchone()
        return json.loads(row["config_json"]) if row else None

    def log_config(self, config: dict[str, Any]) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO config_history (created_at, config_json) VALUES (?,?)",
                (time.time(), json.dumps(config, sort_keys=True, default=str)),
            )


def export_csv(path: str | Path, header: Iterable[str], rows: Iterable[Iterable[Any]]) -> Path:
    """Write rows to CSV. Used for waveform and derived-metric exports."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
    return out
