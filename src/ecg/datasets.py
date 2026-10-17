"""MIT-BIH Arrhythmia Database access: label mapping, resampling and splits.

Facts about the dataset (PhysioNet, ODC-By license):
* 48 half-hour two-lead recordings from 47 subjects, 360 Hz, 11-bit.
* Beat-level annotations (symbols) with sample indices.
* Records 102, 104, 107 and 217 contain paced beats and are excluded from
  AAMI-style evaluation, as is standard practice.

Beat-level labels are NOT recording-level diagnoses. A beat symbol describes
one heartbeat, and a recording can contain many rhythm patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MITDB_RECORDS: tuple[str, ...] = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112",
    "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", "215",
    "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234",
)
EXCLUDED_PACED = frozenset({"102", "104", "107", "217"})
# De Chazal et al. (2004) inter-patient split, DS1 for training and DS2 for testing.
DS1 = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
)
DS2 = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
)
SOURCE_FS = 360.0

# AAMI EC57 beat grouping. Edit this mapping to change the research label set.
DEFAULT_AAMI_MAP: dict[str, str] = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
# Beat annotation symbols used in MIT-BIH. Rhythm and signal-quality symbols
# (e.g. "+", "~", "|") are not beats and are never used as beat labels.
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")


@dataclass(frozen=True)
class Record:
    name: str
    signal_mv: np.ndarray  # lead MLII, resampled, millivolts
    fs: float
    beat_samples: np.ndarray  # annotation sample indices at fs (beat symbols only)
    beat_symbols: np.ndarray  # original symbols
    beat_labels: np.ndarray  # AAMI labels (may be "" for ignored symbols)


def resample_to(x: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    """Polyphase resampling. 360 Hz to 250 Hz is the ratio 25/36."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    ratio = Fraction(dst_fs / src_fs).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator)


def map_labels(symbols: np.ndarray, mapping: dict[str, str] = DEFAULT_AAMI_MAP) -> np.ndarray:
    """AAMI label per beat symbol. Symbols outside the mapping get "" and are excluded."""
    return np.array([mapping.get(str(s), "") for s in symbols], dtype=object)


def load_prepared(path: str | Path) -> Record:
    data = np.load(path, allow_pickle=True)
    return Record(
        name=str(data["name"]),
        signal_mv=data["signal_mv"],
        fs=float(data["fs"]),
        beat_samples=data["beat_samples"],
        beat_symbols=data["beat_symbols"],
        beat_labels=data["beat_labels"],
    )


def save_prepared(path: str | Path, rec: Record) -> None:
    np.savez_compressed(
        path,
        name=rec.name,
        signal_mv=rec.signal_mv.astype(np.float32),
        fs=rec.fs,
        beat_samples=rec.beat_samples.astype(np.int64),
        beat_symbols=rec.beat_symbols.astype(str),
        beat_labels=rec.beat_labels.astype(str),
    )
