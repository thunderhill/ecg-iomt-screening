"""Import a recording for OFFLINE analysis (History page).

    python scripts/import_recording.py --mitdb 208 --minutes 5
    python scripts/import_recording.py --csv my_ecg.csv --fs 250 --column mv

CSV input must contain one numeric column of millivolt values. Imported data is
labelled OFFLINE and stored in millivolts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg.config import AppConfig  # noqa: E402
from ecg.datasets import load_prepared  # noqa: E402
from ecg.storage import Storage  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mitdb", help="prepared MIT-BIH record name, e.g. 208")
    src.add_argument("--csv", type=Path)
    p.add_argument("--fs", type=float, help="sampling rate of the CSV (required with --csv)")
    p.add_argument("--column", default="mv", help="CSV column with millivolt values")
    p.add_argument("--minutes", type=float, default=None, help="keep only the first N minutes")
    args = p.parse_args()

    cfg = AppConfig()
    if args.mitdb:
        rec = load_prepared(ROOT / "data" / "processed" / "mitdb" / f"{args.mitdb}.npz")
        mv, fs, notes = rec.signal_mv, rec.fs, f"MIT-BIH record {args.mitdb} (MLII, resampled to {rec.fs:g} Hz)"
    else:
        if not args.fs or args.fs <= 0:
            sys.exit("--fs is required and must be positive for CSV input")
        df = pd.read_csv(args.csv)
        if args.column not in df.columns:
            sys.exit(f"column {args.column!r} not found; columns: {list(df.columns)}")
        mv = pd.to_numeric(df[args.column], errors="coerce").to_numpy(dtype=float)
        bad = int(np.count_nonzero(~np.isfinite(mv)))
        if bad:
            sys.exit(f"{bad} non-numeric or non-finite values in column {args.column!r}; clean the file first")
        fs, notes = args.fs, f"CSV import: {args.csv.name}"
    if args.minutes:
        mv = mv[: int(args.minutes * 60 * fs)]

    storage = Storage(ROOT / cfg.data_dir)
    info = storage.save_recording(mv, source="OFFLINE", fs=fs, device_id=None, config=cfg.to_dict(),
                                  notes=notes, units="mv")
    print(f"Imported {info.id}: {info.duration_s:.0f} s at {fs:g} Hz, labelled OFFLINE")


if __name__ == "__main__":
    main()
