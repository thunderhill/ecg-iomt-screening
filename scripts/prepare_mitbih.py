"""Download (optional) and prepare the MIT-BIH Arrhythmia Database.

Nothing is downloaded unless you pass ``--download``. The download is about
100 MB from PhysioNet (ODC-By license). The script then:
  1. reads lead MLII (falls back to channel 0 when that name is absent),
  2. resamples 360 Hz -> 250 Hz (the hardware target rate),
  3. maps beat symbols to AAMI classes and keeps the beat sample indices,
  4. writes one compressed .npz per record to data/processed/mitdb/.

Usage:
    python scripts/prepare_mitbih.py --download          # first time, ~100 MB
    python scripts/prepare_mitbih.py --download --mirror aws   # faster official mirror
    python scripts/prepare_mitbih.py                     # prepare from data/raw
    python scripts/prepare_mitbih.py --records 100 101   # subset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg.datasets import (  # noqa: E402
    BEAT_SYMBOLS,
    MITDB_RECORDS,
    SOURCE_FS,
    Record,
    map_labels,
    resample_to,
    save_prepared,
)

RAW_DIR = ROOT / "data" / "raw" / "mitdb"
OUT_DIR = ROOT / "data" / "processed" / "mitdb"
TARGET_FS = 250.0


MIRRORS = {
    "physionet": "https://physionet.org/files/mitdb/1.0.0",
    # PhysioNet's official open-data copy on AWS; same files, often much faster.
    "aws": "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0",
}


def _fetch(url: str, dest: Path, attempts: int = 5) -> str:
    """Download ``url`` to ``dest`` with retries. Returns "skipped" or "downloaded".

    A file whose size already equals the server's Content-Length is kept. Data is
    written to a ``.part`` file and renamed when complete, so an interrupted
    download never leaves a truncated file under the final name.
    """
    import time
    import urllib.error
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                expected = resp.headers.get("Content-Length")
                if expected is not None and dest.exists() and dest.stat().st_size == int(expected):
                    return "skipped"
                with tmp.open("wb") as fh:
                    while chunk := resp.read(1 << 16):
                        fh.write(chunk)
            if expected is not None and tmp.stat().st_size != int(expected):
                raise OSError(f"incomplete download: {tmp.stat().st_size} of {expected} bytes")
            tmp.replace(dest)
            return "downloaded"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if attempt == attempts:
                raise
            wait = 2 ** attempt
            print(f"    retry {attempt}/{attempts - 1} for {dest.name} in {wait} s ({exc})", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def download(raw_dir: Path, records: list[str], mirror: str) -> None:
    """Fetch .hea/.dat/.atr for each record, resuming where a previous run stopped."""
    base = MIRRORS[mirror]
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(records)} MIT-BIH records from {base} to {raw_dir} (about 100 MB) ...", flush=True)
    for i, name in enumerate(records, 1):
        results = [_fetch(f"{base}/{name}.{ext}", raw_dir / f"{name}.{ext}") for ext in ("hea", "dat", "atr")]
        print(f"  [{i}/{len(records)}] {name} ({', '.join(results)})", flush=True)
    print("Download complete.", flush=True)


def prepare_record(name: str, raw_dir: Path) -> Record:
    import wfdb

    base = str(raw_dir / name)
    header = wfdb.rdheader(base)
    channel = header.sig_name.index("MLII") if "MLII" in header.sig_name else 0
    rec = wfdb.rdrecord(base, channels=[channel])
    signal = rec.p_signal[:, 0].astype(np.float64)  # physical units (mV)
    if rec.units and rec.units[0] not in ("mV",):
        raise ValueError(f"record {name}: unexpected units {rec.units[0]!r}")
    ann = wfdb.rdann(base, "atr")

    keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol], dtype=bool)
    symbols = np.array(ann.symbol)[keep]
    samples_360 = np.asarray(ann.sample)[keep]

    resampled = resample_to(signal, SOURCE_FS, TARGET_FS)
    samples_250 = np.round(samples_360 * TARGET_FS / SOURCE_FS).astype(np.int64)
    samples_250 = np.clip(samples_250, 0, resampled.size - 1)
    return Record(
        name=name,
        signal_mv=resampled,
        fs=TARGET_FS,
        beat_samples=samples_250,
        beat_symbols=symbols,
        beat_labels=map_labels(symbols),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true", help="download from PhysioNet (explicit consent)")
    p.add_argument("--mirror", choices=sorted(MIRRORS), default="physionet", help="download source")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--records", nargs="*", default=list(MITDB_RECORDS))
    args = p.parse_args()

    if args.download:
        download(args.raw_dir, args.records, args.mirror)
    if not (args.raw_dir / "100.dat").exists() and not (args.raw_dir / "100.hea").exists():
        print(f"No MIT-BIH files found in {args.raw_dir}. Re-run with --download.", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in args.records:
        out = args.out_dir / f"{name}.npz"
        rec = prepare_record(name, args.raw_dir)
        save_prepared(out, rec)
        labels, counts = np.unique(rec.beat_labels, return_counts=True)
        summary.append({"record": name, "fs": rec.fs, "seconds": round(rec.signal_mv.size / rec.fs, 1),
                        "beats": int(rec.beat_samples.size), "labels": dict(zip(labels.tolist(), counts.tolist()))})
        print(f"{name}: {summary[-1]['seconds']} s, {summary[-1]['beats']} beats", flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(summary)} records to {args.out_dir}")


if __name__ == "__main__":
    main()
