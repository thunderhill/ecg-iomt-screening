"""Run the SIMULATED ESP32 emulator against a running gateway.

Everything sent is labelled simulated. This is not hardware data. The simulator
needs its own device token (registered with --kind simulator):

    python scripts/security_setup.py add-device ecg-sim-01 --kind simulator
    ECG_DEVICE_TOKEN=<token> python scripts/run_simulator.py \
        --url wss://127.0.0.1:8000/ws/ingest --ca secrets/ca.pem --duration 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg.config import AppConfig  # noqa: E402
from ecg.simulator import Faults, run_device  # noqa: E402


def main() -> None:
    cfg = AppConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=f"ws://{cfg.api_host}:{cfg.api_port}/ws/ingest",
                   help="use wss://... when the gateway runs with --tls")
    p.add_argument("--token", default=os.environ.get("ECG_DEVICE_TOKEN"),
                   help="device token (default: $ECG_DEVICE_TOKEN; prefer the variable, arguments are visible to other users)")
    p.add_argument("--ca", default=None, help="CA certificate to verify a wss:// gateway (e.g. secrets/ca.pem)")
    p.add_argument("--device-id", default="ecg-sim-01")
    p.add_argument("--duration", type=float, default=60.0, help="seconds of simulated ECG")
    p.add_argument("--hr", type=float, default=72.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--fault-drop", type=float, default=0.0)
    p.add_argument("--fault-duplicate", type=float, default=0.0)
    p.add_argument("--fault-reorder", type=float, default=0.0)
    p.add_argument("--no-realtime", action="store_true", help="send as fast as possible")
    args = p.parse_args()

    if not args.token:
        p.error("no device token: set ECG_DEVICE_TOKEN or pass --token")
    faults = Faults(drop=args.fault_drop, duplicate=args.fault_duplicate, reorder=args.fault_reorder)
    print("SIMULATED device: this is software-generated ECG, not patient data.")
    result = asyncio.run(
        run_device(
            args.url,
            device_id=args.device_id,
            fs=cfg.sample_rate_hz,
            adc=cfg.adc,
            duration_s=args.duration,
            batch_size=args.batch_size,
            hr_bpm=args.hr,
            seed=args.seed,
            faults=faults,
            realtime=not args.no_realtime,
            token=args.token,
            ca_file=args.ca,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
