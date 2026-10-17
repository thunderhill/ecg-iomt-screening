"""SIMULATED ESP32 + AD8232 device (software only).

This emulates the firmware's network behaviour so the laptop side can be
developed and tested without hardware. Everything it sends is labelled
``"simulated": true`` in the hello message, so the server reports the stream
as SIMULATED. Simulated data is never presented as LIVE, and metrics from it
are never evaluation results.

Packet building and fault injection are pure functions (easy to test). The
``run_device`` coroutine is the thin WebSocket wrapper around them.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import AdcCalibration
from .ingest import PROTOCOL_VERSION
from .synthetic import generate_ecg

FIRMWARE_LABEL = "sim-0.1.0"


@dataclass(frozen=True)
class Faults:
    """Probabilities for injected network faults (applied per packet)."""

    drop: float = 0.0  # packet never arrives
    duplicate: float = 0.0  # packet arrives twice
    reorder: float = 0.0  # packet arrives after the next one

    def validate(self) -> None:
        for name in ("drop", "duplicate", "reorder"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"fault probability {name} must be in [0, 1]")


def hello_message(device_id: str, fs: float, batch_size: int) -> str:
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "type": "hello",
            "device_id": device_id,
            "firmware": FIRMWARE_LABEL,
            "simulated": True,
            "fs": fs,
            "batch_size": batch_size,
        }
    )


def build_packets(
    counts: np.ndarray,
    *,
    device_id: str,
    fs: float,
    batch_size: int,
    start_seq: int = 0,
    t0_us: int = 0,
) -> Iterator[dict]:
    """Yield ``samples`` messages (as dicts) covering ``counts`` in order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    arr = np.asarray(counts, dtype=np.int64)
    for i, start in enumerate(range(0, arr.size, batch_size)):
        chunk = arr[start : start + batch_size]
        yield {
            "v": PROTOCOL_VERSION,
            "type": "samples",
            "device_id": device_id,
            "seq": start_seq + i,
            "t_us": t0_us + int(round(start * 1e6 / fs)),
            "fs": fs,
            "lo_plus": False,
            "lo_minus": False,
            "samples": chunk.tolist(),
        }


def apply_faults(messages: list[dict], faults: Faults, rng: np.random.Generator) -> list[dict]:
    """Return the messages as the network would deliver them, with faults applied."""
    faults.validate()
    out: list[dict] = []
    held: dict | None = None
    for msg in messages:
        if rng.random() < faults.drop:
            continue
        if rng.random() < faults.reorder and held is None:
            held = msg  # delivered after the next message
            continue
        out.append(msg)
        if rng.random() < faults.duplicate:
            out.append(msg)
        if held is not None:
            out.append(held)
            held = None
    if held is not None:
        out.append(held)
    return out


def simulated_counts(duration_s: float, fs: float, adc: AdcCalibration, seed: int, hr_bpm: float = 72.0) -> np.ndarray:
    """ADC counts for a synthetic recording (SIMULATED data, not patient data)."""
    rec = generate_ecg(duration_s, fs=fs, hr_bpm=hr_bpm, seed=seed)
    return adc.mv_to_counts(rec.signal)


async def run_device(
    url: str,
    *,
    device_id: str,
    fs: float,
    adc: AdcCalibration,
    duration_s: float,
    batch_size: int = 10,
    hr_bpm: float = 72.0,
    seed: int = 0,
    faults: Faults = Faults(),
    realtime: bool = True,
    connect_attempts: int = 6,
    token: str | None = None,
    ca_file: str | None = None,
) -> dict:
    """Stream simulated packets to ``url`` at the device's real-time pace.

    Like the firmware, the simulator authenticates with its device ``token`` and,
    for ``wss://`` URLs, verifies the gateway against the project CA (``ca_file``).
    """
    import ssl

    import websockets  # imported lazily: only needed for live simulation

    if not token:
        raise ValueError("a device token is required (scripts/security_setup.py add-device ... --kind simulator)")
    ssl_ctx = None
    if url.startswith("wss://"):
        ssl_ctx = ssl.create_default_context(cafile=ca_file) if ca_file else ssl.create_default_context()
    headers = {"Authorization": f"Bearer {token}"}

    counts = simulated_counts(duration_s, fs, adc, seed=seed, hr_bpm=hr_bpm)
    rng = np.random.default_rng(seed + 1)
    messages = list(build_packets(counts, device_id=device_id, fs=fs, batch_size=batch_size))
    delivered = apply_faults(messages, faults, rng)
    sent = 0
    period = batch_size / fs
    # Like the firmware, retry the connection with exponential backoff.
    delay = 1.0
    for attempt in range(1, connect_attempts + 1):
        try:
            ws = await websockets.connect(url, max_size=2**20, additional_headers=headers, ssl=ssl_ctx)
            break
        except OSError:
            if attempt == connect_attempts:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
    async with ws:
        # The gateway may refuse an unknown token before reading the hello; its
        # reason message is still delivered, so read it even if sending failed.
        try:
            await ws.send(hello_message(device_id, fs, batch_size))
        except websockets.ConnectionClosed:
            pass
        try:
            reply = json.loads(await ws.recv())
        except websockets.ConnectionClosed as exc:
            raise ConnectionRefusedError(f"gateway closed the connection ({exc})") from exc
        if reply.get("type") == "error":
            raise ConnectionRefusedError(f"gateway refused the device: {reply.get('reason')}: {reply.get('detail')}")
        started = time.monotonic()
        for i, msg in enumerate(delivered):
            await ws.send(json.dumps(msg))
            sent += 1
            if realtime:
                target = started + (i + 1) * period
                await asyncio.sleep(max(0.0, target - time.monotonic()))
    return {"messages_planned": len(messages), "messages_sent": sent, "faults": faults.__dict__}
