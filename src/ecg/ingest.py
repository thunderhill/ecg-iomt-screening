"""Packet parsing and stream integrity tracking (network-independent).

``StreamReceiver`` consumes decoded packets and keeps:
* a bounded in-memory buffer of ADC counts for live display,
* counters for malformed, duplicate, lost and restarted packets,
* the lead-off status and the time of the last accepted packet.

Recording to disk is separate (see ``storage.py``): raw samples are saved
whatever the processing outcome.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
ADC_MAX = 4095
RECENT_SEQ_WINDOW = 64  # how far back duplicates are detected
RESTART_SEQ_DROP = 1000  # a seq this far below the last one counts as a restart


class PacketError(ValueError):
    """A packet failed validation. ``reason`` is one of the counter names."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Hello:
    device_id: str
    firmware: str
    simulated: bool
    fs: float
    batch_size: int | None


@dataclass(frozen=True)
class Packet:
    device_id: str
    seq: int
    t_us: int
    fs: float
    samples: np.ndarray  # int64 ADC counts
    lo_plus: bool
    lo_minus: bool


def _require(obj: dict[str, Any], key: str, types: tuple[type, ...]) -> Any:
    if key not in obj:
        raise PacketError("malformed", f"missing field '{key}'")
    value = obj[key]
    if isinstance(value, bool) and bool not in types:
        raise PacketError("malformed", f"field '{key}' must not be a boolean")
    if not isinstance(value, types):
        raise PacketError("malformed", f"field '{key}' has wrong type {type(value).__name__}")
    return value


def _parse_object(text: str | bytes) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PacketError("malformed", f"not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise PacketError("malformed", "message must be a JSON object")
    if obj.get("v") != PROTOCOL_VERSION:
        raise PacketError("malformed", f"unsupported protocol version {obj.get('v')!r}")
    return obj


def parse_hello(text: str | bytes) -> Hello:
    obj = _parse_object(text)
    if obj.get("type") != "hello":
        raise PacketError("malformed", "first message must be of type 'hello'")
    fs = _require(obj, "fs", (int, float))
    if not math.isfinite(fs) or fs <= 0:
        raise PacketError("malformed", "fs must be a positive finite number")
    batch = obj.get("batch_size")
    if batch is not None and (not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0):
        raise PacketError("malformed", "batch_size must be a positive integer")
    return Hello(
        device_id=str(_require(obj, "device_id", (str,))),
        firmware=str(obj.get("firmware", "unknown")),
        simulated=bool(_require(obj, "simulated", (bool,))),
        fs=float(fs),
        batch_size=batch,
    )


def parse_packet(text: str | bytes, expected_fs: float | None = None) -> Packet:
    """Validate and decode one ``samples`` packet. Raises ``PacketError``."""
    obj = _parse_object(text)
    if obj.get("type") != "samples":
        raise PacketError("malformed", f"unexpected message type {obj.get('type')!r}")
    device_id = str(_require(obj, "device_id", (str,)))
    seq = _require(obj, "seq", (int,))
    t_us = _require(obj, "t_us", (int,))
    fs = _require(obj, "fs", (int, float))
    samples_raw = _require(obj, "samples", (list,))
    lo_plus = bool(obj.get("lo_plus", False))
    lo_minus = bool(obj.get("lo_minus", False))
    if seq < 0 or t_us < 0:
        raise PacketError("malformed", "seq and t_us must be non-negative")
    if not math.isfinite(fs) or fs <= 0:
        raise PacketError("malformed", "fs must be a positive finite number")
    if expected_fs is not None and not math.isclose(float(fs), float(expected_fs), rel_tol=1e-9):
        raise PacketError("fs_mismatch", f"packet fs {fs} Hz does not match configured {expected_fs} Hz")
    for v in samples_raw:
        if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v <= ADC_MAX):
            raise PacketError("malformed", f"sample {v!r} is not an integer ADC count in [0, {ADC_MAX}]")
    samples = np.asarray(samples_raw, dtype=np.int64)
    return Packet(device_id, seq, t_us, float(fs), samples, lo_plus, lo_minus)


@dataclass
class IntegrityCounters:
    received: int = 0
    accepted: int = 0
    malformed: int = 0
    fs_mismatch: int = 0
    duplicates: int = 0
    lost_packets: int = 0
    lost_samples_estimated: int = 0
    session_restarts: int = 0
    out_of_order: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class StreamReceiver:
    """Tracks integrity of one device stream and keeps a bounded buffer."""

    fs: float
    max_samples: int
    clock: Any = field(default=time.monotonic, repr=False)
    counters: IntegrityCounters = field(default_factory=IntegrityCounters)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.fs) and self.fs > 0):
            raise ValueError("fs must be a positive finite number")
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._buffer: deque[int] = deque(maxlen=self.max_samples)
        self._recent_seq: deque[int] = deque(maxlen=RECENT_SEQ_WINDOW)
        self._last_seq: int | None = None
        self._last_batch: int = 1
        self._total_samples = 0  # absolute sample index of the next sample
        self._last_packet_at: float | None = None
        self._lead_off = (False, False)
        self.device_id: str | None = None
        self.simulated: bool | None = None
        self.last_error: str | None = None
        self.last_accepted: Packet | None = None  # set only when the last message was accepted

    # ----- lifecycle -----

    def on_hello(self, hello: Hello) -> None:
        if not math.isclose(hello.fs, self.fs, rel_tol=1e-9):
            self.last_error = f"device reports fs {hello.fs} Hz, configured {self.fs} Hz"
            self.counters.fs_mismatch += 1
            raise PacketError("fs_mismatch", self.last_error)
        self.device_id = hello.device_id
        self.simulated = hello.simulated

    # ----- data path -----

    def ingest_text(self, text: str | bytes) -> bool:
        """Parse and ingest one message. Returns True when samples were accepted."""
        self.counters.received += 1
        self.last_accepted = None
        try:
            packet = parse_packet(text, expected_fs=self.fs)
        except PacketError as exc:
            setattr(self.counters, exc.reason, getattr(self.counters, exc.reason) + 1)
            self.last_error = str(exc)
            return False
        return self.ingest_packet(packet)

    def ingest_packet(self, packet: Packet) -> bool:
        self._note_contact()
        self.last_accepted = None
        self._lead_off = (packet.lo_plus, packet.lo_minus)
        if packet.samples.size == 0:
            # Keep-alive: proves the device is alive. Its seq is not tracked, so it
            # never counts as a duplicate or a gap.
            return False
        if packet.seq in self._recent_seq:
            self.counters.duplicates += 1
            return False

        if self._last_seq is not None:
            if packet.seq < self._last_seq:
                if self._last_seq - packet.seq > RESTART_SEQ_DROP:
                    # Device rebooted: a new stream begins.
                    self.counters.session_restarts += 1
                    self._recent_seq.clear()
                    self._last_seq = None
                else:
                    # Late packet from before the last one: keep it only if we
                    # have not already recorded its place as lost.
                    self.counters.out_of_order += 1
                    return False
            elif packet.seq > self._last_seq + 1:
                missing = packet.seq - self._last_seq - 1
                self.counters.lost_packets += missing
                self.counters.lost_samples_estimated += missing * self._last_batch

        self._buffer.extend(int(v) for v in packet.samples)
        self._total_samples += int(packet.samples.size)
        self._last_batch = int(packet.samples.size)
        self._last_seq = packet.seq
        self._recent_seq.append(packet.seq)
        self.counters.accepted += 1
        self.last_accepted = packet
        return True

    def _note_contact(self) -> None:
        self._last_packet_at = self.clock()

    # ----- queries -----

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def lead_off(self) -> tuple[bool, bool]:
        return self._lead_off

    def seconds_since_last_packet(self) -> float | None:
        if self._last_packet_at is None:
            return None
        return max(0.0, self.clock() - self._last_packet_at)

    def latest_counts(self, n: int | None = None) -> np.ndarray:
        """Most recent ``n`` counts (all buffered when ``n`` is None)."""
        data = np.fromiter(self._buffer, dtype=np.int64, count=len(self._buffer))
        if n is None or n >= data.size:
            return data
        return data[-n:]

    @property
    def buffer_start_index(self) -> int:
        """Absolute sample index of the oldest buffered sample."""
        return self._total_samples - len(self._buffer)
