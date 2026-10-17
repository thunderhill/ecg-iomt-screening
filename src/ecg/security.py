"""Authentication for devices and dashboard users (see docs/security.md).

* Tokens are 256-bit random values with a readable prefix (``ecgdev_`` for
  devices, ``ecgadm_`` for the dashboard). Only SHA-256 hashes are stored, and
  comparisons are constant-time.
* The device registry records each device's kind (``hardware`` or ``simulator``).
  The gateway labels streams from this registry, never from the device's claim.
* Dashboard sessions are random ids stored hashed in SQLite with an expiry, so a
  separate process (``security_setup.py rotate-admin``) can end them.
* Failed sign-ins are throttled per client address.

The private CA and TLS keys are handled by ``scripts/security_setup.py``. This
module never sees them.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

DEVICE_PREFIX = "ecgdev_"
ADMIN_PREFIX = "ecgadm_"
DEVICE_KINDS = ("hardware", "simulator")
SESSION_TTL_S = 12 * 3600
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_S = 15 * 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,             -- hardware | simulator
    token_hash TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    last_seen REAL,
    revoked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS admin_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_hash TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    session_hash TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
"""


class SecurityError(RuntimeError):
    pass


def new_token(prefix: str) -> str:
    return prefix + secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Device:
    device_id: str
    kind: str
    created_at: float
    last_seen: float | None
    revoked: bool

    @property
    def source_label(self) -> str:
        return "SIMULATED" if self.kind == "simulator" else "LIVE"


class SecurityStore:
    """Device registry, admin tokens and sessions in the gateway's SQLite database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    # ----- devices -----

    def add_device(self, device_id: str, kind: str) -> str:
        """Register a device and return its token (shown once, stored hashed)."""
        if kind not in DEVICE_KINDS:
            raise SecurityError(f"kind must be one of {DEVICE_KINDS}")
        if not device_id or len(device_id) > 64 or not all(c.isalnum() or c in "-_." for c in device_id):
            raise SecurityError("device id must be 1-64 characters: letters, digits, '-', '_' or '.'")
        token = new_token(DEVICE_PREFIX)
        with self._connect() as con:
            existing = con.execute("SELECT revoked FROM devices WHERE device_id = ?", (device_id,)).fetchone()
            if existing and not existing["revoked"]:
                raise SecurityError(f"device {device_id!r} already registered; revoke it first to issue a new token")
            con.execute(
                "INSERT OR REPLACE INTO devices (device_id, kind, token_hash, created_at, last_seen, revoked) "
                "VALUES (?, ?, ?, ?, NULL, 0)",
                (device_id, kind, hash_token(token), time.time()),
            )
        return token

    def revoke_device(self, device_id: str) -> bool:
        with self._connect() as con:
            cur = con.execute("UPDATE devices SET revoked = 1 WHERE device_id = ? AND revoked = 0", (device_id,))
        return cur.rowcount > 0

    def verify_device(self, token: str | None) -> Device | None:
        """The active device a token belongs to, or None."""
        if not token or not token.startswith(DEVICE_PREFIX):
            return None
        digest = hash_token(token)
        with self._connect() as con:
            rows = con.execute("SELECT * FROM devices WHERE revoked = 0").fetchall()
        # Compare every active hash in constant time rather than looking one up by value.
        match = None
        for row in rows:
            if hmac.compare_digest(row["token_hash"], digest):
                match = row
        return self._device(match) if match is not None else None

    def touch_device(self, device_id: str) -> None:
        with self._connect() as con:
            con.execute("UPDATE devices SET last_seen = ? WHERE device_id = ?", (time.time(), device_id))

    def list_devices(self) -> list[Device]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM devices ORDER BY created_at").fetchall()
        return [self._device(r) for r in rows]

    def active_device_count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM devices WHERE revoked = 0").fetchone()[0])

    @staticmethod
    def _device(row: sqlite3.Row) -> Device:
        return Device(row["device_id"], row["kind"], row["created_at"], row["last_seen"], bool(row["revoked"]))

    # ----- dashboard access token -----

    def rotate_admin_token(self) -> str:
        """Issue a new dashboard token, revoke older ones and end every session."""
        token = new_token(ADMIN_PREFIX)
        with self._connect() as con:
            con.execute("UPDATE admin_tokens SET revoked = 1")
            con.execute("INSERT INTO admin_tokens (token_hash, created_at) VALUES (?, ?)", (hash_token(token), time.time()))
            con.execute("DELETE FROM sessions")
        return token

    def has_admin_token(self) -> bool:
        with self._connect() as con:
            return con.execute("SELECT 1 FROM admin_tokens WHERE revoked = 0").fetchone() is not None

    def verify_admin(self, token: str | None) -> bool:
        if not token or not token.startswith(ADMIN_PREFIX):
            return False
        digest = hash_token(token)
        with self._connect() as con:
            rows = con.execute("SELECT token_hash FROM admin_tokens WHERE revoked = 0").fetchall()
        ok = False
        for row in rows:
            ok |= hmac.compare_digest(row["token_hash"], digest)
        return ok

    # ----- sessions -----

    def create_session(self, ttl_s: float = SESSION_TTL_S) -> str:
        session_id = secrets.token_urlsafe(32)
        now = time.time()
        with self._connect() as con:
            con.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            con.execute("INSERT INTO sessions VALUES (?, ?, ?)", (hash_token(session_id), now, now + ttl_s))
        return session_id

    def session_valid(self, session_id: str | None) -> bool:
        if not session_id:
            return False
        with self._connect() as con:
            row = con.execute("SELECT expires_at FROM sessions WHERE session_hash = ?", (hash_token(session_id),)).fetchone()
        return row is not None and row["expires_at"] > time.time()

    def end_session(self, session_id: str | None) -> None:
        if session_id:
            with self._connect() as con:
                con.execute("DELETE FROM sessions WHERE session_hash = ?", (hash_token(session_id),))


class LoginThrottle:
    """Refuse sign-in from an address after repeated failures (in memory, per process)."""

    def __init__(self, max_failures: int = LOGIN_MAX_FAILURES, window_s: float = LOGIN_WINDOW_S, clock=time.monotonic):
        self.max_failures = max_failures
        self.window_s = window_s
        self._clock = clock
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, key: str) -> deque[float]:
        q = self._failures[key]
        cutoff = self._clock() - self.window_s
        while q and q[0] < cutoff:
            q.popleft()
        return q

    def blocked(self, key: str) -> bool:
        return len(self._prune(key)) >= self.max_failures

    def record_failure(self, key: str) -> None:
        self._prune(key).append(self._clock())

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)


def check_binding(host: str, *, tls: bool, store: SecurityStore) -> None:
    """Enforce the binding rule from docs/security.md before the server starts."""
    if not store.has_admin_token():
        raise SecurityError("no dashboard access token; run: scripts/security_setup.py init")
    if is_loopback(host):
        return
    if not tls:
        raise SecurityError(f"refusing to listen on {host} without TLS; add --tls (see docs/security.md)")
    if store.active_device_count() == 0:
        raise SecurityError("refusing to listen on the network with no registered devices; run security_setup.py add-device")
