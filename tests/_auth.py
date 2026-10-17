"""Test helper: an app client that is signed in and has registered test devices."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ecg.api.app import create_app
from ecg.config import AppConfig

TEST_DEVICES = {"ecg-sim-01": "simulator", "d": "simulator", "hw-01": "hardware"}


def signed_in(cfg: AppConfig, admin_token: str | None = None) -> TestClient:
    """Create the app, ensure an admin token and test devices exist, and sign in.

    The returned client has ``admin_token`` and ``device_tokens`` attributes and a
    ``device_headers(device_id)`` helper for WebSocket connections.
    """
    app = create_app(cfg)
    store = app.state.ecg.security
    if admin_token is None:
        admin_token = store.rotate_admin_token()
    registered = {d.device_id for d in store.list_devices() if not d.revoked}
    tokens = {dev: store.add_device(dev, kind) for dev, kind in TEST_DEVICES.items() if dev not in registered}
    client = TestClient(app)
    assert client.post("/auth/login", json={"token": admin_token}).status_code == 200
    client.admin_token = admin_token
    client.device_tokens = tokens
    client.device_headers = lambda dev: {"Authorization": f"Bearer {tokens[dev]}"}
    return client
