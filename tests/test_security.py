import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from _auth import signed_in
from ecg.api.app import create_app
from ecg.config import AppConfig, SecurityConfig
from ecg.security import LoginThrottle, SecurityError, SecurityStore, check_binding, is_loopback
from ecg.simulator import build_packets, hello_message


@pytest.fixture()
def cfg(tmp_path):
    return AppConfig(data_dir=str(tmp_path), models_dir=str(tmp_path / "none"))


# ---------- HTTP access control ----------

@pytest.mark.parametrize("path", ["/status", "/recordings", "/configuration", "/devices", "/openapi.json", "/docs"])
def test_routes_require_sign_in(cfg, path):
    client = TestClient(create_app(cfg))
    assert client.get(path).status_code == 401


def test_health_is_public(cfg):
    assert TestClient(create_app(cfg)).get("/health").status_code == 200


def test_sign_in_with_wrong_token_fails(cfg):
    app = create_app(cfg)
    app.state.ecg.security.rotate_admin_token()
    client = TestClient(app)
    assert client.post("/auth/login", json={"token": "ecgadm_wrong"}).status_code == 401
    assert client.get("/status").status_code == 401


def test_device_token_cannot_sign_in_to_dashboard(cfg):
    client = signed_in(cfg)
    other = TestClient(client.app)
    assert other.post("/auth/login", json={"token": client.device_tokens["d"]}).status_code == 401


def test_sign_in_cookie_flags(cfg):
    app = create_app(cfg)
    token = app.state.ecg.security.rotate_admin_token()
    r = TestClient(app).post("/auth/login", json={"token": token})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "secure" not in cookie


def test_cookie_is_secure_under_tls(tmp_path):
    cfg = AppConfig(data_dir=str(tmp_path), security=SecurityConfig(tls=True))
    app = create_app(cfg)
    token = app.state.ecg.security.rotate_admin_token()
    r = TestClient(app, base_url="https://testserver").post("/auth/login", json={"token": token})
    assert "secure" in r.headers["set-cookie"].lower()
    assert r.headers["strict-transport-security"].startswith("max-age=")


def test_failed_sign_ins_are_throttled(cfg):
    app = create_app(cfg)
    token = app.state.ecg.security.rotate_admin_token()
    client = TestClient(app)
    for _ in range(5):
        assert client.post("/auth/login", json={"token": "ecgadm_wrong"}).status_code == 401
    # Even the correct token is refused while the address is blocked.
    assert client.post("/auth/login", json={"token": token}).status_code == 429


def test_throttle_window_expires():
    now = [0.0]
    t = LoginThrottle(max_failures=2, window_s=10, clock=lambda: now[0])
    t.record_failure("a")
    t.record_failure("a")
    assert t.blocked("a")
    now[0] = 11.0
    assert not t.blocked("a")


def test_logout_ends_session(cfg):
    client = signed_in(cfg)
    assert client.get("/status").status_code == 200
    client.post("/auth/logout")
    assert client.get("/status").status_code == 401


def test_rotating_admin_token_ends_sessions(cfg):
    client = signed_in(cfg)
    client.app.state.ecg.security.rotate_admin_token()
    assert client.get("/status").status_code == 401
    assert client.post("/auth/login", json={"token": client.admin_token}).status_code == 401


def test_cross_site_writes_are_refused(cfg):
    client = signed_in(cfg)
    r = client.put("/configuration", json={"retention_days": 7}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    same = client.put("/configuration", json={"retention_days": 7}, headers={"Origin": "http://testserver"})
    assert same.status_code == 200


def test_security_headers_on_every_response(cfg):
    r = TestClient(create_app(cfg)).get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert "strict-transport-security" not in r.headers  # no TLS in this app


def test_device_list_never_exposes_tokens(cfg):
    client = signed_in(cfg)
    body = client.get("/devices").text
    assert all(tok not in body for tok in client.device_tokens.values())
    assert "token" not in json.loads(body)["items"][0]


# ---------- device authentication on the ingest WebSocket ----------

def _reject_reason(client, headers, hello):
    with client.websocket_connect("/ws/ingest", headers=headers) as ws:
        ws.send_text(hello)
        reply = json.loads(ws.receive_text())
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1008
    return reply["reason"]


def test_ingest_without_token_is_refused(cfg):
    client = signed_in(cfg)
    assert _reject_reason(client, {}, hello_message("d", 250.0, 10)) == "unauthorized"


def test_ingest_with_unknown_or_dashboard_token_is_refused(cfg):
    client = signed_in(cfg)
    for token in ("ecgdev_unknown", client.admin_token):
        headers = {"Authorization": f"Bearer {token}"}
        assert _reject_reason(client, headers, hello_message("d", 250.0, 10)) == "unauthorized"


def test_revoked_device_is_refused(cfg):
    client = signed_in(cfg)
    client.app.state.ecg.security.revoke_device("d")
    assert _reject_reason(client, client.device_headers("d"), hello_message("d", 250.0, 10)) == "unauthorized"


def test_token_for_another_device_is_refused(cfg):
    client = signed_in(cfg)
    assert _reject_reason(client, client.device_headers("d"), hello_message("ecg-sim-01", 250.0, 10)) == "device_mismatch"


def test_simulator_cannot_claim_to_be_hardware(cfg):
    client = signed_in(cfg)
    hello = json.loads(hello_message("d", 250.0, 10))
    hello["simulated"] = False
    assert _reject_reason(client, client.device_headers("d"), json.dumps(hello)) == "source_mismatch"


def test_hardware_device_is_labelled_live_from_registry(cfg):
    client = signed_in(cfg)
    hello = json.loads(hello_message("hw-01", 250.0, 10))
    hello["simulated"] = False
    with client.websocket_connect("/ws/ingest", headers=client.device_headers("hw-01")) as ws:
        ws.send_text(json.dumps(hello))
        assert json.loads(ws.receive_text())["source"] == "LIVE"
        for msg in build_packets([2048] * 50, device_id="hw-01", fs=250.0, batch_size=10):
            ws.send_text(json.dumps(msg))
    assert client.get("/status").json()["mode"] == "LIVE"
    devices = {d["device_id"]: d for d in client.get("/devices").json()["items"]}
    assert devices["hw-01"]["last_seen"] is not None


# ---------- registry and binding rule ----------

def test_tokens_are_stored_only_as_hashes(tmp_path):
    store = SecurityStore(tmp_path / "ecg.sqlite3")
    device_token = store.add_device("dev-1", "hardware")
    admin_token = store.rotate_admin_token()
    raw = (tmp_path / "ecg.sqlite3").read_bytes()
    assert device_token.encode() not in raw and admin_token.encode() not in raw
    assert store.verify_device(device_token).device_id == "dev-1"
    assert store.verify_admin(admin_token)


def test_device_cannot_be_registered_twice_without_revoking(tmp_path):
    store = SecurityStore(tmp_path / "ecg.sqlite3")
    store.add_device("dev-1", "hardware")
    with pytest.raises(SecurityError):
        store.add_device("dev-1", "hardware")
    store.revoke_device("dev-1")
    new = store.add_device("dev-1", "simulator")
    assert store.verify_device(new).kind == "simulator"


@pytest.mark.parametrize("bad", ["", "has space", "x" * 65, "semi;colon"])
def test_device_id_validation(tmp_path, bad):
    with pytest.raises(SecurityError):
        SecurityStore(tmp_path / "ecg.sqlite3").add_device(bad, "hardware")


def test_binding_rule(tmp_path):
    store = SecurityStore(tmp_path / "ecg.sqlite3")
    with pytest.raises(SecurityError, match="access token"):
        check_binding("127.0.0.1", tls=False, store=store)
    store.rotate_admin_token()
    check_binding("127.0.0.1", tls=False, store=store)  # loopback over HTTP is allowed
    with pytest.raises(SecurityError, match="without TLS"):
        check_binding("0.0.0.0", tls=False, store=store)
    with pytest.raises(SecurityError, match="no registered devices"):
        check_binding("0.0.0.0", tls=True, store=store)
    store.add_device("dev-1", "hardware")
    check_binding("192.168.1.50", tls=True, store=store)


def test_is_loopback():
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.50") and not is_loopback("ecg.local")


def test_data_directory_is_owner_only(cfg, tmp_path):
    create_app(cfg)
    assert (tmp_path.stat().st_mode & 0o777) == 0o700
