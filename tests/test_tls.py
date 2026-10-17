"""PKI and an end-to-end TLS test with a real uvicorn server."""

import asyncio
import socket
import ssl
import threading
import time

import httpx
import pytest
import uvicorn
from cryptography import x509

from ecg.api.app import create_app
from ecg.config import AppConfig, SecurityConfig
from ecg.pki import PkiPaths, create_ca, firmware_ca_header, issue_server_cert
from ecg.simulator import run_device


# ---------- certificate tooling ----------

def test_server_cert_chains_to_ca_with_sans(tmp_path):
    paths = PkiPaths(tmp_path / "secrets")
    ca = create_ca(paths)
    cert = issue_server_cert(paths, ["192.168.1.50", "ecg-laptop.local"])
    cert.verify_directly_issued_by(ca)
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "ecg-laptop.local" in san.get_values_for_type(x509.DNSName)
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    ips = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
    assert {"127.0.0.1", "192.168.1.50"} <= ips
    assert ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    assert not cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca


def test_private_keys_are_owner_only(tmp_path):
    paths = PkiPaths(tmp_path / "secrets")
    create_ca(paths)
    issue_server_cert(paths, [])
    for key in (paths.ca_key, paths.server_key):
        assert (key.stat().st_mode & 0o777) == 0o600
    assert (paths.root.stat().st_mode & 0o777) == 0o700


def test_firmware_header_has_certificate_but_no_key(tmp_path):
    paths = PkiPaths(tmp_path / "secrets")
    create_ca(paths)
    header = firmware_ca_header(paths.ca_cert.read_bytes())
    assert "BEGIN CERTIFICATE" in header and "PRIVATE" not in header
    assert "static const char CA_CERT_PEM[]" in header


# ---------- end to end over TLS ----------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def tls_server(tmp_path):
    paths = PkiPaths(tmp_path / "secrets")
    create_ca(paths)
    issue_server_cert(paths, [])
    cfg = AppConfig(data_dir=str(tmp_path / "data"), models_dir=str(tmp_path / "none"),
                    security=SecurityConfig(tls=True, secrets_dir=str(paths.root)))
    app = create_app(cfg)
    store = app.state.ecg.security
    admin = store.rotate_admin_token()
    device = store.add_device("ecg-sim-01", "simulator")
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                           ssl_certfile=str(paths.server_cert), ssl_keyfile=str(paths.server_key)))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "TLS server did not start"
    yield {"port": port, "ca": str(paths.ca_cert), "admin": admin, "device": device, "tmp": tmp_path}
    server.should_exit = True
    thread.join(timeout=10)


def _stream(srv, url, **kw):
    return asyncio.run(run_device(url, device_id="ecg-sim-01", fs=250.0, adc=AppConfig().adc, duration_s=6.0,
                                  realtime=False, connect_attempts=1, **kw))


def test_simulator_streams_over_wss_and_dashboard_reads_over_https(tls_server):
    port = tls_server["port"]
    result = _stream(tls_server, f"wss://127.0.0.1:{port}/ws/ingest", token=tls_server["device"], ca_file=tls_server["ca"])
    assert result["messages_sent"] == result["messages_planned"]
    ctx = ssl.create_default_context(cafile=tls_server["ca"])
    with httpx.Client(base_url=f"https://127.0.0.1:{port}", verify=ctx) as client:
        assert client.get("/status").status_code == 401
        assert client.post("/auth/login", json={"token": tls_server["admin"]}).status_code == 200
        status = client.get("/status")
        assert status.json()["mode"] == "SIMULATED"
        assert status.headers["strict-transport-security"].startswith("max-age=")


def test_client_trusting_another_ca_refuses_the_gateway(tls_server):
    other = PkiPaths(tls_server["tmp"] / "other")
    create_ca(other)
    with pytest.raises(ssl.SSLCertVerificationError):
        _stream(tls_server, f"wss://127.0.0.1:{tls_server['port']}/ws/ingest",
                token=tls_server["device"], ca_file=str(other.ca_cert))


def test_plaintext_websocket_to_tls_port_fails(tls_server):
    with pytest.raises(Exception):
        _stream(tls_server, f"ws://127.0.0.1:{tls_server['port']}/ws/ingest", token=tls_server["device"])


def test_wrong_device_token_is_refused_over_tls(tls_server):
    with pytest.raises(ConnectionRefusedError, match="unauthorized"):
        _stream(tls_server, f"wss://127.0.0.1:{tls_server['port']}/ws/ingest",
                token="ecgdev_not-a-real-token", ca_file=tls_server["ca"])
