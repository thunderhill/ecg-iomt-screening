"""Start the IoMT gateway (API, device ingest and dashboard).

    python scripts/serve.py                         # http://127.0.0.1:8000, this computer only
    python scripts/serve.py --tls                   # https://127.0.0.1:8000
    python scripts/serve.py --host 0.0.0.0 --tls    # on the network (TLS is mandatory)

Sign-in is always required. Run scripts/security_setup.py init first.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402

from ecg.api.app import create_app  # noqa: E402
from ecg.config import AppConfig  # noqa: E402
from ecg.pki import PkiPaths  # noqa: E402
from ecg.security import SecurityError, SecurityStore, check_binding  # noqa: E402
from ecg.storage import Storage  # noqa: E402


def main() -> None:
    defaults = AppConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=defaults.api_host, help="address to listen on (non-loopback requires --tls)")
    p.add_argument("--port", type=int, default=defaults.api_port)
    p.add_argument("--tls", action="store_true", help="serve HTTPS/WSS with the certificate in secrets/")
    args = p.parse_args()

    secrets_dir = Path(defaults.security.secrets_dir)
    pki = PkiPaths(secrets_dir if secrets_dir.is_absolute() else ROOT / secrets_dir)
    data_dir = Path(defaults.data_dir)
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    store = SecurityStore(Storage(data_dir).db_path)
    try:
        check_binding(args.host, tls=args.tls, store=store)
    except SecurityError as exc:
        sys.exit(f"Not starting: {exc}")
    if args.tls and not (pki.server_cert.exists() and pki.server_key.exists()):
        sys.exit("Not starting: no server certificate; run scripts/security_setup.py init")

    cfg = replace(defaults, api_host=args.host, api_port=args.port, data_dir=str(data_dir),
                  models_dir=str(ROOT / defaults.models_dir),
                  security=replace(defaults.security, tls=args.tls))
    scheme = "https" if args.tls else "http"
    print(f"Dashboard: {scheme}://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/ui/")
    uvicorn.run(
        create_app(cfg),
        host=args.host,
        port=args.port,
        log_level="info",
        ssl_certfile=str(pki.server_cert) if args.tls else None,
        ssl_keyfile=str(pki.server_key) if args.tls else None,
    )


if __name__ == "__main__":
    main()
