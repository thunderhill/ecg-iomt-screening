"""Create TLS material and manage access tokens (see docs/security.md).

    python scripts/security_setup.py init --host 192.168.1.50 [--host ecg-laptop.local]
    python scripts/security_setup.py add-device ecg-esp32-01 --kind hardware
    python scripts/security_setup.py revoke-device ecg-esp32-01
    python scripts/security_setup.py list
    python scripts/security_setup.py rotate-admin
    python scripts/security_setup.py reissue-cert --host 192.168.1.51

Tokens are printed once and stored only as hashes. Private keys stay in secrets/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryptography import x509  # noqa: E402

from ecg.config import AppConfig  # noqa: E402
from ecg.pki import PkiPaths, create_ca, firmware_ca_header, issue_server_cert  # noqa: E402
from ecg.security import DEVICE_KINDS, SecurityError, SecurityStore  # noqa: E402
from ecg.storage import Storage  # noqa: E402

FIRMWARE_CA_HEADER = ROOT / "firmware" / "esp32_ecg_logger" / "ca_cert.h"


def _store(cfg: AppConfig) -> SecurityStore:
    data = Path(cfg.data_dir)
    data = data if data.is_absolute() else ROOT / data
    return SecurityStore(Storage(data).db_path)


def _paths(cfg: AppConfig) -> PkiPaths:
    root = Path(cfg.security.secrets_dir)
    return PkiPaths(root if root.is_absolute() else ROOT / root)


def _show_token(label: str, token: str) -> None:
    print(f"\n{label}:\n\n    {token}\n\nThis is shown only once. Store it somewhere safe.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init", help="create the CA, server certificate and dashboard token")
    init.add_argument("--host", action="append", default=[], help="name or IP the gateway is reached at (repeatable)")
    init.add_argument("--force", action="store_true", help="replace an existing CA (devices must be reflashed)")
    re = sub.add_parser("reissue-cert", help="new server certificate from the existing CA (e.g. new IP)")
    re.add_argument("--host", action="append", default=[])
    add = sub.add_parser("add-device", help="register a device and print its token")
    add.add_argument("device_id")
    add.add_argument("--kind", choices=DEVICE_KINDS, required=True)
    rev = sub.add_parser("revoke-device", help="revoke a device token")
    rev.add_argument("device_id")
    sub.add_parser("list", help="list registered devices")
    sub.add_parser("rotate-admin", help="new dashboard token; ends every session")
    args = p.parse_args()

    cfg = AppConfig()
    store = _store(cfg)
    paths = _paths(cfg)
    try:
        if args.cmd == "init":
            if paths.ca_cert.exists() and not args.force:
                sys.exit(f"A CA already exists in {paths.root}. Use reissue-cert, or init --force to replace it "
                         "(every device must then be reflashed with the new ca_cert.h).")
            create_ca(paths)
            cert = issue_server_cert(paths, args.host)
            FIRMWARE_CA_HEADER.write_text(firmware_ca_header(paths.ca_cert.read_bytes()))
            names = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            print(f"CA and server certificate written to {paths.root} (keys are mode 0600).")
            print(f"Server certificate is valid for: {', '.join(str(n.value) for n in names.value)}")
            print(f"Firmware CA header written to {FIRMWARE_CA_HEADER}.")
            print(f"Import {paths.ca_cert} into your browser as a trusted authority to avoid warnings.")
            _show_token("Dashboard access token", store.rotate_admin_token())
        elif args.cmd == "reissue-cert":
            if not paths.ca_key.exists():
                sys.exit("No CA found. Run init first.")
            issue_server_cert(paths, args.host)
            print(f"New server certificate written to {paths.server_cert}. Restart the gateway.")
        elif args.cmd == "add-device":
            token = store.add_device(args.device_id, args.kind)
            _show_token(f"Token for {args.kind} device {args.device_id!r} (put it in the device configuration)", token)
        elif args.cmd == "revoke-device":
            if store.revoke_device(args.device_id):
                print(f"Revoked {args.device_id!r}. Its next connection will be refused.")
            else:
                sys.exit(f"No active device named {args.device_id!r}.")
        elif args.cmd == "list":
            devices = store.list_devices()
            if not devices:
                print("No devices registered.")
            for d in devices:
                seen = dt.datetime.fromtimestamp(d.last_seen).isoformat(timespec="seconds") if d.last_seen else "never"
                print(f"{d.device_id:24} {d.kind:10} {'REVOKED' if d.revoked else 'active':8} last seen {seen}")
        elif args.cmd == "rotate-admin":
            _show_token("New dashboard access token (all existing sessions have been ended)", store.rotate_admin_token())
    except SecurityError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
