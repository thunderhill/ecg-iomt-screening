"""Local certificate authority and server certificate for the gateway's TLS.

The project runs its own small CA so that devices and browsers can verify the
gateway without a public certificate: the ESP32 has the CA certificate compiled
in, and browsers can import it. Keys are EC P-256. Private keys are written with
mode 0600 into a 0700 directory and never leave the laptop.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

CA_DAYS = 3650
SERVER_DAYS = 825  # browsers reject longer-lived server certificates


@dataclass(frozen=True)
class PkiPaths:
    root: Path

    @property
    def ca_key(self) -> Path:
        return self.root / "ca.key"

    @property
    def ca_cert(self) -> Path:
        return self.root / "ca.pem"

    @property
    def server_key(self) -> Path:
        return self.root / "server.key"

    @property
    def server_cert(self) -> Path:
        return self.root / "server.pem"


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    path.chmod(0o600)


def _key_pem(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _san(hosts: list[str]) -> x509.SubjectAlternativeName:
    names: list[x509.GeneralName] = []
    for h in dict.fromkeys(hosts):  # de-duplicate, keep order
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            names.append(x509.DNSName(h))
    return x509.SubjectAlternativeName(names)


def create_ca(paths: PkiPaths, *, name: str = "ECG IoMT local CA") -> x509.Certificate:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.root.chmod(0o700)
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=CA_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False,
                                     data_encipherment=False, key_agreement=False, key_cert_sign=True,
                                     crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    _write_private(paths.ca_key, _key_pem(key))
    paths.ca_cert.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert


def issue_server_cert(paths: PkiPaths, hosts: list[str]) -> x509.Certificate:
    """Server certificate valid for ``hosts`` (always includes localhost and 127.0.0.1)."""
    ca_key = serialization.load_pem_private_key(paths.ca_key.read_bytes(), password=None)
    ca_cert = x509.load_pem_x509_certificate(paths.ca_cert.read_bytes())
    all_hosts = ["localhost", "127.0.0.1", *hosts]
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, all_hosts[-1])]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=SERVER_DAYS))
        .add_extension(_san(all_hosts), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_cert.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    _write_private(paths.server_key, _key_pem(key))
    paths.server_cert.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert


def firmware_ca_header(ca_pem: bytes) -> str:
    """C header embedding the CA certificate for the ESP32 (public data, no key)."""
    lines = ca_pem.decode("ascii").strip().splitlines()
    body = "\n".join(f'"{line}\\n"' for line in lines)
    return (
        "// Generated by scripts/security_setup.py: the CA the device trusts for wss://.\n"
        "// Public certificate only. Regenerate if the CA changes. Git-ignored.\n"
        "#pragma once\n\n"
        f"static const char CA_CERT_PEM[] =\n{body};\n"
    )
