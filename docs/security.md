# Security model

This prototype is an **IoMT** (Internet of Medical Things) system: a body-worn
sensing device sends ECG data over Wi-Fi to an analysis gateway, which serves a
dashboard. ECG waveforms are health data. This document describes how each
communication channel is protected, how to set it up, and what is **not**
protected.

> Research prototype, not a medical device. These controls are a sound baseline for a
> lab or classroom network. They have not been through a formal security review.

## What we protect, and from whom

| Asset | Why it matters |
|---|---|
| ECG waveforms and derived results | Health data about a person |
| Device identity | Fake or replayed data could produce false screening results |
| Settings and screening thresholds | Changing them silently changes every result |
| Access tokens, TLS keys | Whoever holds them can impersonate a device or the dashboard user |

The main threat is **another computer on the same Wi-Fi**. It could:

- read traffic (eavesdropping)
- send fake ECG packets as if it were the device (spoofing or injection)
- open the dashboard or API and read data or change settings
- pose as the server to the ESP32 (man-in-the-middle)

Out of scope: an attacker with administrator access to the laptop, physical
attacks on the ESP32, and denial of service.

## Channels and controls

| Channel | Protection |
|---|---|
| **Device → gateway** (`/ws/ingest`) | WebSocket over **TLS (`wss://`)**. The device verifies the server certificate against the project's own CA, which is compiled into the firmware. The device proves its identity with a **per-device token** sent in the `Authorization: Bearer` header of the WebSocket upgrade request. |
| **Browser → gateway** (dashboard and API) | **HTTPS** whenever the server listens on the network. Users sign in with an **access token**. The server then issues a session in an `HttpOnly`, `SameSite=Strict` cookie, marked `Secure` under TLS. Every API route except `/health` and `/auth/login` requires that session. |
| **Gateway → PhysioNet** (dataset download) | HTTPS, already in place (`scripts/prepare_mitbih.py`). |
| **Simulator → gateway** | Same as the device: `wss://` when TLS is on, and its own token, registered as kind `simulator`. |

### Data source labelling is enforced, not declared

Each device token is registered with a **kind**: `hardware` or `simulator`. The
gateway labels a stream **LIVE** or **SIMULATED** from that registry entry. The device's
own `simulated` flag must agree with it, or the connection is refused. A simulator
therefore cannot pass itself off as live hardware, even by mistake.

### When TLS is required

| Server binding | TLS | Sign-in |
|---|---|---|
| `127.0.0.1` (this computer only, the default) | optional (`--tls` to enable) | required |
| any network address, e.g. `0.0.0.0` or a LAN IP | **required**. The server refuses to start without a certificate, and without at least one registered device. | required |

Loopback traffic never leaves the computer, so plain HTTP there exposes nothing on the
network. Sign-in is still required, because other user accounts on the same computer
can reach loopback.

### Other controls

- **Tokens** are 256-bit random values. Only their SHA-256 hashes are stored, in the
  SQLite database, and comparison is constant-time. A token is shown once, when it is
  created.
- **Revocation:** a device token can be revoked at any time. Its next connection is
  refused.
- **Login throttling:** after 5 failed sign-ins from one address within 15 minutes,
  further attempts from that address are refused for 15 minutes.
- **Sessions** expire after 12 hours. Signing out deletes the session on the server.
- **Cross-site requests:** state-changing requests (`POST`, `PUT`) carrying a
  browser `Origin` header from another site are refused. Combined with
  `SameSite=Strict` cookies, this blocks cross-site request forgery.
- **Response headers:** `Strict-Transport-Security` (under TLS),
  `Content-Security-Policy: frame-ancestors 'none'`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`.
- **Logs** never contain tokens, cookies or waveform samples.
- **File permissions:** `secrets/` is created with mode 0700 and its key files with
  0600. The data directory is created with 0700.

The sign-in screen states whether the connection is encrypted:

![Sign-in screen](screenshots/sign-in.png)

## Setting it up

All secrets live in `secrets/`, which is git-ignored.

```bash
# 1. Create the local CA, the server certificate and the dashboard access token.
#    List every name or address the ESP32 and browsers will use to reach the laptop.
.venv/bin/python scripts/security_setup.py init --host 192.168.1.50 --host ecg-laptop.local

# 2. Register each device. The token is printed once: copy it into the device's config.
.venv/bin/python scripts/security_setup.py add-device ecg-esp32-01 --kind hardware
.venv/bin/python scripts/security_setup.py add-device ecg-sim-01  --kind simulator

# 3. Start the server on the network with TLS.
.venv/bin/python scripts/serve.py --host 0.0.0.0 --tls
```

Then:

- **Browser:** open `https://<laptop-address>:8000/ui/` and sign in with the access
  token from step 1. The certificate comes from your own CA, so the browser warns until
  you import `secrets/ca.pem` as a trusted authority (or accept the warning for this
  site only).
- **ESP32:** `init` writes `firmware/esp32_ecg_logger/ca_cert.h`. Put the device token
  in `config.h` and flash. See [wiring.md](wiring.md).
- **Simulator:**
  `ECG_DEVICE_TOKEN=<token> .venv/bin/python scripts/run_simulator.py --url wss://127.0.0.1:8000/ws/ingest --ca secrets/ca.pem`

Other commands: `list` shows devices and their last-seen times, `revoke-device <id>`
revokes a device, and `rotate-admin` replaces the dashboard token and ends all sessions.

For quick local development, `scripts/serve.py` with no options listens on
`127.0.0.1` over plain HTTP. Sign-in and device tokens are still required.

## What is not protected (residual risks)

- **Data at rest is not encrypted by the application.** Recordings and results are
  stored in `data/` as SQLite and NumPy files. No names or patient identifiers are
  stored, but the waveforms are still health data. Use full-disk encryption on the
  laptop.
- **ESP32 flash is not encrypted by default.** Anyone holding the device can read the
  Wi-Fi password and device token from it. Enable ESP32 flash encryption and secure boot
  for anything beyond lab use, and revoke the token of a lost device.
- **Certificate dates need a clock.** TLS certificate validity is checked against the
  ESP32's clock, which the firmware sets over SNTP. On a network without internet
  access, provide a local NTP server, or the connection may fail.
- **Local CA trust.** Anyone holding `secrets/ca.key` can issue certificates that the
  devices and your browser trust. Keep it private, and never commit it.
- **Single dashboard role.** There is one access token, with full rights. There are no
  separate user accounts or audit trail of who changed what. Configuration changes
  are recorded with a timestamp only.
- **No replay protection beyond TLS.** Within one TLS connection, packets cannot be
  replayed or altered. Sequence numbers detect gaps and duplicates, not attacks.

## Interoperability (FHIR)

FHIR is **not included**. This system does not exchange data with an electronic health
record or any external clinical system, stores no patient identity, and its outputs are
research screening flags rather than clinical observations. Adding a FHIR interface
would widen the attack surface and imply a clinical use the prototype does not support.

If integration is needed later, the natural mapping is a FHIR R4 `Observation` export
per recording: heart rate, RR intervals and HRV as components, the waveform as
`SampledData`, and screening flags clearly marked as research output. It would be
exposed through an authenticated, read-only endpoint.
