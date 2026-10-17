# Device ↔ gateway protocol (v1)

This is the IoMT device link: the ESP32 sensing device (or the software simulator) is the
client, and the laptop gateway is the server.

Transport: WebSocket over TLS, `wss://<gateway>:8000/ws/ingest`, with text frames carrying
JSON. Plain `ws://` is accepted only when the gateway listens on `127.0.0.1` (local
development). See [security.md](security.md).

## 0. Connecting and authenticating

1. **TLS.** The device opens a TLS connection and verifies the gateway certificate
   against the project CA compiled into the firmware (`ca_cert.h`). If the chain or the
   host name does not match, the device does not connect.
2. **Device token.** The WebSocket upgrade request carries the device's token:

   ```
   GET /ws/ingest HTTP/1.1
   Authorization: Bearer ecgdev_…
   ```

   Tokens are issued with `scripts/security_setup.py add-device <id> --kind hardware|simulator`.
   The gateway stores only a hash of each token.
3. **Hello.** The first message is the hello described below. Its `device_id` must be
   the device the token was issued to.

Rejections happen before any data is accepted. The gateway sends one JSON error message
and closes the connection with code 1008 (policy violation):

```json
{"type": "error", "reason": "unauthorized", "detail": "missing, unknown or revoked device token"}
```

| `reason` | Cause |
|---|---|
| `unauthorized` | No `Authorization` header, or the token is unknown or revoked |
| `device_mismatch` | `hello.device_id` differs from the device the token belongs to |
| `source_mismatch` | `hello.simulated` disagrees with the registered kind (hardware or simulator) |
| `fs_mismatch` | The device's sampling rate differs from the gateway's configured rate |
| `malformed` | The first message is not a valid hello |

## 1. Hello (first message after connecting)

```json
{"v": 1, "type": "hello", "device_id": "ecg-esp32-01", "firmware": "0.1.0",
 "simulated": false, "fs": 250, "batch_size": 10}
```

- The stream is labelled from the **device registry**: `LIVE` for devices registered as
  `hardware` and `SIMULATED` for devices registered as `simulator`. `simulated` must agree
  with the registration, or the connection is refused (`source_mismatch`).
- `fs` is the sampling rate the device used. The server rejects the stream if
  it differs from the configured rate, to avoid silently wrong timing.

## 2. Samples

```json
{"v": 1, "type": "samples", "device_id": "ecg-esp32-01", "seq": 1042,
 "t_us": 4183200000, "fs": 250, "lo_plus": false, "lo_minus": false,
 "samples": [2051, 2049, 2060, 2110, 2300, 2050, 2047, 2049, 2051, 2050]}
```

| Field | Type | Meaning |
|---|---|---|
| `v` | int | protocol version, must be 1 |
| `type` | string | `"samples"` |
| `device_id` | string | configured device identifier |
| `seq` | int ≥ 0 | packet sequence number, increments by 1 per packet |
| `t_us` | int ≥ 0 | device microsecond timestamp of the **first** sample |
| `fs` | number > 0 | sampling rate (Hz) |
| `lo_plus`, `lo_minus` | bool | AD8232 lead-off indicators at the packet time |
| `samples` | int[] | ADC counts, 0 to 4095, one value per sample period |

Sample *i* has device time `t_us + i * 1e6 / fs`.

Samples are raw **integer ADC counts**. The server converts them to millivolts
using `AdcCalibration` (see `src/ecg/config.py`). Those constants are
placeholders and must be calibrated per device.

## 3. Validation rules (server side)

| Condition | Result |
|---|---|
| Not JSON, or not an object | rejected, `malformed` counter increases |
| Missing field or wrong type | rejected, `malformed` |
| Non-integer, negative or > 4095 sample | rejected, `malformed` |
| `v` ≠ 1 | rejected, `malformed` |
| `fs` differs from configured rate | rejected, `fs_mismatch` |
| `seq` equal to a recent seq | discarded, `duplicates` |
| `seq` jumps forward | gap recorded, `lost_packets` increases by the gap size |
| `seq` goes back to a much lower value | treated as a device restart, `session_restarts` |

Lost samples are counted as `lost_packets × batch size` (the batch size is fixed per
device). The server never fabricates samples to fill gaps. A packet that arrives after a
later one is counted as `out_of_order` and discarded.

## 4. Reconnection

The device retries with exponential backoff (1 s, doubling up to 30 s). On
reconnect it repeats the TLS handshake, sends its token again and sends a new `hello`. Sequence numbers continue from the device's
last value, so the server records any gap that occurred while disconnected.
If the device restarts, `seq` restarts from 0 and the server records a session
restart.

## 5. Keep-alive

The device sends a keep-alive at least every 2 s when no sample packet has been sent. The
server marks the device as disconnected after 5 s without any message.

A keep-alive is a `samples` message with an empty `samples` array. The server records that the device is alive, but
it does not track the keep-alive's `seq`, so it never counts as a duplicate or a gap. The
firmware reuses the sequence number of the last sample packet, and the next sample packet
continues the normal numbering.
