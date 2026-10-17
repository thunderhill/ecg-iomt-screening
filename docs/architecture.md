# IoMT architecture

The system follows the usual Internet of Medical Things (IoMT) layering: a medical
sensing device, a secure transport, an edge gateway that does the analysis, and a
dashboard for the person reviewing results.

```
 ┌───────────────────────────┐   wss:// (TLS, CA-pinned)    ┌───────────────────────────────┐
 │ 1. Sensing device         │   + per-device token         │ 3. Edge gateway (laptop)       │
 │ AD8232 front end          │ ───────────────────────────▶ │ FastAPI server                 │
 │ ESP32: sampling, batching │                              │  ingest: validation, gaps      │
 │ (or software simulator)   │                              │  processing: filter, R peaks   │
 └───────────────────────────┘                              │  HR, RR, HRV, quality          │
            ▲                                               │  screening models + XAI        │
            │ electrodes (patient side, isolated power)     │  SQLite + NumPy storage        │
                                                            └──────────────┬────────────────┘
                                                                           │ https:// (TLS when on
                                                                           │ the network) + session
                                                            ┌──────────────▼────────────────┐
                                                            │ 4. Dashboard (browser)         │
                                                            │ React: live, screening,        │
                                                            │ explanations, history, eval    │
                                                            └───────────────────────────────┘
```

Layer 2, the **secure transport**, is the TLS channel between the device and the
gateway. It is described in [security.md](security.md).

## Layers

| Layer | Component | Code | Responsibilities |
|---|---|---|---|
| 1. Sensing device | AD8232 + ESP32 | `firmware/esp32_ecg_logger/` | Timed ADC sampling, lead-off detection, bounded buffering, batching, reconnecting |
| 1. Sensing device (software) | Simulator | `src/ecg/simulator.py` | The same protocol with synthetic ECG; always labelled SIMULATED |
| 2. Secure transport | WebSocket over TLS | `docs/protocol.md` | Confidentiality, server authentication (pinned CA), device authentication (token) |
| 3. Edge gateway | FastAPI | `src/ecg/api/`, `src/ecg/pipeline.py` | Validation, signal processing, screening, explanations, storage |
| 4. Dashboard | React | `web/` | Presenting results with evidence and limitations; signed-in access only |

Analysis runs on the gateway, not on the ESP32. The ESP32 only acquires and transmits.
Latency figures in the model card are therefore laptop CPU figures.

## Trust boundaries

1. **Patient ↔ device:** electrical safety. Use battery or isolated power only (see
   [safety.md](safety.md)).
2. **Device ↔ network:** everything leaving the ESP32 is encrypted, and the device only
   talks to a server whose certificate chains to the project CA.
3. **Network ↔ gateway:** every connection must authenticate. Devices use a device token,
   and people use the dashboard access token.
4. **Gateway ↔ disk:** data at rest is not encrypted by the application. Rely on OS disk
   encryption ([security.md](security.md), residual risks).

## Data flow

1. The ESP32 samples at 250 Hz and sends batches of 10 samples, with a sequence number
   and timestamp, over `wss://`.
2. The gateway checks the device token, the declared sampling rate and every packet. It
   tracks gaps, duplicates and restarts on the device timeline, and never fills in
   missing samples.
3. Processing produces heart rate, RR intervals, HRV, signal quality, rule-based and
   model-based screening flags, and explanations.
4. The dashboard polls the gateway over the signed-in session.
5. Recordings are saved on request as raw samples, labelled with their source (LIVE,
   SIMULATED or OFFLINE).

## Interoperability

No FHIR or HL7 interface is included, because nothing in this system exchanges data
with clinical systems. The reasoning, and how one would be added later, is in
[security.md](security.md#interoperability-fhir).
