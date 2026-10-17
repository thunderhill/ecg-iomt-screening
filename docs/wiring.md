# Wiring and configuration (ESP32 + AD8232)

> **Read `docs/safety.md` first.** Power and isolation matter more than any
> pin choice in this document.

## Pin map (default, `firmware/esp32_ecg_logger/config.example.h`)

| AD8232 pin | ESP32 pin | Notes |
|---|---|---|
| OUTPUT | GPIO34 (ADC1_CH6) | Analog ECG. GPIO34 is input-only, which suits an ADC input. |
| LO+ | GPIO25 | Digital. High when the right-arm electrode is off. |
| LO- | GPIO26 | Digital. High when the left-arm electrode is off. |
| SDN | GPIO27 | Digital. High keeps the AD8232 active. |
| 3.3V | 3V3 | Use a regulated 3.3 V supply. |
| GND | GND | Single ground reference. |

Use **ADC1** pins for analog reads. ADC2 pins are unavailable while Wi-Fi is
active, and that is why the firmware uses GPIO34. Other ESP32 variants
(ESP32-S2, S3, C3) have different ADC pins and different ADC behaviour. Check
the pinout for your exact board before wiring.

## Electrodes

Standard three-lead placement: RA (right arm), LA (left arm), RL (right leg,
reference). Place electrodes according to the AD8232 datasheet. Clean and dry
the skin, and use fresh electrodes. Poor contact causes most lead-off and
noise events.

## Sampling

- Target: `SAMPLE_RATE_HZ = 250`. The firmware measures the achieved rate and
  prints it every 5 s (`diag: rate=...`). Check that value on your board.
- A FreeRTOS task with `vTaskDelayUntil` provides the timing. Jitter from the
  scheduler is typically around a millisecond. It is visible in the timestamps
  and is not hidden by the design.
- Samples are 12-bit ADC counts (0–4095) at 11 dB attenuation.

## Wi-Fi, security and server

The ESP32 is an IoMT device carrying health data, so it only talks to the gateway over
TLS and authenticates with its own token. See [security.md](security.md) for the full
model.

1. On the laptop, create the security material once, listing the address the ESP32
   will use to reach it:
   `.venv/bin/python scripts/security_setup.py init --host 192.168.1.50`.
   This writes `firmware/esp32_ecg_logger/ca_cert.h` (the CA the device will trust).
2. Register the board and copy the printed token:
   `.venv/bin/python scripts/security_setup.py add-device ecg-esp32-01 --kind hardware`.
3. `cp config.example.h config.h`, then fill in the Wi-Fi name and password, the laptop
   address (it must match one of the `--host` values from step 1), the device id and the
   device token. `config.h` and `ca_cert.h` are git-ignored.
4. Start the gateway on the network with TLS:
   `.venv/bin/python scripts/serve.py --host 0.0.0.0 --tls`. It refuses to listen on the
   network without TLS.
5. Upload the sketch with Arduino IDE (ESP32 core 3.x) or `arduino-cli`, with the
   **WebSockets** library (Links2004) installed.

The firmware sets its clock over SNTP before connecting, because TLS checks the
certificate's validity dates. On a Wi-Fi network without internet access, set
`NTP_SERVER` in `config.h` to a local time server.

## Verifying your build

- Serial monitor at 115200 baud should show `wifi: connecting`, `time: synced`, then
  `ws: connected`, then `diag:` lines.
- If it shows `ws: disconnected` repeatedly, check the dashboard's device list (Settings)
  and the server log. An `unauthorized` rejection means a wrong or revoked token, and a
  TLS failure usually means the laptop address is not in the certificate or the clock is
  not set.
- The dashboard header should show **LIVE**, not SIMULATED.
- `dropped` in the diag line should stay at 0 on a good network.
