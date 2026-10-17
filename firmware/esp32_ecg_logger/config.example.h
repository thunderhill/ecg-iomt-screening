// Copy this file to config.h and edit it. config.h is git-ignored, so Wi-Fi
// credentials never enter version control.
#pragma once

// ---- Wi-Fi ----
#define WIFI_SSID        "your-network-name"
#define WIFI_PASSWORD    "your-password"

// ---- Laptop gateway (see docs/protocol.md and docs/security.md) ----
// Must match a --host given to `scripts/security_setup.py init`, or TLS verification fails.
#define SERVER_HOST      "192.168.1.50"
#define SERVER_PORT      8000
#define SERVER_PATH      "/ws/ingest"

// ---- Device identity and credentials ----
#define DEVICE_ID        "ecg-esp32-01"
#define FIRMWARE_VERSION "0.2.0"
// Printed once by `scripts/security_setup.py add-device ecg-esp32-01 --kind hardware`.
#define DEVICE_TOKEN     "ecgdev_paste-your-token-here"

// ---- Clock (TLS checks certificate dates) ----
#define NTP_SERVER       "pool.ntp.org"   // use a local time server on networks without internet

// ---- Acquisition ----
// Target sampling rate. The sampler checks its real achieved rate and reports
// it in the diagnostics line; it does not assume the target was met.
#define SAMPLE_RATE_HZ   250
#define BATCH_SIZE       10              // samples per packet (about 40 ms at 250 Hz)
#define ADC_PIN          34              // GPIO34: ADC1 channel 6, input-only
#define LO_PLUS_PIN      25              // AD8232 LO+ (high = lead off)
#define LO_MINUS_PIN     26              // AD8232 LO- (high = lead off)
#define SDN_PIN          27              // AD8232 SDN (low = shutdown); keep high when running

// Bounded queue between the sampler and the network task (in batches).
#define QUEUE_DEPTH      64              // about 2.5 s of samples at 250 Hz / 10 per batch

// Keep-alive interval when no samples are queued (ms).
#define KEEPALIVE_MS    2000
