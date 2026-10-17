# Safety and intended use

**This is a research prototype. It is not a medical device.** It has not been
clinically validated, and it must not be used to make or guide any medical
decision.

## Patient-connected hardware

- The AD8232 and its electrodes connect to a person. Such a circuit needs
  appropriate **isolation** and electrical-safety design before any real use.
  This prototype does not provide that design.
- **Power the ESP32 from a battery or an isolated supply.** Never connect the
  patient circuit to mains-powered equipment, including a laptop charger
  connected to a grounded outlet, without a suitable isolation design.
- Do not connect a USB programming cable to a person while the board is
  connected to a mains-powered computer. Program the board first, disconnect
  it, then attach electrodes.
- Use only electrodes and leads that are rated and intended for ECG use.

## Measurement limitations

- The ESP32 ADC is non-linear and noisy. Its readings are **not clinically
  calibrated**. `AdcCalibration` in `src/ecg/config.py` holds placeholder
  values that must be replaced by a per-device calibration and validation.
- Filtering reduces noise and drift. It does not remove all artifacts.
- Motion, poor electrode contact, muscle activity, mains interference and
  lead-off all change the signal. Signal-quality warnings are heuristics.

## Interpreting screening output

- Screening results are **preliminary research flags**, not diagnoses.
- "No configured flag detected" does **not** mean the rhythm is normal.
- Thresholds are configurable research parameters, not universal clinical
  thresholds.
- Simulated data (`SIMULATED`) is software-generated. It must never be
  presented as patient data, and it must never be used to claim clinical
  accuracy.

## Data handling

- Recordings contain no patient name or identifier. Do not add one unless the
  data has a proper consent and protection plan.
- ECG waveforms are health data. Every connection to the gateway is authenticated, and
  network traffic is encrypted with TLS. The application does not encrypt stored data,
  so use full-disk encryption on the laptop. See [security.md](security.md).
