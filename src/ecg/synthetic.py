"""SYNTHETIC ECG GENERATOR: for software testing and the hardware simulator only.

Output from this module is NOT patient data and must never be used to claim
clinical accuracy. Every object it produces carries ``source="synthetic"`` so
that the rest of the system can label it SIMULATED and refuse to mix it with
real recordings in evaluation.

The waveform is a sum of Gaussian components per beat (P, Q, R, S, T), in the
spirit of McSharry et al. (2003), with seeded RNG for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCE = "synthetic"

# (amplitude mV, centre offset from R in s, width s)
_WAVES = {
    "P": (0.12, -0.16, 0.025),
    "Q": (-0.10, -0.035, 0.010),
    "R": (1.00, 0.0, 0.012),
    "S": (-0.20, 0.03, 0.012),
    "T": (0.28, 0.22, 0.045),
}


@dataclass(frozen=True)
class SyntheticECG:
    signal: np.ndarray  # mV
    fs: float
    r_peak_indices: np.ndarray  # ground-truth R-peak sample indices
    source: str = SOURCE


def generate_ecg(
    duration_s: float,
    fs: float = 250.0,
    hr_bpm: float = 70.0,
    hr_std_bpm: float = 2.0,
    noise_std_mv: float = 0.01,
    baseline_amp_mv: float = 0.1,
    baseline_hz: float = 0.25,
    seed: int = 0,
) -> SyntheticECG:
    """Generate a synthetic ECG recording of ``duration_s`` seconds.

    Heart rate varies per beat around ``hr_bpm`` with standard deviation
    ``hr_std_bpm``. Gaussian measurement noise and sinusoidal baseline wander
    are added. The same seed always yields the same output.
    """
    if duration_s <= 0 or fs <= 0 or hr_bpm <= 0:
        raise ValueError("duration_s, fs and hr_bpm must be positive")
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs

    # Beat schedule: draw RR intervals until the recording is covered.
    beat_times = []
    current = 0.6  # first beat about 0.6 s in
    mean_rr = 60.0 / hr_bpm
    # Converting an HR spread to an RR spread: d(RR) = 60 * d(HR) / HR^2.
    rr_std = 60.0 * hr_std_bpm / (hr_bpm**2)
    while current < duration_s + 0.5:
        beat_times.append(current)
        rr = rng.normal(mean_rr, rr_std)
        current += float(np.clip(rr, 0.35, 2.0))
    beat_times_arr = np.asarray(beat_times)

    x = np.zeros(n)
    for bt in beat_times_arr:
        lo = int(max(0, np.floor((bt - 0.5) * fs)))
        hi = int(min(n, np.ceil((bt + 0.6) * fs)))
        if hi <= lo:
            continue
        seg_t = t[lo:hi]
        for amp, offset, width in _WAVES.values():
            x[lo:hi] += amp * np.exp(-((seg_t - (bt + offset)) ** 2) / (2 * width**2))

    x += baseline_amp_mv * np.sin(2 * np.pi * baseline_hz * t + rng.uniform(0, 2 * np.pi))
    x += rng.normal(0.0, noise_std_mv, size=n)

    r_idx = np.round(beat_times_arr * fs).astype(int)
    r_idx = r_idx[(r_idx >= 0) & (r_idx < n)]
    return SyntheticECG(signal=x, fs=float(fs), r_peak_indices=r_idx)
