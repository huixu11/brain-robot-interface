from __future__ import annotations

import numpy as np


def eeg_window_features(x: np.ndarray, fs_hz: float) -> np.ndarray:
    """Compute cheap, deterministic features for an EEG window.

    x: (n_samples, n_channels)
    Returns: (n_features,) float32
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x.ndim==2, got shape={x.shape}")
    n, c = x.shape
    if n < 8:
        raise ValueError("Window too short for FFT features")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")

    x = x.astype(np.float32, copy=False)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    rms = np.sqrt((x * x).mean(axis=0))

    # Simple bandpower estimates via rFFT bins (no SciPy dependency).
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz).astype(np.float32)
    spec = np.fft.rfft(x, axis=0)
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)

    def bandpower(f_lo: float, f_hi: float) -> np.ndarray:
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(mask):
            return np.zeros((c,), dtype=np.float32)
        # Mean power within band per channel.
        return power[mask].mean(axis=0)

    alpha = bandpower(8.0, 12.0)
    beta = bandpower(13.0, 30.0)

    feat = np.concatenate([mean, std, rms, alpha, beta], axis=0).astype(np.float32)
    return feat

