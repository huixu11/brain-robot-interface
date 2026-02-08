from __future__ import annotations

import numpy as np


def eeg_window_features(x: np.ndarray, fs_hz: float, *, include_fft: bool = True) -> np.ndarray:
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

    # Some dataset chunks contain NaNs; make feature extraction robust.
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # Remove DC offset per channel (helps stability across sessions/subjects).
    x = x - x.mean(axis=0, keepdims=True)

    # Time-domain energy proxy (commonly used in EEG pipelines): log-variance per channel.
    var = np.var(x, axis=0).astype(np.float32)
    logvar = np.log(var + 1e-6).astype(np.float32)

    if not include_fft:
        return logvar

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

    # Classic motor-related EEG bands.
    theta = bandpower(4.0, 8.0)
    alpha = bandpower(8.0, 13.0)  # mu/alpha
    beta = bandpower(13.0, 30.0)
    total = bandpower(4.0, 30.0)

    # Log-bandpower reduces heavy-tailed variance; relative bandpower reduces amplitude scaling effects.
    log_theta = np.log(theta + 1e-6)
    log_alpha = np.log(alpha + 1e-6)
    log_beta = np.log(beta + 1e-6)
    rel_theta = theta / (total + 1e-6)
    rel_alpha = alpha / (total + 1e-6)
    rel_beta = beta / (total + 1e-6)

    feat = np.concatenate([logvar, log_theta, log_alpha, log_beta, rel_theta, rel_alpha, rel_beta], axis=0).astype(
        np.float32
    )
    return feat
