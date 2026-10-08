"""Scaled Hamming window, STFT / STFT magnitude, overlap-add inverse, and the SER of the paper.

Framing convention: frame m covers samples [m*S, m*S + L) of the padded signal.
The signal is padded with L - S zeros in front (and enough zeros at the end) so that
every original sample is covered by the same number of frames.
"""
from __future__ import annotations

import numpy as np

HAMMING_A = 0.54
HAMMING_B = -0.46


def hamming(L: int) -> np.ndarray:
    """Plain (unscaled) Hamming window a + b*cos(2*pi*n/L), n = 1..L."""
    n = np.arange(1, L + 1)
    return HAMMING_A + HAMMING_B * np.cos(2 * np.pi * n / L)


def scaled_hamming(L: int, S: int) -> np.ndarray:
    """Scaled Hamming window, Eq. (6) of Zhu et al. 2007.

    The factor 2*sqrt(S) / sqrt((4a^2 + 2b^2) L) makes the sum of the squared,
    overlapping windows equal to 1 (for L = 4S, and also for L = 8S).
    """
    a, b = HAMMING_A, HAMMING_B
    return 2 * np.sqrt(S) / np.sqrt((4 * a ** 2 + 2 * b ** 2) * L) * hamming(L)


def window_square_sum(w: np.ndarray, S: int, n_frames: int = 16) -> np.ndarray:
    """Sum of the squared windows overlap-added with hop S (used to check the scaling)."""
    L = len(w)
    total = np.zeros((n_frames - 1) * S + L)
    for m in range(n_frames):
        total[m * S:m * S + L] += w ** 2
    return total


def pad_signal(x: np.ndarray, L: int, S: int) -> np.ndarray:
    """Pad L - S zeros in front and zeros at the end so that the last frame fits exactly."""
    front = L - S
    n_frames = int(np.ceil((len(x) + front) / S))   # last frame starts at or after the last sample
    total = (n_frames - 1) * S + L
    return np.concatenate([np.zeros(front), x, np.zeros(total - front - len(x))])


def unpad_signal(y: np.ndarray, n_samples: int, L: int, S: int) -> np.ndarray:
    return y[L - S:L - S + n_samples]


def frames(x_pad: np.ndarray, L: int, S: int) -> np.ndarray:
    """Cut the padded signal into frames of length L with hop S (no window applied)."""
    n_frames = (len(x_pad) - L) // S + 1
    idx = np.arange(L)[None, :] + S * np.arange(n_frames)[:, None]
    return x_pad[idx]


def stft(x_pad: np.ndarray, w: np.ndarray, S: int) -> np.ndarray:
    """STFT, Eq. (1): one row per frame, bins 0..L/2 (real signal, so the other half is redundant)."""
    return np.fft.rfft(frames(x_pad, len(w), S) * w, axis=1)


def stft_magnitude(x_pad: np.ndarray, w: np.ndarray, S: int) -> np.ndarray:
    """STFT magnitude, Eq. (2). This is all the reconstruction algorithms get to see."""
    return np.abs(stft(x_pad, w, S))


def istft(X: np.ndarray, w: np.ndarray, S: int) -> np.ndarray:
    """Overlap-add inverse: y = sum_m w(n - mS) * IFFT(X_m), Eq. (4) with sum w^2 = 1."""
    L = len(w)
    y = np.zeros((X.shape[0] - 1) * S + L)
    time_frames = np.fft.irfft(X, n=L, axis=1)
    for m in range(X.shape[0]):
        y[m * S:m * S + L] += w * time_frames[m]
    return y


def ser_db(x_pad: np.ndarray, y_pad: np.ndarray, w: np.ndarray, S: int) -> float:
    """Signal-to-error ratio, Eq. (7), computed on STFT magnitudes (not on the waveform).

    The integrals over frequency become sums over all L FFT bins.
    """
    L = len(w)
    X = np.abs(np.fft.fft(frames(x_pad, L, S) * w, axis=1))
    Y = np.abs(np.fft.fft(frames(y_pad, L, S) * w, axis=1))
    return 10 * np.log10(np.sum(X ** 2) / np.sum((X - Y) ** 2))
