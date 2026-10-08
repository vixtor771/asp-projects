"""RTISI: real-time iterative spectrogram inversion (Zhu et al. 2007, Section II-A, Fig. 2).

Frames are built one at a time, left to right. For frame m:
  partial = overlap-add of the frames already built, inside frame m's range
  buffer  = partial
  repeat n_iter times (one "iteration" = one FFT + one IFFT):
      phase    = angle(FFT(buffer * w))
      estimate = IFFT(target magnitude * exp(j * phase))
      buffer   = estimate * w + partial          (reset, not accumulated)
  the last buffer is kept as the output in frame m's range.
For the first frame the partial frame is all zeros, so the first phase is zero.
"""
from __future__ import annotations

import numpy as np


def rtisi(target_mag: np.ndarray, w: np.ndarray, S: int, n_iter: int) -> np.ndarray:
    """Rebuild a (padded) signal from its STFT magnitude.

    target_mag : (frames, L/2 + 1) magnitude spectrogram, from stft.stft_magnitude
    w          : analysis and synthesis window (the scaled Hamming window)
    S          : hop size
    n_iter     : transform iterations per frame
    """
    L = len(w)
    n_frames = target_mag.shape[0]
    y = np.zeros((n_frames - 1) * S + L)
    for m in range(n_frames):
        seg = slice(m * S, m * S + L)
        partial = y[seg].copy()
        buffer = partial
        for _ in range(n_iter):
            phase = np.angle(np.fft.rfft(buffer * w))
            estimate = np.fft.irfft(target_mag[m] * np.exp(1j * phase), n=L)
            buffer = estimate * w + partial
        y[seg] = buffer
    return y
