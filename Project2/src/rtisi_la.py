"""RTISI-LA: RTISI with look-ahead (Zhu et al. 2007, Sections II-B to II-D, Figs. 3 and 4).

A frame buffer holds the frames that are not committed yet: frame m and the
`look_ahead` frames after it. At each step a new frame m+k enters the buffer
(its estimate starts at zero), then n_iter iterations are run on all
uncommitted frames together:
  buffer signal = committed signal + sum over uncommitted frames of (estimate * w)
  for every uncommitted frame: read its range from the buffer signal, apply its
  analysis window, FFT, keep the phase, combine with the target magnitude, IFFT
After the iterations, frame m is committed (added to the output for good).
Each frame stays in the buffer for (look_ahead + 1) steps, so it gets
n_iter * (look_ahead + 1) iterations in total.

Asymmetric analysis window (Section II-D): the newest frame only overlaps
earlier frames, so its partial signal fades out towards the end. Its analysis
window is the time-reversed envelope of that partial signal, Fig. 4(a); after
its first iteration the envelope also includes the new frame itself, Fig. 4(b).
All other uncommitted frames use the scaled Hamming window, Fig. 4(c).
With look_ahead = 0 and no asymmetric window this is the same as RTISI.
"""
from __future__ import annotations

import numpy as np


def asymmetric_windows(w: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray]:
    """Analysis windows for the newest frame: (first iteration, later iterations).

    Each frame enters the output as estimate * w, and the estimate itself carries
    one window, so the envelope of the overlap-added frames is a sum of w^2.
    """
    L = len(w)
    w_sq = np.concatenate([w ** 2, np.zeros(L)])           # w^2 with zeros after the window
    shifted = [w_sq[j * S:j * S + L] for j in range(L // S)]   # w^2(n + jS), n = 0..L-1
    envelope_before = np.sum(shifted[1:], axis=0)          # earlier frames only, Fig. 4(a)
    envelope_with_new = np.sum(shifted, axis=0)            # earlier frames + the new frame, Fig. 4(b)
    return envelope_before[::-1].copy(), envelope_with_new[::-1].copy()


def rtisi_la(target_mag: np.ndarray, w: np.ndarray, S: int, n_iter: int, look_ahead: int,
             asymmetric: bool = True) -> np.ndarray:
    """Rebuild a (padded) signal from its STFT magnitude.

    target_mag : (frames, L/2 + 1) magnitude spectrogram, from stft.stft_magnitude
    w          : synthesis window and normal analysis window (the scaled Hamming window)
    S          : hop size
    n_iter     : iterations per step (each frame gets n_iter * (look_ahead + 1) in total)
    look_ahead : number of future frames k kept uncommitted
    asymmetric : use the asymmetric analysis window for the newest frame
    """
    L = len(w)
    n_frames = target_mag.shape[0]
    first_window, later_window = asymmetric_windows(w, S)

    y = np.zeros((n_frames - 1) * S + L)       # committed frames, overlap-added
    estimates = np.zeros((look_ahead + 1, L))  # time-domain estimates of the buffered frames
    oldest = 0                                 # index of the first uncommitted frame

    for step in range(n_frames + look_ahead):
        newest = min(step, n_frames - 1)       # after the last frame enters, the buffer just drains
        new_frame_entered = step < n_frames
        buffered = np.arange(oldest, newest + 1)
        count = len(buffered)
        if new_frame_entered:
            estimates[count - 1] = 0.0

        windows = np.tile(w, (count, 1))
        start = oldest * S
        stop = newest * S + L
        for it in range(n_iter):
            # overlap-add committed signal and the current estimates of the buffered frames
            signal = y[start:stop].copy()
            for i, m in enumerate(buffered):
                signal[m * S - start:m * S - start + L] += estimates[i] * w

            # read every buffered frame back, window it, and do the M-constrained transform
            if asymmetric and new_frame_entered:
                windows[-1] = first_window if it == 0 else later_window
            segments = np.stack([signal[m * S - start:m * S - start + L] for m in buffered])
            phase = np.angle(np.fft.rfft(segments * windows, axis=1))
            estimates[:count] = np.fft.irfft(target_mag[buffered] * np.exp(1j * phase), n=L, axis=1)

        if step >= look_ahead:
            # commit the oldest frame and shift the buffer by one
            y[oldest * S:oldest * S + L] += estimates[0] * w
            estimates[:-1] = estimates[1:].copy()
            oldest += 1

    return y
