"""Run one reconstruction setting on one clip and measure it.

Measures:
  SER        Eq. (7) of the paper, on STFT magnitudes
  loudness   RMS level of the reconstruction minus RMS level of the original, in dB
             (the assignment asks for similar loudness; 0 dB means the same)
  runtime    seconds spent in the reconstruction only
  RTF        real-time factor = runtime / audio duration (below 1 means faster than real time)
"""
from __future__ import annotations

import time

import numpy as np

from common import rms_db, window_length
from rtisi import rtisi
from rtisi_la import rtisi_la
from stft import pad_signal, scaled_hamming, ser_db, stft_magnitude, unpad_signal


def reconstruct(x: np.ndarray, fs: int, method: str, window_ms: float, hop_divisor: int, n_iter: int,
                look_ahead: int = 0, asymmetric: bool = True, window: np.ndarray | None = None) -> dict:
    """Throw away the phase of x, rebuild it with RTISI or RTISI-LA, and score the result.

    method      : "rtisi" or "rtisi_la"
    window_ms   : window length L in ms (rounded to a multiple of 8 samples)
    hop_divisor : S = L / hop_divisor (4 means 75 % overlap, 8 means 87.5 %)
    n_iter      : RTISI: iterations per frame; RTISI-LA: iterations per step
    window      : optional window to use instead of the scaled Hamming window
    """
    L = window_length(fs, window_ms)
    S = L // hop_divisor
    w = scaled_hamming(L, S) if window is None else window

    x_pad = pad_signal(x, L, S)
    target_mag = stft_magnitude(x_pad, w, S)          # magnitude only, the phase is discarded here

    start = time.perf_counter()
    if method == "rtisi":
        y_pad = rtisi(target_mag, w, S, n_iter)
    elif method == "rtisi_la":
        y_pad = rtisi_la(target_mag, w, S, n_iter, look_ahead, asymmetric)
    else:
        raise ValueError(f"unknown method {method!r}")
    runtime = time.perf_counter() - start

    y = unpad_signal(y_pad, len(x), L, S)
    duration = len(x) / fs
    return {
        "L": L,
        "S": S,
        "total_iter_per_frame": n_iter * (look_ahead + 1) if method == "rtisi_la" else n_iter,
        "ser_db": ser_db(x_pad, y_pad, w, S),
        "loudness_diff_db": rms_db(y) - rms_db(x),
        "runtime_s": runtime,
        "rtf": runtime / duration,
        "y": y,
    }
