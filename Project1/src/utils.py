"""Shared helpers: project paths, tone synthesis, audio I/O, figure saving, band analysis."""
from __future__ import annotations

import os

import numpy as np
import soundfile as sf

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Project1/, parent of src/
DATA_DIR = os.path.join(PROJECT_DIR, "data")
AUDIO_IN_DIR = os.path.join(PROJECT_DIR, "audio", "input")
AUDIO_OUT_DIR = os.path.join(PROJECT_DIR, "audio", "output")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
DEFAULT_FS = 44100


def ensure_dirs() -> None:
    for d in (DATA_DIR, AUDIO_IN_DIR, AUDIO_OUT_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)


def db_to_lin(db):
    return 10.0 ** (np.asarray(db, dtype=float) / 20.0)


def lin_to_db(x):
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-12))


def make_tone(freq: float, level_db: float, duration: float = 0.7, fs: int = DEFAULT_FS,
              ramp: float = 0.02) -> np.ndarray:
    """Pure tone at `level_db` dBFS (peak) with raised-cosine on/off ramps to avoid clicks."""
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * freq * t)
    nr = min(int(ramp * fs), n // 2)
    if nr > 0:
        r = 0.5 * (1 - np.cos(np.pi * np.arange(nr) / nr))
        x[:nr] *= r
        x[-nr:] *= r[::-1]
    return (db_to_lin(level_db) * x).astype(np.float32)


def silence(duration: float, fs: int = DEFAULT_FS) -> np.ndarray:
    return np.zeros(int(round(duration * fs)), dtype=np.float32)


def load_audio(path: str, mono: bool = True):
    """Return (x, fs). x is float32, shape (n,) if mono else (n, channels)."""
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    if mono:
        x = x.mean(axis=1)
    return x, int(fs)


def save_audio(path: str, x: np.ndarray, fs: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, np.asarray(x, dtype=np.float32), fs, subtype="PCM_16")


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def save_figure(fig, path_no_ext: str) -> None:
    """Save as 300 dpi PNG and vector PDF."""
    os.makedirs(os.path.dirname(os.path.abspath(path_no_ext)), exist_ok=True)
    fig.savefig(path_no_ext + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")


def third_octave_levels(x: np.ndarray, fs: int, f_min: float = 25.0, f_max: float = 16000.0):
    """Band levels (dB re digital full scale, mean-square) in nominal 1/3-octave bands.

    Returns (centre_freqs, levels_db). Bands above fs/2 are NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    power = 2.0 * np.abs(spec) ** 2 / n ** 2  # one-sided mean-square per bin
    power[0] /= 2.0
    if n % 2 == 0:
        power[-1] /= 2.0
    centres = 1000.0 * 2.0 ** (np.arange(-16, 13) / 3.0)
    centres = centres[(centres >= f_min) & (centres <= f_max)]
    levels = np.full(len(centres), np.nan)
    for i, fc in enumerate(centres):
        lo, hi = fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6)
        if hi > fs / 2:
            break
        m = (f >= lo) & (f < hi)
        if m.any():
            levels[i] = 10 * np.log10(max(power[m].sum(), 1e-20))
    return centres, levels


CURVE_ORDER = ["threshold", "very_low", "low", "comfortable", "loud"]


def order_contours(curves: dict, min_gap_db: float = 2.0):
    """Enforce the physical ordering of measured contours at every frequency.

    A contour matched to a louder 1 kHz reference cannot lie below a contour matched to a
    quieter reference, and no contour can lie below the hearing threshold. Points that
    violate this (judgement noise near threshold) are raised to the lower curve plus
    min_gap_db. Returns (corrected {name: array}, changed {name: bool array}).
    """
    names = [n for n in CURVE_ORDER if n in curves] + [n for n in curves if n not in CURVE_ORDER]
    if "ref_dbfs" in str(curves.get(names[0], {})):
        pass
    values = {n: np.asarray([np.nan if v is None else v for v in curves[n]["dbfs"]], dtype=float)
              for n in names}
    changed = {n: np.zeros(len(values[n]), dtype=bool) for n in names}
    ordered = [n for n in names if n in CURVE_ORDER]
    ordered.sort(key=lambda n: (CURVE_ORDER.index(n)))
    # sort measured levels by their reference level, threshold first
    ordered = [n for n in ordered if n == "threshold"] + sorted(
        [n for n in ordered if n != "threshold"], key=lambda n: curves[n]["ref_dbfs"])
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        lo, up = values[lower], values[upper]
        m = np.isfinite(lo) & np.isfinite(up) & (up < lo + min_gap_db)
        up[m] = lo[m] + min_gap_db
        changed[upper] |= m
    return values, changed


def list_measurements() -> list[str]:
    """All elc_measured*.json files in data/, oldest first."""
    import glob
    files = glob.glob(os.path.join(DATA_DIR, "elc_measured*.json"))
    return sorted(files, key=os.path.getmtime)
