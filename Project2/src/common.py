"""Shared settings and helpers: paths, the list of clips, audio loading and saving."""
from __future__ import annotations

import os

import numpy as np
import soundfile as sf

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Project2/, parent of src/
AUDIO_IN_DIR = os.path.join(PROJECT_DIR, "audio", "input")
AUDIO_OUT_DIR = os.path.join(PROJECT_DIR, "audio", "output")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")

# Speech or music, decided by looking at the spectrograms of the given clips.
# audio7 is speech with some background music; it is counted as speech.
CLIP_TYPE = {
    "audio1": "music",
    "audio2": "speech",
    "audio3": "speech",
    "audio4": "speech",
    "audio5": "speech",
    "audio6": "speech",
    "audio7": "speech",
    "audio8": "speech",
    "audio9": "music",
    "audio10": "music",
    "audio11": "music",
    "audio12": "music",
}
CLIP_NAMES = list(CLIP_TYPE)

# Window length used in the paper (23.2 ms = 512 samples at 22050 Hz)
WINDOW_MS = 23.2


def ensure_dirs() -> None:
    for d in (AUDIO_OUT_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)


def load_clip(name: str) -> tuple[np.ndarray, int]:
    """Read audio/input/<name>.mp3; stereo clips are averaged to mono."""
    x, fs = sf.read(os.path.join(AUDIO_IN_DIR, name + ".mp3"), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


def save_wav(name: str, y: np.ndarray, fs: int) -> str:
    path = os.path.join(AUDIO_OUT_DIR, name + ".wav")
    sf.write(path, np.clip(y, -1.0, 1.0), fs, subtype="PCM_16")
    return path


def window_length(fs: int, window_ms: float = WINDOW_MS) -> int:
    """Window length in samples, rounded to a multiple of 8 so that S = L/4 and S = L/8 are integers."""
    return int(8 * round(window_ms * fs / 1000 / 8))


def rms_db(x: np.ndarray) -> float:
    """RMS level in dB (relative to full scale 1.0)."""
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def save_fig(fig, name: str) -> None:
    """Save a figure as 300 DPI PNG and as PDF in figures/."""
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
