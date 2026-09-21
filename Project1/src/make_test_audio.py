"""Generate test audio for the equalizer (saved to audio/input/).

    speech.wav       macOS text-to-speech (falls back to a synthetic vowel sequence)
    music_synth.wav  8 s synthetic pop loop: bass, chords, kick, hi-hat
    pink_noise.wav   5 s pink noise (equal energy per octave)
    sweep.wav        6 s logarithmic sine sweep 20 Hz to 20 kHz

Any of your own wav / flac / mp3 files can be dropped into audio/input/ as well.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.signal import chirp, resample_poly

from utils import AUDIO_IN_DIR, DEFAULT_FS, ensure_dirs, load_audio, save_audio

SPEECH_TEXT = ("This is a test of the equal loudness equalizer. "
               "The quick brown fox jumps over the lazy dog, "
               "while the bass drum and the cymbals play in the background.")


def _normalise(x, peak_db=-3.0):
    return (x / max(np.abs(x).max(), 1e-9) * 10 ** (peak_db / 20)).astype(np.float32)


def _env(n, attack, decay, fs):
    t = np.arange(n) / fs
    e = np.exp(-t / decay)
    a = min(int(attack * fs), n)
    e[:a] *= np.linspace(0, 1, a)
    return e


def make_speech(fs=DEFAULT_FS):
    say = shutil.which("say")
    if say:
        tmp = os.path.join(tempfile.gettempdir(), "asp_speech.aiff")
        subprocess.run([say, "-o", tmp, "--data-format=BEI16@44100", SPEECH_TEXT], check=True)
        x, fs_in = load_audio(tmp, mono=True)
        os.remove(tmp)
        if fs_in != fs:
            x = resample_poly(x, fs, fs_in).astype(np.float32)
        return _normalise(x)
    # fallback: vowel-like formant tones so the file exists on non-mac systems
    vowels = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240)]
    out = []
    for f1, f2, f3 in vowels * 2:
        n = int(0.4 * fs)
        t = np.arange(n) / fs
        f0 = 120 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
        src = np.zeros(n)
        for k in range(1, 40):
            src += np.sin(2 * np.pi * k * np.cumsum(f0) / fs) / k
        spec = np.fft.rfft(src)
        f = np.fft.rfftfreq(n, 1 / fs)
        h = sum(1 / (1 + ((f - fc) / 120) ** 2) for fc in (f1, f2, f3))
        out.append(np.fft.irfft(spec * h, n) * _env(n, 0.02, 0.6, fs))
        out.append(np.zeros(int(0.1 * fs)))
    return _normalise(np.concatenate(out))


def make_music(fs=DEFAULT_FS, seconds=8.0, bpm=120):
    beat = 60 / bpm
    n = int(seconds * fs)
    t = np.arange(n) / fs
    y = np.zeros(n)
    bass_notes = [55.0, 65.41, 73.42, 82.41]         # A1 C2 D2 E2
    chords = [(220.0, 261.63, 329.63), (261.63, 329.63, 392.0),
              (293.66, 349.23, 440.0), (329.63, 415.30, 493.88)]
    bars = int(seconds / (2 * beat))
    for b in range(bars):
        s, e = int(b * 2 * beat * fs), int(min((b + 1) * 2 * beat, seconds) * fs)
        tt = t[s:e] - t[s]
        env = _env(e - s, 0.01, 0.8, fs)
        f0 = bass_notes[b % 4]
        y[s:e] += 0.5 * env * (np.sin(2 * np.pi * f0 * tt) + 0.5 * np.sin(4 * np.pi * f0 * tt)
                               + 0.25 * np.sin(6 * np.pi * f0 * tt))
        ch = np.zeros(e - s)
        for fc in chords[b % 4]:
            for k in range(1, 9):
                ch += np.sin(2 * np.pi * k * fc * tt) / k
        y[s:e] += 0.08 * _env(e - s, 0.02, 0.5, fs) * ch
    for k in range(int(seconds / beat)):                 # kick on every beat
        s = int(k * beat * fs)
        m = int(0.3 * fs)
        tt = np.arange(m) / fs
        fk = 50 + 100 * np.exp(-tt / 0.03)
        y[s:s + m] += 0.8 * np.exp(-tt / 0.12) * np.sin(2 * np.pi * np.cumsum(fk) / fs)
    rng = np.random.default_rng(1)
    for k in range(int(2 * seconds / beat)):             # hi-hat on eighth notes
        s = int(k * beat / 2 * fs)
        m = int(0.06 * fs)
        noise = rng.standard_normal(m)
        spec = np.fft.rfft(noise)
        f = np.fft.rfftfreq(m, 1 / fs)
        noise = np.fft.irfft(spec * (f > 6000), m)
        y[s:s + m] += 0.25 * np.exp(-np.arange(m) / fs / 0.02) * noise
    return _normalise(y[:n])


def make_pink_noise(fs=DEFAULT_FS, seconds=5.0, seed=0):
    n = int(seconds * fs)
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    return _normalise(np.fft.irfft(spec, n))


def make_sweep(fs=DEFAULT_FS, seconds=6.0):
    t = np.arange(int(seconds * fs)) / fs
    x = chirp(t, f0=20, t1=seconds, f1=20000, method="logarithmic")
    r = int(0.05 * fs)
    x[:r] *= np.linspace(0, 1, r)
    x[-r:] *= np.linspace(1, 0, r)
    return _normalise(x)


def main(fs=DEFAULT_FS):
    ensure_dirs()
    items = {"speech.wav": make_speech, "music_synth.wav": make_music,
             "pink_noise.wav": make_pink_noise, "sweep.wav": make_sweep}
    for name, fn in items.items():
        path = os.path.join(AUDIO_IN_DIR, name)
        x = fn(fs)
        save_audio(path, x, fs)
        print(f"  wrote {os.path.relpath(path)}  ({len(x) / fs:.1f} s)")


if __name__ == "__main__":
    main()
