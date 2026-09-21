"""ELC-based audio equalizer: short-time Fourier magnitude weighting.

The equalizer boosts or cuts each frequency by the amount the chosen equal-loudness
contour says the ear needs, so that all frequencies are perceived at the same
loudness. Processing is frame based (default 2048 samples, 50 percent overlap,
sqrt-Hann analysis and synthesis windows), so the same code runs on files and on a
live microphone stream with about 46 ms of latency.

Gain
  G(f) = L(f) - L(1 kHz) for the chosen contour: every frequency is lifted to the
  loudness of the 1 kHz tone. --strength scales the whole curve.

Examples
    python src/equalizer.py audio/input/speech.wav                       # ISO 226, 40 phon
    python src/equalizer.py in.wav -o out.wav --phon 60 --max-boost 15
    python src/equalizer.py in.wav --curve data/elc_measured.json --level comfortable
    python src/equalizer.py in.wav --phon 40 --strength 0.5
    python src/equalizer.py --record 10                                   # record 10 s, equalize, save
    python src/equalizer.py --mic 10 -o audio/output/mic_eq.wav           # 10 s live with monitoring
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import scipy.fft as fft
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal.windows import hann

from iso226 import interp_log_freq, iso226_spl
from utils import (AUDIO_OUT_DIR, DATA_DIR, DEFAULT_FS, db_to_lin, ensure_dirs, load_audio, order_contours,
                   rms, save_audio)


# ----------------------------------------------------------------------------- gain design
@dataclass
class GainDesign:
    fs: int
    nfft: int
    bin_freqs: np.ndarray      # frequency of each rfft bin
    gain_db: np.ndarray        # gain applied to each bin (after clipping)
    curve_freqs: np.ndarray    # frequencies of the source contour
    curve_gain_db: np.ndarray  # unclipped gain at the contour points
    label: str

    @property
    def gain_lin(self) -> np.ndarray:
        return db_to_lin(self.gain_db).astype(np.float32)

    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.label.lower()).strip("_")


def load_measured_curve(path: str, level: str):
    """Return (freqs, dbfs, metadata) of one curve from an elc_measure.py JSON file."""
    with open(path) as fh:
        d = json.load(fh)
    if level not in d["curves"]:
        raise ValueError(f"curve '{level}' not found; available: {list(d['curves'])}")
    freqs = np.asarray(d["freqs"], dtype=float)
    y = order_contours(d["curves"])[0][level]   # physically ordered version of the curves
    m = np.isfinite(y)
    if m.sum() < 3:
        raise ValueError(f"curve '{level}' has fewer than 3 valid points")
    return freqs[m], y[m], d


def relative_to_ref(freqs, level, ref_freq=1000.0):
    """Contour minus its value at the reference frequency."""
    return np.asarray(level, float) - interp_log_freq(freqs, level, [ref_freq])[0]


def smooth_log_freq(bins, gain_db, octaves: float):
    """Gaussian smoothing of a per-bin gain curve along log frequency (sigma = octaves/2)."""
    from scipy.ndimage import gaussian_filter1d
    step = 1.0 / 48                                   # grid points per octave
    lf = np.log2(bins[1:])
    grid = np.arange(lf[0], lf[-1] + step, step)
    g = np.interp(grid, lf, gain_db[1:])
    gs = gaussian_filter1d(g, sigma=max(octaves / 2 / step, 1e-6), mode="nearest")
    out = np.array(gain_db, dtype=float)
    out[1:] = np.interp(lf, grid, gs)
    return out


def design_gain(fs: int, nfft: int, curve: str = "iso", phon: float = 40.0,
                level: str = "comfortable", max_boost: float = 20.0, max_cut: float = 20.0,
                strength: float = 1.0, smooth_oct: float = 0.0) -> GainDesign:
    """Build the per-bin gain vector for an rfft of size nfft.

    smooth_oct > 0 smooths the interpolated curve along log frequency (useful for
    measured contours, whose single points carry a few dB of judgement noise).
    """
    if curve == "iso":
        f1, l1 = iso226_spl(phon)
        label = f"ISO226 {phon:g}phon"
    else:
        f1, l1, _ = load_measured_curve(curve, level)
        label = f"measured {level}"
    g = relative_to_ref(f1, l1) * float(strength)
    if strength != 1.0:
        label += f" x{strength:g}"
    bins = fft.rfftfreq(nfft, 1.0 / fs)
    gb = interp_log_freq(f1, g, bins)          # clamps to edge values outside the contour range
    if smooth_oct > 0:
        gb = smooth_log_freq(bins, gb, smooth_oct)
        label += f" sm{smooth_oct:g}"
    low = bins < 20.0
    gb[low] *= bins[low] / 20.0                 # fade to 0 dB below 20 Hz (no subsonic or DC boost)
    gb = np.clip(gb, -abs(max_cut), abs(max_boost))
    return GainDesign(fs, nfft, bins, gb, np.asarray(f1), np.asarray(g), label)


# ----------------------------------------------------------------------------- STFT engine
class STFTEqualizer:
    """Weighted overlap-add STFT processor with a fixed magnitude gain per bin.

    process_block() handles one hop of samples at a time (for live streams);
    process_signal() is a vectorised offline version with identical results.
    """

    def __init__(self, gain_lin: np.ndarray, nfft: int = 2048, hop: int | None = None):
        hop = hop or nfft // 2
        if nfft % hop != 0 or hop > nfft // 2:
            raise ValueError("hop must divide nfft and be at most nfft/2")
        self.nfft, self.hop = int(nfft), int(hop)
        self.gain = np.asarray(gain_lin, dtype=np.float32)
        if self.gain.shape != (nfft // 2 + 1,):
            raise ValueError("gain must have nfft//2 + 1 values")
        self.win = np.sqrt(hann(nfft, sym=False)).astype(np.float32)
        # normalisation so that the analysis*synthesis windows sum to one across overlaps
        wsum = np.zeros(nfft)
        for k in range(0, nfft, hop):
            wsum += np.roll(self.win.astype(float) ** 2, k)
        self.norm = np.float32(1.0 / wsum.max())
        self.reset()

    @property
    def delay_samples(self) -> int:
        return self.nfft - self.hop

    def reset(self) -> None:
        self.inbuf = np.zeros(self.nfft, dtype=np.float32)
        self.outbuf = np.zeros(self.nfft, dtype=np.float32)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Feed `hop` samples, get `hop` samples back (delayed by nfft - hop samples)."""
        hop, nfft = self.hop, self.nfft
        x = np.asarray(x, dtype=np.float32).ravel()
        if len(x) != hop:
            raise ValueError(f"process_block expects exactly {hop} samples")
        self.inbuf[:-hop] = self.inbuf[hop:]
        self.inbuf[-hop:] = x
        spec = fft.rfft(self.inbuf * self.win)
        frame = fft.irfft(spec * self.gain, n=nfft).astype(np.float32) * self.win * self.norm
        self.outbuf += frame
        out = self.outbuf[:hop].copy()
        self.outbuf[:-hop] = self.outbuf[hop:]
        self.outbuf[-hop:] = 0.0
        return out

    def process_signal(self, x: np.ndarray) -> np.ndarray:
        """Offline, vectorised over frames. Output is time aligned with the input."""
        hop, nfft = self.hop, self.nfft
        x = np.asarray(x, dtype=np.float32).ravel()
        n = len(x)
        pad_front = nfft - hop
        n_frames = int(np.ceil((n + pad_front + (nfft - hop) - nfft) / hop)) + 1
        total = (n_frames - 1) * hop + nfft
        xp = np.zeros(total, dtype=np.float32)
        xp[pad_front:pad_front + n] = x
        frames = sliding_window_view(xp, nfft)[::hop] * self.win
        spec = fft.rfft(frames, axis=1, workers=-1)
        y = fft.irfft(spec * self.gain, n=nfft, axis=1, workers=-1).astype(np.float32) * (self.win * self.norm)
        out = np.zeros(total, dtype=np.float32)
        for j in range(nfft // hop):                      # overlap-add, one slab per overlap position
            out[j * hop:j * hop + n_frames * hop] += y[:, j * hop:(j + 1) * hop].reshape(-1)
        return out[pad_front:pad_front + n]


# ----------------------------------------------------------------------------- file processing
def normalise(x: np.ndarray, y: np.ndarray, mode: str = "rms", ceiling: float = 0.99):
    """Scale the equalized signal. Returns (y, info)."""
    info = {"norm": mode, "scale_db": 0.0, "peak_limited": False}
    if mode == "rms":
        s = rms(x) / max(rms(y), 1e-12)
    elif mode == "peak":
        s = ceiling / max(np.abs(y).max(), 1e-12)
    else:
        s = 1.0
    y = y * np.float32(s)
    info["scale_db"] = float(20 * np.log10(max(s, 1e-12)))
    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak > ceiling:
        y = y * np.float32(ceiling / peak)
        info["peak_limited"] = True
        info["scale_db"] += float(20 * np.log10(ceiling / peak))
    info["peak_dbfs"] = float(20 * np.log10(max(np.abs(y).max(), 1e-12)))
    return y, info


def equalize_signal(x: np.ndarray, design: GainDesign, hop: int | None = None, norm: str = "rms"):
    eq = STFTEqualizer(design.gain_lin, design.nfft, hop)
    if x.ndim == 1:
        y = eq.process_signal(x)
    else:
        y = np.stack([eq.process_signal(x[:, c]) for c in range(x.shape[1])], axis=1)
        eq.reset()
    y, info = normalise(x, y, norm)
    return y, info


def default_output_path(in_path: str, design: GainDesign) -> str:
    stem = os.path.splitext(os.path.basename(in_path))[0]
    return os.path.join(AUDIO_OUT_DIR, f"{stem}__{design.slug()}.wav")


def equalize_file(in_path: str, out_path: str | None = None, design: GainDesign | None = None,
                  hop: int | None = None, norm: str = "rms", mono: bool = False, verbose: bool = True,
                  **design_kwargs) -> dict:
    """Equalize one audio file and save it. Returns a dict with timing and level info."""
    ensure_dirs()
    x, fs = load_audio(in_path, mono=mono)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if design is None:
        design = design_gain(fs, design_kwargs.pop("nfft", 2048), **design_kwargs)
    elif design.fs != fs:
        design = _redesign(design, fs)
    out_path = out_path or default_output_path(in_path, design)
    t0 = time.perf_counter()
    y, info = equalize_signal(x, design, hop, norm)
    dt = time.perf_counter() - t0
    save_audio(out_path, y, fs)
    dur = len(x) / fs
    info.update({"input": in_path, "output": out_path, "fs": fs, "duration_s": dur,
                 "process_s": dt, "realtime_factor": dur / max(dt, 1e-9), "label": design.label,
                 "nfft": design.nfft, "hop": hop or design.nfft // 2})
    if verbose:
        print(f"  {os.path.basename(in_path)} -> {os.path.relpath(out_path)}")
        print(f"    curve {design.label}, nfft {design.nfft}, hop {info['hop']}, fs {fs} Hz")
        print(f"    {dur:.1f} s of audio processed in {dt * 1000:.0f} ms "
              f"({info['realtime_factor']:.0f}x real time); gain {info['scale_db']:+.1f} dB, "
              f"peak {info['peak_dbfs']:.1f} dBFS" + (", peak limited" if info["peak_limited"] else ""))
    return info


def _redesign(design: GainDesign, fs: int) -> GainDesign:
    """Re-interpolate an existing design onto a different sample rate."""
    bins = fft.rfftfreq(design.nfft, 1.0 / fs)
    gb = interp_log_freq(design.bin_freqs[1:], design.gain_db[1:], bins)
    gb[0] = 0.0
    return GainDesign(fs, design.nfft, bins, gb, design.curve_freqs, design.curve_gain_db, design.label)


# ----------------------------------------------------------------------------- live microphone
def run_microphone(design: GainDesign, seconds: float, out_path: str, hop: int | None = None,
                   monitor: bool = True, device=None, raw_path: str | None = None):
    """Stream microphone -> equalizer -> speakers, and save raw and equalized audio."""
    import sounddevice as sd
    ensure_dirs()
    fs = design.fs
    eq = STFTEqualizer(design.gain_lin, design.nfft, hop)
    mid = (design.bin_freqs >= 100) & (design.bin_freqs <= 6000)
    makeup = np.float32(db_to_lin(-float(np.mean(design.gain_db[mid]))))  # keep mid band level roughly unchanged
    raw, out, times, clipped = [], [], [], 0

    def callback(indata, outdata, frames, time_info, status):
        nonlocal clipped
        if status:
            print("   stream status:", status)
        t0 = time.perf_counter()
        x = indata[:, 0]
        y = eq.process_block(x) * makeup
        c = np.abs(y) > 0.99
        clipped += int(c.sum())
        y = np.clip(y, -0.99, 0.99)
        raw.append(x.copy())
        out.append(y)
        outdata[:, 0] = y if monitor else 0.0
        times.append(time.perf_counter() - t0)

    if monitor:
        print("  Monitoring is ON: wear headphones to avoid feedback (or pass --no-monitor).")
    print(f"  Recording {seconds:.0f} s from the microphone with {design.label} ... speak now.")
    with sd.Stream(samplerate=fs, blocksize=eq.hop, channels=1, dtype="float32",
                   callback=callback, device=device):
        sd.sleep(int(seconds * 1000))
    x = np.concatenate(raw) if raw else np.zeros(0, np.float32)
    y = np.concatenate(out) if out else np.zeros(0, np.float32)
    save_audio(out_path, y, fs)
    raw_path = raw_path or out_path.replace(".wav", "_raw.wav")
    save_audio(raw_path, x, fs)
    blk_ms = 1000 * eq.hop / fs
    info = {"output": out_path, "raw": raw_path, "blocks": len(times),
            "block_ms": blk_ms, "mean_proc_ms": 1000 * float(np.mean(times)) if times else 0.0,
            "max_proc_ms": 1000 * float(np.max(times)) if times else 0.0,
            "latency_ms": 1000 * (eq.hop + eq.delay_samples) / fs, "clipped_samples": clipped}
    print(f"  Saved {os.path.relpath(out_path)} (raw: {os.path.relpath(raw_path)})")
    print(f"  {info['blocks']} blocks of {blk_ms:.1f} ms; processing {info['mean_proc_ms']:.2f} ms mean, "
          f"{info['max_proc_ms']:.2f} ms max per block; algorithmic latency {info['latency_ms']:.0f} ms; "
          f"clipped samples {clipped}")
    return info


def record_and_equalize(design: GainDesign, seconds: float, out_path: str, hop: int | None = None,
                        norm: str = "rms", device=None) -> dict:
    """Record from the microphone, equalize offline, save processed and raw wav files."""
    import sounddevice as sd
    ensure_dirs()
    fs = design.fs
    print(f"  Recording {seconds:.0f} s from the microphone ... speak now.")
    x = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="float32", device=device)
    sd.wait()
    x = x[:, 0]
    print("  Recording finished, processing ...")
    t0 = time.perf_counter()
    y, info = equalize_signal(x, design, hop, norm)
    dt = time.perf_counter() - t0
    raw_path = out_path.replace(".wav", "_raw.wav")
    save_audio(out_path, y, fs)
    save_audio(raw_path, x, fs)
    info.update({"output": out_path, "raw": raw_path, "duration_s": seconds, "process_s": dt,
                 "label": design.label})
    return info


# ----------------------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="input audio file (wav, flac, mp3, aiff ...)")
    p.add_argument("-o", "--output", default=None, help="output wav path")
    p.add_argument("--curve", default="iso", help="'iso' or path to a JSON from elc_measure.py")
    p.add_argument("--phon", type=float, default=40.0, help="ISO contour (playback level) in phon")
    p.add_argument("--level", default="comfortable", help="curve name in the measured JSON")
    p.add_argument("--max-boost", type=float, default=20.0, help="gain ceiling in dB")
    p.add_argument("--max-cut", type=float, default=20.0, help="attenuation floor in dB")
    p.add_argument("--strength", type=float, default=1.0, help="scale the dB gain curve (0 to 1)")
    p.add_argument("--smooth", type=float, default=0.0, metavar="OCTAVES",
                   help="smooth the gain curve along log frequency (e.g. 0.5 for measured curves)")
    p.add_argument("--nfft", type=int, default=2048, help="analysis window length in samples")
    p.add_argument("--hop", type=int, default=None, help="hop size (default nfft/2)")
    p.add_argument("--norm", choices=["rms", "peak", "none"], default="rms", help="output normalisation")
    p.add_argument("--mono", action="store_true", help="mix a stereo file down to mono")
    p.add_argument("--record", type=float, default=None, metavar="SECONDS",
                   help="record from the microphone, then equalize and save")
    p.add_argument("--mic", type=float, default=None, metavar="SECONDS", help="live microphone mode (monitoring)")
    p.add_argument("--no-monitor", action="store_true", help="do not play the live output")
    p.add_argument("--fs", type=int, default=DEFAULT_FS, help="sample rate for microphone mode")
    p.add_argument("--device", default=None, help="sounddevice device index or name")
    p.add_argument("--play", action="store_true", help="play the processed file when done")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    kw = dict(curve=args.curve, phon=args.phon, level=args.level, max_boost=args.max_boost,
              max_cut=args.max_cut, strength=args.strength, smooth_oct=args.smooth)
    device = None
    if args.device is not None:
        device = int(args.device) if args.device.isdigit() else args.device

    if args.record is not None:
        design = design_gain(args.fs, args.nfft, **kw)
        out = args.output or os.path.join(AUDIO_OUT_DIR, f"mic__{design.slug()}.wav")
        info = record_and_equalize(design, args.record, out, hop=args.hop, norm=args.norm, device=device)
        print(f"  Saved {os.path.relpath(out)} (raw recording: {os.path.relpath(info['raw'])})")
        out_path = out
    elif args.mic is not None:
        design = design_gain(args.fs, args.nfft, **kw)
        out = args.output or os.path.join(AUDIO_OUT_DIR, f"mic__{design.slug()}.wav")
        run_microphone(design, args.mic, out, hop=args.hop, monitor=not args.no_monitor, device=device)
        out_path = out
    elif args.input:
        x, fs = load_audio(args.input, mono=False)
        design = design_gain(fs, args.nfft, **kw)
        info = equalize_file(args.input, args.output, design, hop=args.hop, norm=args.norm, mono=args.mono)
        out_path = info["output"]
    else:
        build_parser().print_help()
        return
    if args.play:
        import sounddevice as sd
        y, fs = load_audio(out_path, mono=False)
        print("  Playing ...")
        sd.play(y, fs, device=device)
        sd.wait()


if __name__ == "__main__":
    main()
