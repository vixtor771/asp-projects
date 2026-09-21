"""Run the equalizer on every file in audio/input/ with two curves (ISO 226 40 phon and the
measured comfortable curve) and analyse the result.

    python src/analyze_eq.py                       # all presets, all input files
    python src/analyze_eq.py --elc data/elc_measured.json --nfft 4096

Outputs
    audio/output/<input>__<preset>.wav
    figures/fig2_eq_gain_curves      gain curves of the presets
    figures/eq_summary.md            band-level changes and timing, as a markdown table
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
import numpy as np

from equalizer import design_gain, equalize_file
from plot_elc import find_elc_file
from utils import AUDIO_IN_DIR, AUDIO_OUT_DIR, FIG_DIR, ensure_dirs, load_audio, save_figure, third_octave_levels

AUDIO_EXT = (".wav", ".flac", ".mp3", ".aiff", ".aif", ".ogg")


def make_presets(elc_path: str | None, max_boost: float, max_cut: float):
    presets = {"iso40": dict(curve="iso", phon=40.0)}
    if elc_path:
        presets["measured_comfortable"] = dict(curve=elc_path, level="comfortable", smooth_oct=0.5)
    for p in presets.values():
        p.update(max_boost=max_boost, max_cut=max_cut)
    return presets


def band_change(fc, before, after):
    """Mean level change (dB) in low (<250 Hz), mid (250 to 2000 Hz) and high (>=2 kHz) bands.

    Bands whose input level is more than 60 dB below the loudest band are ignored.
    """
    diff = after - before
    valid = np.isfinite(before) & np.isfinite(after) & (before > np.nanmax(before) - 60)
    groups = {"low": fc < 250, "mid": (fc >= 250) & (fc < 2000), "high": fc >= 2000}
    return {k: float(np.mean(diff[m & valid])) if (m & valid).any() else np.nan for k, m in groups.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elc", default=None, help="measured ELC JSON (default: data/elc_measured.json or elc_demo.json)")
    ap.add_argument("--nfft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=None)
    ap.add_argument("--max-boost", type=float, default=20.0)
    ap.add_argument("--max-cut", type=float, default=20.0)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="no per-file progress lines")
    args = ap.parse_args(argv)
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dirs()
    elc = args.elc or find_elc_file()
    presets = make_presets(elc, args.max_boost, args.max_cut)
    inputs = sorted(p for p in glob.glob(os.path.join(AUDIO_IN_DIR, "*")) if p.lower().endswith(AUDIO_EXT))
    if not inputs:
        print("  no input audio in audio/input/. Run make_test_audio.py first.")
        return
    if not args.quiet:
        print(f"  {len(inputs)} input files, {len(presets)} presets, ELC file: {elc and os.path.relpath(elc)}")

    # Figure 2: gain curves
    fig, ax = plt.subplots(figsize=(7, 4.3))
    cmap = plt.get_cmap("tab10")
    for i, (name, kw) in enumerate(presets.items()):
        d = design_gain(44100, args.nfft, **kw)
        ax.semilogx(d.bin_freqs[1:], d.gain_db[1:], color=cmap(i), lw=1.8, label=d.label)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlim(20, 20000)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain applied (dB)")
    ax.set_title("Equalizer gain curves: how much each frequency is boosted or cut", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG_DIR, "fig2_eq_gain_curves"))

    # Process every input with every preset
    rows = []
    bands = {}
    for in_path in inputs:
        stem = os.path.splitext(os.path.basename(in_path))[0]
        x, fs = load_audio(in_path, mono=True)
        fc, before = third_octave_levels(x, fs)
        bands[stem] = {"fc": fc, "input": before, "fs": fs}
        for pname, kw in presets.items():
            design = design_gain(fs, args.nfft, **kw)
            out = os.path.join(AUDIO_OUT_DIR, f"{stem}__{pname}.wav")
            info = equalize_file(in_path, out, design, hop=args.hop, norm="rms", mono=True, verbose=False)
            y, _ = load_audio(out, mono=True)
            _, after = third_octave_levels(y, fs)
            bands[stem][pname] = after
            ch = band_change(fc, before, after)
            rows.append((stem, pname, design.label, ch, info))
            if not args.quiet:
                print(f"  {stem:12s} {pname:22s} low {ch['low']:+5.1f}  mid {ch['mid']:+5.1f}  high {ch['high']:+5.1f} dB"
                      f"   {info['realtime_factor']:6.0f}x realtime" + ("  peak limited" if info["peak_limited"] else ""))

    # Summary table
    lines = ["| Input | Preset | Low (<250 Hz) | Mid (250 to 2k) | High (>2 kHz) | Speed (x real time) | Peak limited |",
             "|---|---|---:|---:|---:|---:|:---:|"]
    for stem, pname, label, ch, info in rows:
        lines.append(f"| {stem} | {label} | {ch['low']:+.1f} dB | {ch['mid']:+.1f} dB | {ch['high']:+.1f} dB | "
                     f"{info['realtime_factor']:.0f} | {'yes' if info['peak_limited'] else 'no'} |")
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(FIG_DIR, "eq_summary.md"), "w") as fh:
        fh.write(summary)
    if not args.quiet:
        print("\n  wrote figures/fig2 and figures/eq_summary.md")
    if not args.no_show:
        plt.show()
    return {"inputs": [os.path.basename(p) for p in inputs], "presets": [r[2] for r in rows[:len(presets)]],
            "n_outputs": len(rows), "nfft": args.nfft, "max_boost": args.max_boost}


def presets_label(presets, name, fs, nfft):
    return design_gain(fs, nfft, **presets[name]).label


if __name__ == "__main__":
    main()
