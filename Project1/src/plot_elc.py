"""Plot the measured equal-loudness curves and write them as a markdown table.

    python src/plot_elc.py                                   # newest data/elc_measured*.json
    python src/plot_elc.py --elc data/elc_measured_20260921_1530.json

Writes figures/fig1_<name>.png/.pdf and a markdown table figures/<name>_table.md.
The figure also shows ISO 226 contours (dashed) for comparison, one per measured curve,
shifted to pass through the measured 1 kHz point. Which contour: the measured 1 kHz
threshold is taken to be the ISO threshold at 1 kHz (2.4 dB SPL), which turns every
1 kHz reference level into a loudness level in phon (or --spl-offset, if given).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np

from iso226 import iso226_spl
from utils import DATA_DIR, FIG_DIR, ensure_dirs, list_measurements, order_contours, save_figure

ISO_THRESHOLD_1K = 2.4   # dB SPL, ISO 226 hearing threshold at 1 kHz


def find_elc_file() -> str | None:
    """Newest measurement in data/, or the demo file if there is none."""
    files = list_measurements()
    if files:
        return files[-1]
    demo = os.path.join(DATA_DIR, "elc_demo.json")
    return demo if os.path.exists(demo) else None


def _curve_values(c):
    return np.asarray([np.nan if v is None else v for v in c["dbfs"]], dtype=float)


def _fmt_freq(f: float) -> str:
    return f"{f / 1000:g}k" if f >= 1000 else f"{f:g}"


def iso_overlay(freqs, values, curves, spl_offset=None):
    """ISO 226 contour to compare with each measured curve: {name: (f, level, phon)}.

    Each contour is shifted so that it passes through the measured 1 kHz point. Its phon
    value is ref_dbfs + offset, where offset = dB SPL of 0 dBFS: either --spl-offset or,
    if not given, the value that puts the measured 1 kHz threshold at the ISO threshold.
    Returns {} when neither is available.
    """
    freqs = np.asarray(freqs, dtype=float)
    if 1000.0 not in freqs:
        return {}
    i1k = int(np.where(freqs == 1000.0)[0][0])
    offset = spl_offset
    if offset is None:
        thr = values.get("threshold")
        if thr is None or not np.isfinite(thr[i1k]):
            return {}
        offset = ISO_THRESHOLD_1K - thr[i1k]
    f_dense = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), 200)
    out = {}
    for name, c in curves.items():
        y1k = values[name][i1k]
        if not np.isfinite(y1k):
            continue
        phon = 0.0 if c.get("ref_dbfs") is None else float(np.clip(c["ref_dbfs"] + offset, 0.0, 90.0))
        iso = iso226_spl(phon, f_dense)[1]
        iso_1k = iso226_spl(phon, [1000.0])[1][0]
        out[name] = (f_dense, iso - iso_1k + y1k, phon)
    return out


def plot_measured(path: str, out_no_ext: str):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with open(path) as fh:
        d = json.load(fh)
    freqs = np.asarray(d["freqs"], dtype=float)
    curves = d["curves"]
    values, changed = order_contours(curves)
    offset = d.get("spl_offset")
    sim = " (simulated listener)" if d.get("simulated") else ""
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(curves)))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for (name, c), col in zip(curves.items(), colors):
        y = values[name] + (offset or 0.0)
        label = name.replace("_", " ")
        if c.get("ref_dbfs") is not None:
            label += f" (1 kHz at {c['ref_dbfs']:.0f} dBFS)"
        ax.semilogx(freqs, y, "o-", color=col, lw=1.7, ms=4.5, label=label)
    overlay = iso_overlay(freqs, values, curves, offset)
    for (name, c), col in zip(curves.items(), colors):
        if name in overlay:
            fd, lv, phon = overlay[name]
            ax.semilogx(fd, lv + (offset or 0.0), "--", color=col, lw=1.1, alpha=0.85)
            ax.annotate(f"{phon:.0f} phon", (fd[-1], lv[-1] + (offset or 0.0)), xytext=(3, 0),
                        textcoords="offset points", fontsize=7, color=col, va="center")
    handles, labels = ax.get_legend_handles_labels()
    if overlay:
        handles.append(Line2D([], [], color="0.3", ls="--", lw=1.1))
        labels.append("ISO 226 contour, matched at 1 kHz")
    ax.set_xlim(freqs.min() * 0.85, freqs.max() * 1.45)
    ax.set_xticks(freqs)
    ax.set_xticklabels([_fmt_freq(f) for f in freqs], fontsize=8)
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Level (dB SPL, approximate)" if offset is not None else "Level (dBFS)")
    ax.set_title("Measured equal-loudness curves" + sim, fontsize=10)
    ax.grid(True, which="major", alpha=0.3)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.2 * (hi - lo), hi)           # room for the legend below the curves
    ax.legend(handles, labels, fontsize=7.5, loc="lower left", ncol=2, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, out_no_ext)
    return fig


def write_table(path: str, out_md: str) -> str:
    with open(path) as fh:
        d = json.load(fh)
    freqs = d["freqs"]
    names = list(d["curves"])
    values, changed = order_contours(d["curves"])
    lines = ["Levels in dBFS (0 dBFS = digital full scale).", "",
             "| Frequency (Hz) | " + " | ".join(n.replace("_", " ") for n in names) + " |",
             "|---:|" + "---:|" * len(names)]
    for i, f in enumerate(freqs):
        row = []
        for n in names:
            v = values[n][i]
            row.append("" if not np.isfinite(v) else f"{v:.1f}")
        lines.append(f"| {f:.0f} | " + " | ".join(row) + " |")
    extra = []
    for n in names:
        rc = d["curves"][n].get("ref_check_db")
        if rc is not None:
            extra.append(f"{n.replace('_', ' ')}: 1 kHz self-match {rc:+.1f} dB")
    thr = d["curves"].get("threshold")
    if thr:
        extra.append(f"threshold: {thr['catch_trials']} catch trials, {thr['false_alarms']} false alarms")
    text = "\n".join(lines) + ("\n\n" + "; ".join(extra) if extra else "") + "\n"
    with open(out_md, "w") as fh:
        fh.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elc", default=None, help="JSON file from elc_measure.py")
    ap.add_argument("--no-show", action="store_true", help="do not open figure windows")
    args = ap.parse_args(argv)
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dirs()
    elc = args.elc or find_elc_file()
    if elc is None:
        print("  no measurement file found in data/. Run elc_measure.py first.")
    else:
        stem = os.path.splitext(os.path.basename(elc))[0]
        fig = os.path.join(FIG_DIR, f"fig1_{stem}")
        plot_measured(elc, fig)
        print(f"  wrote {os.path.relpath(fig)}.png/.pdf from {os.path.relpath(elc)}")
        table_path = os.path.join(FIG_DIR, f"{stem}_table.md")
        table = write_table(elc, table_path)
        print(f"  wrote {os.path.relpath(table_path)}\n")
        print(table)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
