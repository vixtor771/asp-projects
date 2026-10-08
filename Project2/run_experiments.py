"""Parameter experiments on all 12 clips, following Section III of the paper.

    python run_experiments.py                 # all five experiments (about 15 minutes)
    python run_experiments.py lookahead       # only one: scaling, window, overlap, lookahead or iterations

  scaling     Window scaling check: sum of the squared, overlap-added windows (scaled Hamming
              gives 1, plain Hamming about 1.59); true-phase STFT + overlap-add gives back the
              original; loudness change of RTISI-LA output with scaled vs plain Hamming.
  window      Window length L (8, 16, 23.2, 32, 64 ms), S = L/4.
              RTISI with 8 iterations and RTISI-LA with k = 3, 2 iterations per step.
  overlap     Table I: RTISI-LA (k = 3, 2 iterations per step) with S = L/4 or L/8,
              normal or asymmetric analysis window for the newest frame.
  lookahead   Table II: RTISI-LA with k = 0 to 10 look-ahead frames, 2 iterations per step.
  iterations  Tables III, IV and Fig. 5: SER and speed against the total number of
              iterations per frame (4 to 100); RTISI-LA uses k = 3, so it runs
              (total / 4) iterations per step. Also Table III's k = 10, 20 iterations per step.

Unless stated otherwise: L = 23.2 ms, S = L/4, asymmetric window on.
Output per experiment: a table of means per clip type printed in the terminal, and a figure
in figures/ (fig1 scaling, fig2 window, fig3 iterations; overlap and lookahead have no figure)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))   # all other code lives in src/

import numpy as np

from common import CLIP_NAMES, CLIP_TYPE, WINDOW_MS, ensure_dirs, load_clip, rms_db, save_fig, window_length
from evaluate import reconstruct
from plot_style import AQUA, BLUE, GRAY, METHOD_COLOR, METHOD_LABEL, ORANGE, plt
from stft import hamming, istft, pad_signal, scaled_hamming, stft, window_square_sum

TYPES = ("speech", "music")
CLIPS = {name: load_clip(name) for name in CLIP_NAMES}


def run_setting(setting: dict) -> list[dict]:
    """Run one setting on every clip; returns one row per clip."""
    rows = []
    for name, (x, fs) in CLIPS.items():
        args = {k: v for k, v in setting.items() if k != "label"}
        result = reconstruct(x, fs, **args)
        rows.append({
            "label": setting.get("label", ""),
            "clip": name,
            "type": CLIP_TYPE[name],
            "method": setting["method"],
            "window_ms": setting["window_ms"],
            "L": result["L"],
            "S": result["S"],
            "hop_divisor": setting["hop_divisor"],
            "n_iter": setting["n_iter"],
            "look_ahead": setting.get("look_ahead", 0),
            "asymmetric": setting.get("asymmetric", True) if setting["method"] == "rtisi_la" else "",
            "total_iter_per_frame": result["total_iter_per_frame"],
            "ser_db": round(result["ser_db"], 3),
            "loudness_diff_db": round(result["loudness_diff_db"], 3),
            "runtime_s": round(result["runtime_s"], 3),
            "rtf": round(result["rtf"], 4),
        })
    return rows


def run_experiment(name: str, settings: list[dict]) -> list[dict]:
    """Run all settings and print a table of means per clip type."""
    rows = []
    for i, setting in enumerate(settings, 1):
        print(f"  [{name}] {i}/{len(settings)} {setting['label']}", flush=True)
        rows += run_setting(setting)

    lines = ["| Setting | Speech SER (dB) | Music SER (dB) | Speech loudness change (dB) | "
             "Music loudness change (dB) | Real-time factor |",
             "|---|---|---|---|---|---|"]
    for setting in settings:
        sel = [r for r in rows if r["label"] == setting["label"]]
        ser = {t: mean(sel, "ser_db", t) for t in TYPES}
        loud = {t: mean(sel, "loudness_diff_db", t) for t in TYPES}
        rtf = np.sum([r["runtime_s"] for r in sel]) / np.sum([r["runtime_s"] / r["rtf"] for r in sel])
        lines.append(f"| {setting['label']} | {ser['speech']:.2f} | {ser['music']:.2f} | "
                     f"{loud['speech']:+.2f} | {loud['music']:+.2f} | {rtf:.3f} |")
    print("\n".join(lines))
    return rows


def mean(rows: list[dict], key: str, clip_type: str | None = None) -> float:
    return float(np.mean([r[key] for r in rows if clip_type is None or r["type"] == clip_type]))


def overall_rtf(rows: list[dict]) -> float:
    return float(np.sum([r["runtime_s"] for r in rows]) / np.sum([r["runtime_s"] / r["rtf"] for r in rows]))


# Experiments

def exp_scaling() -> None:
    L = 1024
    sums = {
        "Hamming, S = L/4": window_square_sum(hamming(L), L // 4, n_frames=16),
        "Scaled Hamming, S = L/4": window_square_sum(scaled_hamming(L, L // 4), L // 4, n_frames=16),
        "Scaled Hamming, S = L/8": window_square_sum(scaled_hamming(L, L // 8), L // 8, n_frames=29),
    }
    for label, s in sums.items():
        middle = s[L:-L]
        print(f"{label:26s} sum of w^2 in the fully overlapped part: {middle.min():.4f} to {middle.max():.4f}")

    rows = []
    for name, (x, fs) in CLIPS.items():
        Lc = window_length(fs)
        S = Lc // 4
        w = scaled_hamming(Lc, S)
        x_pad = pad_signal(x, Lc, S)
        y_pad = istft(stft(x_pad, w, S), w, S)          # true phase kept
        scaled = reconstruct(x, fs, "rtisi_la", WINDOW_MS, 4, n_iter=2, look_ahead=3)
        plain = reconstruct(x, fs, "rtisi_la", WINDOW_MS, 4, n_iter=2, look_ahead=3, window=hamming(Lc))
        rows.append({"clip": name, "scaled": scaled["loudness_diff_db"], "plain": plain["loudness_diff_db"]})
        print(f"{name:8s} true phase error {np.max(np.abs(y_pad - x_pad)):.1e} "
              f"(loudness {rms_db(y_pad) - rms_db(x_pad):+.2f} dB)   RTISI-LA loudness change: "
              f"scaled {scaled['loudness_diff_db']:+.2f} dB, plain {plain['loudness_diff_db']:+.2f} dB")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1, 1.4]})
    for (label, s), color in zip(sums.items(), (GRAY, BLUE, AQUA)):
        ax1.plot(np.arange(len(s)) / L, s, color=color, label=label)
    ax1.set_xlabel("Time (in window lengths)")
    ax1.set_ylabel("Sum of squared windows")
    ax1.set_ylim(0, 1.8)
    ax1.set_title("(a) Overlap-added squared windows")
    ax1.legend(loc="lower center")

    pos = np.arange(len(rows))
    ax2.bar(pos - 0.2, [r["plain"] for r in rows], 0.38, color=GRAY, label="Hamming")
    ax2.bar(pos + 0.2, [r["scaled"] for r in rows], 0.38, color=ORANGE, label="Scaled Hamming")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(pos, [r["clip"].replace("audio", "") for r in rows])
    ax2.set_xlabel("Clip number")
    ax2.set_ylabel("Output minus input level (dB)")
    ax2.set_title("(b) Loudness change after RTISI-LA")
    ax2.set_ylim(-0.5, 5)
    ax2.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    save_fig(fig, "fig1_window_scaling")


def exp_window() -> None:
    window_list = [8, 16, WINDOW_MS, 32, 64]
    settings = []
    for ms in window_list:
        settings.append({"label": f"RTISI, L = {ms} ms", "method": "rtisi", "window_ms": ms,
                         "hop_divisor": 4, "n_iter": 8})
        settings.append({"label": f"RTISI-LA, L = {ms} ms", "method": "rtisi_la", "window_ms": ms,
                         "hop_divisor": 4, "n_iter": 2, "look_ahead": 3})
    rows = run_experiment("window", settings)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, t in zip(axes, TYPES):
        for method in ("rtisi", "rtisi_la"):
            ser = [mean([r for r in rows if r["method"] == method and r["window_ms"] == ms], "ser_db", t)
                   for ms in window_list]
            ax.plot(window_list, ser, "o-", color=METHOD_COLOR[method], label=METHOD_LABEL[method])
        ax.set_xscale("log")
        ax.set_xticks(window_list, [f"{ms:g}" for ms in window_list])
        ax.minorticks_off()
        ax.set_xlabel("Window length L (ms)")
        ax.set_title(f"{t.capitalize()} (mean of {sum(CLIP_TYPE[c] == t for c in CLIPS)} clips)")
    axes[0].set_ylabel("SER (dB)")
    axes[0].legend(loc="lower center")
    fig.suptitle("SER against window length (S = L/4, 8 iterations per frame)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig2_window_length")


def exp_overlap() -> None:
    settings = []
    for asym in (False, True):
        for hop in (8, 4):
            name = "asymmetric window" if asym else "normal window"
            settings.append({"label": f"{name}, S = L/{hop}", "method": "rtisi_la", "window_ms": WINDOW_MS,
                             "hop_divisor": hop, "n_iter": 2, "look_ahead": 3, "asymmetric": asym})
    run_experiment("overlap", settings)


def exp_lookahead() -> None:
    k_list = list(range(11))
    settings = [{"label": f"k = {k}", "method": "rtisi_la", "window_ms": WINDOW_MS, "hop_divisor": 4,
                 "n_iter": 2, "look_ahead": k} for k in k_list]
    run_experiment("lookahead", settings)


def exp_iterations() -> None:
    total_list = [4, 8, 12, 20, 40, 60, 80, 100]
    settings = []
    for total in total_list:
        settings.append({"label": f"RTISI, {total} iterations", "method": "rtisi", "window_ms": WINDOW_MS,
                         "hop_divisor": 4, "n_iter": total})
        settings.append({"label": f"RTISI-LA k = 3, {total // 4} per step ({total} total)", "method": "rtisi_la",
                         "window_ms": WINDOW_MS, "hop_divisor": 4, "n_iter": total // 4, "look_ahead": 3})
    settings.append({"label": "RTISI-LA k = 10, 20 per step (220 total)", "method": "rtisi_la",
                     "window_ms": WINDOW_MS, "hop_divisor": 4, "n_iter": 20, "look_ahead": 10})
    rows = run_experiment("iterations", settings)
    rows = [r for r in rows if r["look_ahead"] in (0, 3)]   # the k = 10 row is only in the table

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, t in zip(axes[:2], TYPES):
        for method in ("rtisi", "rtisi_la"):
            ser = [mean([r for r in rows if r["method"] == method and r["total_iter_per_frame"] == n], "ser_db", t)
                   for n in total_list]
            ax.plot(total_list, ser, "o-", color=METHOD_COLOR[method], label=METHOD_LABEL[method])
        ax.set_xlabel("Iterations per frame")
        ax.set_ylabel("SER (dB)")
        ax.set_title(f"({'ab'[TYPES.index(t)]}) {t.capitalize()}")
    axes[0].legend(loc="center right")
    for method in ("rtisi", "rtisi_la"):
        rtf = [overall_rtf([r for r in rows if r["method"] == method and r["total_iter_per_frame"] == n])
               for n in total_list]
        axes[2].plot(total_list, rtf, "o-", color=METHOD_COLOR[method], label=METHOD_LABEL[method])
    axes[2].axhline(1, color=AQUA, linewidth=1, linestyle="--")
    axes[2].text(total_list[0], 1, " real time", color=AQUA, va="bottom", fontsize=8)
    axes[2].set_xlabel("Iterations per frame")
    axes[2].set_ylabel("Runtime / audio duration")
    axes[2].set_title("(c) Speed (all 12 clips)")
    axes[2].set_ylim(bottom=0)
    fig.suptitle("SER and speed against iterations per frame (RTISI-LA: k = 3, S = L/4)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig3_iterations")


EXPERIMENTS = {"scaling": exp_scaling, "window": exp_window, "overlap": exp_overlap, "lookahead": exp_lookahead,
               "iterations": exp_iterations}


def main() -> None:
    ensure_dirs()
    names = sys.argv[1:] or list(EXPERIMENTS)
    for name in names:
        if name not in EXPERIMENTS:
            sys.exit(f"Unknown experiment {name!r}; choose from {', '.join(EXPERIMENTS)}")
        print(f"\n=== {name} ===")
        EXPERIMENTS[name]()


if __name__ == "__main__":
    main()
