"""Final run: rebuild all 12 clips from their STFT magnitude with RTISI and RTISI-LA. Run from main.py.

Settings come from the experiments in experiments.py (see FINAL below).
Output:
  audio/output/<clip>_rtisi.wav, audio/output/<clip>_rtisi_la.wav   reconstructed audio (mono)
  terminal                                                            SER, loudness change, speed
  figures/fig4_final_ser.png/.pdf                                     SER of every clip
"""
from __future__ import annotations

import numpy as np

from common import CLIP_NAMES, CLIP_TYPE, ensure_dirs, load_clip, save_fig, save_wav
from evaluate import reconstruct
from plot_style import METHOD_COLOR, METHOD_LABEL, plt

FINAL = {
    "rtisi": {"method": "rtisi", "window_ms": 32, "hop_divisor": 4, "n_iter": 20},
    "rtisi_la": {"method": "rtisi_la", "window_ms": 32, "hop_divisor": 4, "n_iter": 5, "look_ahead": 3,
                 "asymmetric": True},
}


def run() -> None:
    ensure_dirs()
    rows = []
    for name in CLIP_NAMES:
        x, fs = load_clip(name)
        for key, setting in FINAL.items():
            result = reconstruct(x, fs, **setting)
            save_wav(f"{name}_{key}", result["y"], fs)
            rows.append({
                "clip": name,
                "type": CLIP_TYPE[name],
                "method": key,
                "fs": fs,
                "duration_s": round(len(x) / fs, 2),
                "L": result["L"],
                "S": result["S"],
                "total_iter_per_frame": result["total_iter_per_frame"],
                "ser_db": round(result["ser_db"], 3),
                "loudness_diff_db": round(result["loudness_diff_db"], 3),
                "runtime_s": round(result["runtime_s"], 3),
                "rtf": round(result["rtf"], 4),
            })
            print(f"{name:8s} {METHOD_LABEL[key]:9s} SER {result['ser_db']:6.2f} dB   loudness "
                  f"{result['loudness_diff_db']:+.2f} dB   {result['runtime_s']:.2f} s (RTF {result['rtf']:.3f})")

    print_summary(rows)
    plot_ser(rows)
    print("Saved audio/output/*.wav, figures/fig4")


def print_summary(rows: list[dict]) -> None:
    lines = ["| Method | Clip type | Mean SER (dB) | Min SER (dB) | Max SER (dB) | Mean loudness change (dB) | "
             "Real-time factor |",
             "|---|---|---|---|---|---|---|"]
    for key in FINAL:
        for t in ("speech", "music"):
            sel = [r for r in rows if r["method"] == key and r["type"] == t]
            ser = [r["ser_db"] for r in sel]
            rtf = sum(r["runtime_s"] for r in sel) / sum(r["duration_s"] for r in sel)
            lines.append(f"| {METHOD_LABEL[key]} | {t} | {np.mean(ser):.2f} | {np.min(ser):.2f} | {np.max(ser):.2f} | "
                         f"{np.mean([r['loudness_diff_db'] for r in sel]):+.2f} | {rtf:.3f} |")
    print("\n".join(lines))


def plot_ser(rows: list[dict]) -> None:
    order = [c for c in CLIP_NAMES if CLIP_TYPE[c] == "speech"] + [c for c in CLIP_NAMES if CLIP_TYPE[c] == "music"]
    pos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 3))
    for i, key in enumerate(FINAL):
        ser = [next(r["ser_db"] for r in rows if r["clip"] == c and r["method"] == key) for c in order]
        ax.bar(pos + (i - 0.5) * 0.38, ser, 0.36, color=METHOD_COLOR[key], label=METHOD_LABEL[key])
    n_speech = sum(CLIP_TYPE[c] == "speech" for c in order)
    ax.axvline(n_speech - 0.5, color="black", linewidth=0.8)
    ax.text(n_speech / 2 - 0.5, 1.0, "Speech", ha="center", transform=ax.get_xaxis_transform())
    ax.text(n_speech + (len(order) - n_speech) / 2 - 0.5, 1.0, "Music", ha="center",
            transform=ax.get_xaxis_transform())
    ax.set_xticks(pos, [c.replace("audio", "") for c in order])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Clip number")
    ax.set_ylabel("SER (dB)")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    save_fig(fig, "fig4_final_ser")
