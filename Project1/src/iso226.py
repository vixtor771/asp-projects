"""ISO 226:2003 equal-loudness contours in analytic form.

The standard tabulates three parameters (alpha_f, L_U, T_f) at 29 frequencies from
20 Hz to 12.5 kHz and gives a closed-form expression for the sound pressure level
L_p (dB SPL) of a pure tone whose loudness level is L_N phon:

    A_f = 4.47e-3 * (10 ** (0.025 * L_N) - 1.15)
          + (0.4 * 10 ** ((T_f + L_U) / 10 - 9)) ** alpha_f
    L_p = (10 / alpha_f) * log10(A_f) - L_U + 94

The formula is specified for 20 to 90 phon (the 90 phon contour only up to 4 kHz).
The hearing threshold (0 phon) is the tabulated T_f.

Run this file directly to show the reference contours, or save them as a PNG:
    python src/iso226.py
    python src/iso226.py "figures/Standard ELC.png"
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

ISO226_FREQS = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500], dtype=float)

_ALPHA_F = np.array([
    0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, 0.301,
    0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, 0.243, 0.242,
    0.242, 0.245, 0.254, 0.271, 0.301])

_L_U = np.array([
    -31.6, -27.2, -23.0, -19.1, -16.1, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1, -2.0, -1.1,
    -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1])

_T_F = np.array([
    78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2, 4.4,
    3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3])


def interp_log_freq(f_known, y_known, f_query):
    """Shape-preserving (PCHIP) interpolation on a log-frequency axis.

    Queries outside [f_known[0], f_known[-1]] are clamped to the edge values, which also
    keeps the DC bin (0 Hz) safe.
    """
    f_known = np.asarray(f_known, dtype=float)
    y_known = np.asarray(y_known, dtype=float)
    f_query = np.atleast_1d(np.asarray(f_query, dtype=float))
    order = np.argsort(f_known)
    f_known, y_known = f_known[order], y_known[order]
    fq = np.clip(f_query, f_known[0], f_known[-1])
    if len(f_known) < 2:
        return np.full(fq.shape, y_known[0])
    return PchipInterpolator(np.log10(f_known), y_known)(np.log10(fq))


def iso226_spl(phon: float, freqs=None):
    """Sound pressure level (dB SPL) of the ISO 226:2003 contour at `phon`.

    Returns (freqs, spl). With freqs=None the 29 standard frequencies are used;
    otherwise the contour is interpolated to the requested frequencies.
    """
    phon = float(phon)
    if not 0.0 <= phon <= 90.0:
        raise ValueError("ISO 226 contours are defined for 0 to 90 phon")
    if phon == 0.0:
        spl = _T_F.copy()
    else:
        a_f = (4.47e-3 * (10 ** (0.025 * phon) - 1.15)
               + (0.4 * 10 ** ((_T_F + _L_U) / 10 - 9)) ** _ALPHA_F)
        spl = (10.0 / _ALPHA_F) * np.log10(np.maximum(a_f, 1e-12)) - _L_U + 94.0
    if freqs is None:
        return ISO226_FREQS.copy(), spl
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    return freqs, interp_log_freq(ISO226_FREQS, spl, freqs)


def iso226_threshold(freqs=None):
    return iso226_spl(0.0, freqs)


def plot_iso226_contours(path: str | None = None, phons=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90)):
    """Reference contours, each labelled with its phon value at its right end.
    Saves a 300 dpi PNG when `path` is given."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for p in phons:
        f, spl = iso226_spl(p)
        if p == 90:
            m = f <= 4000  # standard only specifies 90 phon up to 4 kHz
            f, spl = f[m], spl[m]
        ax.semilogx(f, spl, color="C3", lw=1.4)
        ax.annotate("threshold" if p == 0 else f"{p} phon", (f[-1], spl[-1]),
                    xytext=(4, 0), textcoords="offset points", fontsize=7.5, va="center")
    ax.set_xlim(20, 24000)
    ax.set_ylim(-10, 130)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Sound pressure level (dB SPL)")
    ax.set_title("ISO 226:2003 equal-loudness contours (pure tones, free field)", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    plot_iso226_contours(sys.argv[1] if len(sys.argv) > 1 else None)
    plt.show()
