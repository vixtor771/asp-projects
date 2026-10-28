"""
ECE 520.445/645 Audio Signal Processing - Project 3: Head-Related Transfer Functions

Requirements: numpy, scipy, matplotlib, h5py, soundfile
Run:          python project3.py
Outputs:      figures/       plots for Task 2
              output_audio/  audio for Task 2c and Task 4
"""
import os
import glob
from math import gcd

import numpy as np
import h5py
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, resample_poly, get_window

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HRTF_DIR = os.path.join(BASE_DIR, "Data", "HRTFs")
AUDIO_DIR = os.path.join(BASE_DIR, "Data", "Audio")
FIG_DIR = os.path.join(BASE_DIR, "figures")
OUT_DIR = os.path.join(BASE_DIR, "output_audio")

LEFT, RIGHT = 0, 1  # SOFA receiver index: 0 = left ear (+y), 1 = right ear (-y)


# =============================================================================
# Task 2a: load SOFA files and HRIRs
# =============================================================================
def load_sofa(path):
    """Read the HRIRs and source directions directly from a SOFA (HDF5) file."""
    with h5py.File(path, "r") as f:
        hrirs = f["Data.IR"][:]          # (M directions, 2 ears, N taps)
        pos = f["SourcePosition"][:]     # (M, 3): azimuth [deg], elevation [deg], distance [m]
        fs = float(f["Data.SamplingRate"][0])
    name = os.path.basename(path).split("_")[0]
    return {"name": name, "hrirs": hrirs, "az": pos[:, 0] % 360, "el": pos[:, 1], "fs": fs}


def find_index(sofa, az, el):
    """Index of the measured direction closest to (az, el)."""
    d_az = (sofa["az"] - az + 180) % 360 - 180
    return int(np.argmin(d_az ** 2 + (sofa["el"] - el) ** 2))


def get_hrir(sofa, az, el, ear):
    """HRIR of one ear for the measured direction closest to (az, el)."""
    return sofa["hrirs"][find_index(sofa, az, el), ear]


def explore_sofa(sofa):
    """Print the structure of the azimuth / elevation indices."""
    az = np.unique(np.round(sofa["az"], 1))
    el = np.unique(np.round(sofa["el"], 1))
    M, R, N = sofa["hrirs"].shape
    print(f"[{sofa['name']}] Data.IR shape = {M} directions x {R} ears x {N} taps, fs = {sofa['fs']:.0f} Hz")
    print(f"  {len(az)} azimuths: {az.min():.0f} ... {az.max():.0f} deg, step {az[1] - az[0]:.0f} deg")
    print(f"  {len(el)} elevations (deg): {el}")
    print("  Index layout (first 20 directions):")
    for i in range(20):
        print(f"    index {i:4d}: az = {sofa['az'][i]:6.1f}, el = {sofa['el'][i]:6.2f}")
    print(f"  -> index = azimuth_index * {len(el)} + elevation_index")


# =============================================================================
# Task 2b: HRIR <-> HRTF
# =============================================================================
def hrir_to_hrtf(hrir, nfft=None):
    return np.fft.rfft(hrir, n=nfft, axis=-1)


def hrtf_to_hrir(hrtf, n):
    return np.fft.irfft(hrtf, n=n, axis=-1)


# =============================================================================
# Task 2c: apply an HRIR filter to an audio clip
# =============================================================================
def load_audio(path):
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


def resample_hrir(hrir, fs_in, fs_out):
    """HRIRs are 44.1 kHz, the audio is 48 kHz: resample the HRIRs to the audio rate."""
    if fs_in == fs_out:
        return hrir
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(hrir, int(fs_out) // g, int(fs_in) // g, axis=-1)


def apply_hrir(x, hrir_left, hrir_right):
    """Filter a mono signal with left / right HRIRs -> stereo (n, 2)."""
    return np.stack([fftconvolve(x, hrir_left), fftconvolve(x, hrir_right)], axis=1)


# =============================================================================
# Task 2d: azimuth / elevation magnitude maps
# =============================================================================
def azimuth_map(sofa, el=0.0, ear=LEFT):
    """|HRTF| at fixed elevation. Returns azimuths (A,), freqs (F,), magnitude (F, A)."""
    el_used = sofa["el"][find_index(sofa, 0, el)]
    idx = np.where(np.abs(sofa["el"] - el_used) < 1)[0]
    idx = idx[np.argsort(sofa["az"][idx])]
    mag = np.abs(hrir_to_hrtf(sofa["hrirs"][idx, ear])).T
    freqs = np.fft.rfftfreq(sofa["hrirs"].shape[-1], 1 / sofa["fs"])
    return sofa["az"][idx], freqs, mag


def elevation_map(sofa, az=0.0, ear=LEFT):
    """|HRTF| at fixed azimuth. Returns elevations (E,), freqs (F,), magnitude (F, E)."""
    az_used = sofa["az"][find_index(sofa, az, 0)]
    d_az = (sofa["az"] - az_used + 180) % 360 - 180
    idx = np.where(np.abs(d_az) < 1)[0]
    idx = idx[np.argsort(sofa["el"][idx])]
    mag = np.abs(hrir_to_hrtf(sofa["hrirs"][idx, ear])).T
    freqs = np.fft.rfftfreq(sofa["hrirs"].shape[-1], 1 / sofa["fs"])
    return sofa["el"][idx], freqs, mag


def to_db(mag):
    return 20 * np.log10(np.maximum(mag, 1e-12))


def plot_heatmap(ax, x, freqs, mag, vmin, vmax):
    # skip the DC bin so the frequency axis can be log scaled
    im = ax.pcolormesh(x, freqs[1:], to_db(mag[1:]), shading="nearest",
                       cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_yscale("log")
    return im


def plot_heatmaps(az_maps, el_maps, path):
    """Azimuth heatmap (left) and elevation heatmap (right) side by side, one row per subject.
    az_maps / el_maps: lists of (name, x, freqs, mag)."""
    all_db = np.concatenate([to_db(m[3][1:]).ravel() for m in az_maps + el_maps])
    vmax = np.ceil(all_db.max())
    vmin = vmax - 60
    fig, axes = plt.subplots(len(az_maps), 2, figsize=(10, 1.8 * len(az_maps)), sharey=True)
    for row, (a, e) in enumerate(zip(az_maps, el_maps)):
        plot_heatmap(axes[row, 0], a[1], a[2], a[3], vmin, vmax)
        im = plot_heatmap(axes[row, 1], e[1], e[2], e[3], vmin, vmax)
        axes[row, 0].set_ylabel(f"{a[0]}\nFrequency (Hz)")
    axes[0, 0].set_title("Azimuth heatmap (elevation = 0 deg)")
    axes[0, 1].set_title("Elevation heatmap (azimuth = 0 deg)")
    axes[-1, 0].set_xlabel("Azimuth (deg)")
    axes[-1, 1].set_xlabel("Elevation (deg)")
    fig.colorbar(im, ax=axes, label="Magnitude (dB)", shrink=0.6)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Task 2f: mean HRTF and MSE
# =============================================================================
def mean_and_mse(mags):
    """mags: (S, F, D) linear magnitudes. Mean magnitude over subjects and per-subject
    MSE (in dB, DC bin excluded) between each subject and the mean."""
    mean_mag = mags.mean(axis=0)
    mse = np.mean((to_db(mags[:, 1:]) - to_db(mean_mag[1:])) ** 2, axis=(1, 2))
    return mean_mag, mse


# =============================================================================
# Task 3: surround sound tool (moving source, rotating azimuth, fixed elevation)
# =============================================================================
def spatialize(sofa_path, audio_path, az_start, omega, el=0.0, frame_len=1024, out_path=None):
    """Render a mono audio sample as a source rotating around the listener.

    sofa_path  : HRTF configuration file (one subject)
    audio_path : mono audio sample
    az_start   : starting azimuth in degrees (0 = front, 90 = left, SOFA convention)
    omega      : azimuth rotation speed in degrees per second (positive = counter-clockwise)
    el         : fixed elevation in degrees
    frame_len  : frame length in samples; hop = frame_len / 2

    Each frame is Hann-windowed (periodic Hann at 50% overlap sums to 1), filtered with the
    HRIR pair of the azimuth at the frame's center time, and overlap-added into the output.
    Returns (stereo signal (n, 2), fs).
    """
    sofa = load_sofa(sofa_path)
    x, fs = load_audio(audio_path)
    hrirs = resample_hrir(sofa["hrirs"], sofa["fs"], fs)
    n_ir = hrirs.shape[-1]

    hop = frame_len // 2
    win = get_window("hann", frame_len)
    L = len(x)
    n_frames = int(np.ceil((L + hop) / hop))
    x_pad = np.zeros(n_frames * hop + frame_len)
    x_pad[hop:hop + L] = x  # leading hop of zeros so the first samples are covered by two frames

    y = np.zeros((len(x_pad) + n_ir - 1, 2))
    for k in range(n_frames):
        start = k * hop
        seg = x_pad[start:start + frame_len] * win
        t = (start + frame_len / 2 - hop) / fs
        i = find_index(sofa, (az_start + omega * t) % 360, el)
        for ear in (LEFT, RIGHT):
            y[start:start + frame_len + n_ir - 1, ear] += fftconvolve(seg, hrirs[i, ear])

    y = y[hop:hop + L + n_ir - 1]
    y *= 0.99 / np.max(np.abs(y))  # normalize to avoid clipping
    if out_path:
        sf.write(out_path, y, fs)
    return y, fs


# =============================================================================
# Main: run all tasks
# =============================================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    sofa_paths = sorted(glob.glob(os.path.join(HRTF_DIR, "*.sofa")))
    subjects = [load_sofa(p) for p in sofa_paths]
    names = [s["name"] for s in subjects]

    # ---- 2a: structure + HRIR of one direction / ear ----
    s0 = subjects[0]
    explore_sofa(s0)
    az_demo, el_demo, ear_demo = 90, 0, LEFT
    i = find_index(s0, az_demo, el_demo)
    hrir = get_hrir(s0, az_demo, el_demo, ear_demo)
    print(f"\nHRIR for {s0['name']}, left ear, requested az={az_demo}, el={el_demo} "
          f"-> index {i} (az={s0['az'][i]:.1f}, el={s0['el'][i]:.2f})")

    # ---- 2b: HRIR -> HRTF -> HRIR ----
    hrtf = hrir_to_hrtf(hrir)
    hrir_rec = hrtf_to_hrir(hrtf, len(hrir))
    print(f"Max |HRIR - IFFT(FFT(HRIR))| = {np.max(np.abs(hrir - hrir_rec)):.2e}")
    freqs = np.fft.rfftfreq(len(hrir), 1 / s0["fs"])
    t_ms = np.arange(len(hrir)) / s0["fs"] * 1000
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_ms, hrir, label="HRIR (from SOFA)")
    axes[0].plot(t_ms, hrir_rec, "--", label="IFFT(HRTF)")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"HRIR, {s0['name']}, left ear, az={s0['az'][i]:.0f}, el={s0['el'][i]:.1f}")
    axes[0].legend()
    axes[1].semilogx(freqs[1:], to_db(np.abs(hrtf[1:])))
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_title("HRTF = FFT(HRIR)")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "2ab_hrir_hrtf.png"), dpi=120)
    plt.close(fig)

    # ---- 2c: apply HRIR filter to an audio clip ----
    x, fs = load_audio(os.path.join(AUDIO_DIR, "speech.wav"))
    i = find_index(s0, az_demo, el_demo)
    h = resample_hrir(s0["hrirs"][i], s0["fs"], fs)
    y = apply_hrir(x, h[LEFT], h[RIGHT])
    y *= 0.99 / np.max(np.abs(y))
    sf.write(os.path.join(OUT_DIR, f"2c_{s0['name']}_speech_az{az_demo}_el{el_demo}.wav"), y, fs)
    fig, axes = plt.subplots(3, 1, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, sig, label in zip(axes, [x / np.max(np.abs(x)), y[:, LEFT], y[:, RIGHT]],
                              ["Input (mono)", "Output, left ear", "Output, right ear"]):
        ax.plot(np.arange(len(sig)) / fs, sig, linewidth=0.5)
        ax.set_ylabel(label)
    axes[0].set_title(f"Speech filtered with the HRIRs of {s0['name']}, az={az_demo}, el={el_demo}")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "2c_apply_hrir.png"), dpi=150)
    plt.close(fig)

    # ---- 2d / 2f: azimuth / elevation maps for all subjects (left ear) ----
    az_maps = [(s["name"],) + azimuth_map(s, el=0, ear=LEFT) for s in subjects]
    el_maps = [(s["name"],) + elevation_map(s, az=0, ear=LEFT) for s in subjects]

    # ---- 2f: mean HRTF + MSE ----
    az_x, freqs, _ = az_maps[0][1:]
    el_x = el_maps[0][1]
    mean_az, mse_az = mean_and_mse(np.stack([m[3] for m in az_maps]))
    mean_el, mse_el = mean_and_mse(np.stack([m[3] for m in el_maps]))

    vmax = np.ceil(max(to_db(mean_az[1:]).max(), to_db(mean_el[1:]).max()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_heatmap(axes[0], az_x, freqs, mean_az, vmax - 60, vmax)
    axes[0].set_title("Mean HRTF over subjects - azimuth (el = 0 deg)")
    axes[0].set_xlabel("Azimuth (deg)")
    axes[0].set_ylabel("Frequency (Hz)")
    im = plot_heatmap(axes[1], el_x, freqs, mean_el, vmax - 60, vmax)
    axes[1].set_title("Mean HRTF over subjects - elevation (az = 0 deg)")
    axes[1].set_xlabel("Elevation (deg)")
    fig.colorbar(im, ax=axes, label="Magnitude (dB)")
    fig.savefig(os.path.join(FIG_DIR, "2f_mean_hrtf.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\nMSE (dB^2) between each subject and the mean HRTF:")
    print(f"  {'subject':8s} {'azimuth':>10s} {'elevation':>10s}")
    for n, a, e in zip(names, mse_az, mse_el):
        print(f"  {n:8s} {a:10.2f} {e:10.2f}")
    print(f"  Azimuth:   closest = {names[np.argmin(mse_az)]}, farthest = {names[np.argmax(mse_az)]}")
    print(f"  Elevation: closest = {names[np.argmin(mse_el)]}, farthest = {names[np.argmax(mse_el)]}")

    # ---- 4: apply the surround tool to the audio samples, 5 subjects ----
    # closest / farthest are taken along azimuth, since the tool rotates azimuth
    closest = int(np.argmin(mse_az))
    farthest = int(np.argmax(mse_az))
    rest = [k for k in range(len(subjects)) if k not in (closest, farthest)]
    rng = np.random.default_rng(0)
    random3 = sorted(rng.choice(rest, size=3, replace=False).tolist())
    test_subjects = [closest, farthest] + random3
    print(f"\nTask 4 subjects: closest = {names[closest]}, farthest = {names[farthest]}, "
          f"random = {[names[k] for k in random3]}")

    # ---- 2d: heatmaps of the tested subjects, azimuth and elevation side by side ----
    plot_heatmaps([az_maps[k] for k in test_subjects], [el_maps[k] for k in test_subjects],
                  os.path.join(FIG_DIR, "2d_heatmaps.png"))

    az_start, omega = 0, 60  # start in front, 60 deg/s (one revolution every 6 s)

    # ---- 3: example output of the surround tool (left / right level over time) ----
    y, fs = spatialize(sofa_paths[closest], os.path.join(AUDIO_DIR, "speech.wav"), az_start, omega)
    win = int(0.05 * fs)
    n = len(y) // win
    level = 10 * np.log10(np.mean(y[:n * win].reshape(n, win, 2) ** 2, axis=1) + 1e-12)
    t = (np.arange(n) + 0.5) * win / fs
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    axes[0].plot(t, (az_start + omega * t) % 360, ".", markersize=2)
    axes[0].set_ylabel("Azimuth (deg)")
    axes[0].set_yticks([0, 90, 180, 270, 360])
    axes[0].set_title(f"Surround tool output: {names[closest]}, speech, start {az_start} deg, {omega} deg/s")
    active = level.max(axis=1) > level.max() - 40  # skip silent frames
    axes[1].plot(t[active], level[active, LEFT] - level[active, RIGHT], ".", markersize=3)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("Left - right level (dB)")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "3_surround_output.png"), dpi=150)
    plt.close(fig)

    audio_paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*")))
    for k in test_subjects:
        for a in audio_paths:
            audio_name = os.path.splitext(os.path.basename(a))[0]
            out = os.path.join(OUT_DIR, f"{names[k]}_{audio_name}_az{az_start}_w{omega}.wav")
            spatialize(sofa_paths[k], a, az_start, omega, el=0, out_path=out)
            print(f"  wrote {os.path.relpath(out, BASE_DIR)}")


if __name__ == "__main__":
    main()
