"""
Project 4: Linear Predictive Coding (LPC vocoder)
ECE 520.445/645 Audio Signal Processing, Fall 2026

Run:  python project4.py

Everything is written from scratch with numpy:
  - LPC coefficients: autocorrelation method + Levinson-Durbin (no lpc/levinson library call)
  - voicing / pitch: normalized autocorrelation (no pitch library call)
Library functions used only for utilities:
  - soundfile (read/write wav), matplotlib (plots)
  - scipy.signal.lfilter / lfiltic (running the all-pole synthesis filter)
  - scipy.signal.resample_poly (only if the own recording is not at 16 kHz)
  - scipy.signal.spectrogram (plots only)
  - pesq (https://pypi.org/project/pesq/): quality score only, not part of the codec
"""

import os
import numpy as np
import soundfile as sf
from scipy.signal import lfilter, lfiltic, resample_poly, spectrogram
from pesq import pesq as pesq_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(HERE, "Project4", "Project4_LPC_Audio")
VOICING_DIR = os.path.join(HERE, "Project4", "Project4_LPC_Voicing")
RECORDING = os.path.join(HERE, "my_recording.wav")       # Part 2 input
FIG_DIR = os.path.join(HERE, "figures")
OUT_DIR = os.path.join(HERE, "output_audio")

FS = 16000
VOICING_RATE = 100          # voicing frames per second (same as the provided files)
PREEMPH = 0.97
BITS_PER_PARAM = 16         # assumed word length when a rate is quoted in bits/s

# Default vocoder settings (chosen from the sweeps in main)
ORDER = 16
FRAME_RATE = 50             # LPC frames per second
WIN_MS = 30                 # analysis window length
GLOTTAL = "rosenberg_tc40"   # tilt-compensated Rosenberg pulse, 40% open quotient


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_audio(path):
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != FS:
        g = np.gcd(int(fs), FS)
        x = resample_poly(x, FS // g, int(fs) // g)
    return x.astype(float)


def save_audio(path, y):
    peak = np.max(np.abs(y))
    if peak > 0.99:
        y = y * 0.99 / peak
    sf.write(path, y, FS, subtype="PCM_16")


def load_voicing(path):
    return np.array([float(v) for v in open(path).read().split()])


# ----------------------------------------------------------------------------
# LPC analysis (from scratch)
# ----------------------------------------------------------------------------
def autocorr(x, maxlag):
    return np.array([np.dot(x[:len(x) - k], x[k:]) for k in range(maxlag + 1)])


def levinson(r, order):
    """Levinson-Durbin recursion. Returns a (a[0] = 1, A(z) = 1 + sum a_k z^-k)
    and the prediction error energy E."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    E = r[0]
    for i in range(1, order + 1):
        if E <= 0:
            break
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / E
        a_prev = a.copy()
        a[i] = k
        a[1:i] = a_prev[1:i] + k * a_prev[i - 1:0:-1]
        E *= (1.0 - k * k)
    return a, E


def lpc_frame(x, start, hop, win, order):
    """LPC coefficients and gain of the frame that covers x[start:start+hop].
    The analysis window (Hamming, win samples) is centered on that hop."""
    c = start + hop // 2
    lo, hi = c - win // 2, c - win // 2 + win
    seg = np.zeros(win)
    a0, b0 = max(lo, 0), min(hi, len(x))
    if b0 > a0:
        seg[a0 - lo:b0 - lo] = x[a0:b0]
    w = np.hamming(win)
    seg = seg * w
    r = autocorr(seg, order)
    if r[0] < 1e-12:                       # silent frame
        a = np.zeros(order + 1)
        a[0] = 1.0
        return a, 0.0
    r[0] *= 1.0 + 1e-9                     # tiny white-noise correction for stability
    a, E = levinson(r, order)
    G = np.sqrt(max(E, 0.0) / np.sum(w * w))   # gain for a unit-power excitation
    return a, G


# ----------------------------------------------------------------------------
# Voicing / pitch estimator (from scratch, autocorrelation method)
# ----------------------------------------------------------------------------
def lowpass_fir(cutoff_hz, ntaps=101):
    n = np.arange(ntaps) - (ntaps - 1) / 2
    h = np.sinc(2 * cutoff_hz / FS * n) * np.hamming(ntaps)
    return h / h.sum()


def median_smooth(v, k):
    half = k // 2
    out = v.copy()
    for i in range(len(v)):
        out[i] = np.median(v[max(0, i - half):i + half + 1])
    return out


def estimate_voicing(x, fmin=60.0, fmax=400.0, win_ms=40, corr_thr=0.45,
                     zcr_thr=0.20, energy_db=-30.0):
    """Returns the pitch in Hz for every 10 ms frame (0 = unvoiced or silence).

    For each frame: low-pass to 900 Hz, center-clip, normalized autocorrelation
    over the lags of fmin..fmax. The frame is voiced if the autocorrelation peak
    is high, the zero-crossing rate is low and the frame is not silent."""
    hop = FS // VOICING_RATE
    win = int(win_ms * FS / 1000)
    nfr = int(np.ceil(len(x) / hop))
    xl = np.convolve(x, lowpass_fir(900.0), mode="same")
    pad = win
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad + hop)])
    xlp = np.concatenate([np.zeros(pad), xl, np.zeros(pad + hop)])
    lag_min, lag_max = int(FS / fmax), int(FS / fmin)

    rms = np.zeros(nfr)
    zcr = np.zeros(nfr)
    peak = np.zeros(nfr)
    f0 = np.zeros(nfr)
    for i in range(nfr):
        c = pad + i * hop + hop // 2
        seg = xp[c - win // 2:c + win // 2]
        rms[i] = np.sqrt(np.mean(seg ** 2))
        short = xp[c - hop:c + hop]
        zcr[i] = np.mean(np.abs(np.diff(np.sign(short))) > 0)

        s = xlp[c - win // 2:c + win // 2 + lag_max + 2].copy()
        clip = 0.3 * np.max(np.abs(s[:win]))
        s = np.where(s > clip, s - clip, np.where(s < -clip, s + clip, 0.0))
        ref = s[:win]
        e0 = np.dot(ref, ref)
        if e0 <= 0:
            continue
        ncc = np.zeros(lag_max + 2)
        for k in range(lag_min - 1, lag_max + 2):
            seg_k = s[k:k + win]
            ek = np.dot(seg_k, seg_k)
            if ek > 0:
                ncc[k] = np.dot(ref, seg_k) / np.sqrt(e0 * ek)
        search = ncc[lag_min:lag_max + 1]
        best = search.max()
        # take the shortest lag that is a local peak close to the best one (avoids pitch halving)
        k_best = lag_min + int(np.argmax(search))
        for k in range(lag_min, lag_max + 1):
            if ncc[k] >= 0.9 * best and ncc[k] >= ncc[k - 1] and ncc[k] >= ncc[k + 1]:
                k_best = k
                break
        # parabolic interpolation of the peak
        y0, y1, y2 = ncc[k_best - 1], ncc[k_best], ncc[k_best + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        peak[i] = best
        f0[i] = FS / (k_best + np.clip(delta, -0.5, 0.5))

    level_db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    voiced = (peak > corr_thr) & (zcr < zcr_thr) & (level_db > energy_db)

    # post-processing: remove isolated decisions, then smooth the pitch track
    v = voiced.copy()
    for i in range(1, nfr - 1):
        if voiced[i - 1] == voiced[i + 1] != voiced[i]:
            v[i] = voiced[i - 1]
    f0 = np.where(v, f0, 0.0)
    idx = np.where(v)[0]
    if len(idx):
        filled = np.interp(np.arange(nfr), idx, f0[idx])   # fill the frames just switched to voiced
        f0 = np.where(v, median_smooth(np.where(f0 > 0, f0, filled), 5), 0.0)
    return f0


def compare_voicing(own, ref):
    """Agreement between two 10 ms pitch tracks."""
    n = min(len(own), len(ref))
    own, ref = own[:n], ref[:n]
    vo, vr = own > 0, ref > 0
    both = vo & vr
    rel = np.abs(own[both] - ref[both]) / ref[both]
    return {
        "agree": np.mean(vo == vr) * 100,
        "v_as_u": np.sum(vr & ~vo) / max(np.sum(vr), 1) * 100,   # reference voiced, ours unvoiced
        "u_as_v": np.sum(~vr & vo) / max(np.sum(~vr), 1) * 100,
        "pitch_err": np.median(rel) * 100 if both.any() else 0.0,
        "gross": np.mean(rel > 0.2) * 100 if both.any() else 0.0,
    }


# ----------------------------------------------------------------------------
# Transmitter
# ----------------------------------------------------------------------------
def frame_pitch(f0_track, start, hop):
    """Pitch of an LPC frame from the 10 ms voicing track: voiced if at least half
    of the 10 ms frames inside it are voiced; pitch = their median."""
    vh = FS // VOICING_RATE
    i0 = start // vh
    i1 = max(i0 + 1, int(round((start + hop) / vh)))
    seg = f0_track[i0:min(i1, len(f0_track))]
    if len(seg) == 0:
        return 0.0
    nz = seg[seg > 0]
    return float(np.median(nz)) if len(nz) * 2 >= len(seg) else 0.0


def encode(x, f0_track, order=ORDER, frame_rate=FRAME_RATE, win_ms=WIN_MS,
           order_unvoiced=None):
    """Transmitter: per frame {a_1..a_p, G, pitch (0 = unvoiced)}.
    order_unvoiced, if given, is a lower LPC order used for unvoiced frames."""
    hop = int(round(FS / frame_rate))
    win = max(int(win_ms * FS / 1000), hop)
    xp = lfilter([1.0, -PREEMPH], [1.0], x)
    nfr = int(np.ceil(len(x) / hop))
    frames = []
    nparams = 0
    for i in range(nfr):
        f0 = frame_pitch(f0_track, i * hop, hop)
        p = order if (f0 > 0 or order_unvoiced is None) else order_unvoiced
        a, G = lpc_frame(xp, i * hop, hop, win, p)
        frames.append((a, G, f0))
        nparams += p + 2                       # p coefficients + gain + pitch/voicing
    dur = len(x) / FS
    return {"frames": frames, "hop": hop, "n": len(x),
            "params_per_sec": nparams / dur}


# ----------------------------------------------------------------------------
# Receiver
# ----------------------------------------------------------------------------
def rosenberg(T, oq=0.56):
    """Rosenberg glottal flow pulse. oq = open quotient (open part of the period);
    the opening phase takes 5/7 of it and the closing phase 2/7 (40% + 16% for oq = 0.56)."""
    n1, n2 = max(int(oq * 5 / 7 * T), 2), max(int(oq * 2 / 7 * T), 1)
    return np.concatenate([0.5 * (1 - np.cos(np.pi * np.arange(n1) / n1)),
                           np.cos(np.pi * np.arange(n2) / (2 * n2))])


def glottal_pulse(shape, T):
    """One excitation pulse for pitch period T (samples), scaled so that a train of
    them has unit power.
      impulse          single sample
      rosenberg        Rosenberg glottal flow pulse
      rosenberg_d      its derivative (flow + lip radiation)
      rosenberg_tcXX   derivative + tilt compensation, open quotient XX %. The pulse is
                       passed through the pre-emphasis filter, which cancels the
                       de-emphasis at the output, so the voiced path becomes
                       glottal flow -> radiation -> vocal tract filter."""
    if shape == "impulse":
        g = np.array([1.0])
    elif shape == "rosenberg":
        g = rosenberg(T)
    elif shape == "rosenberg_d":
        g = np.diff(rosenberg(T), prepend=0.0, append=0.0)
    elif shape.startswith("rosenberg_tc"):
        g = np.diff(rosenberg(T, int(shape[12:]) / 100), prepend=0.0, append=0.0)
        g = np.convolve(g, [1.0, -PREEMPH])
    else:
        raise ValueError(shape)
    return g * np.sqrt(T / np.sum(g * g))


def make_excitation(frames, hop, n, shape=GLOTTAL, seed=0):
    """Pulse train for voiced frames, white noise for unvoiced frames (unit power).
    The position of the next pitch pulse is carried from frame to frame."""
    rng = np.random.default_rng(seed)
    e = np.zeros(n + 2 * hop + FS // 50)
    pos = 0.0                                   # position of the next pitch pulse
    for i, (_, _, f0) in enumerate(frames):
        s, t = i * hop, (i + 1) * hop
        if f0 > 0:
            T = FS / f0
            pos = max(pos, s)
            while pos < t:
                g = glottal_pulse(shape, T)
                k = int(round(pos))
                e[k:k + len(g)] += g
                pos += T
        else:
            e[s:t] += rng.standard_normal(hop)
            pos = t
    return e[:n]


def synth_filter(frames, hop, exc, gains=None):
    """Run the time-varying all-pole filter G/A(z), one frame at a time, keeping
    the filter memory (past output samples) across frames; then de-emphasis."""
    n = len(exc)
    y = np.zeros(n)
    pmax = max(len(a) for a, _, _ in frames) - 1
    for i, (a, G, _) in enumerate(frames):
        s, t = i * hop, min((i + 1) * hop, n)
        if s >= n:
            break
        g = G if gains is None else gains[i]
        past = y[max(0, s - pmax):s][::-1]
        zi = lfiltic([1.0], a, past[:len(a) - 1])
        y[s:t], _ = lfilter([1.0], a, g * exc[s:t], zi=zi)
    return lfilter([1.0], [1.0, -PREEMPH], y)


def decode(code, shape=GLOTTAL, seed=0):
    """Receiver: rebuild the speech from {a_k, G, pitch} only."""
    exc = make_excitation(code["frames"], code["hop"], code["n"], shape, seed)
    return synth_filter(code["frames"], code["hop"], exc)


# ----------------------------------------------------------------------------
# Residual (prediction error) excited synthesis
# ----------------------------------------------------------------------------
# step sizes of the optimal uniform quantizer for a unit-variance Gaussian (Max, 1960)
QSTEP = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.335}


def residual_codec(x, code, bits=None):
    """Send the prediction error instead of the pulse/noise excitation.
    bits = None: unquantized residual; otherwise a uniform quantizer with that many
    bits per sample, scaled by the frame's residual RMS (sent in place of G)."""
    frames, hop, n = code["frames"], code["hop"], code["n"]
    xp = lfilter([1.0, -PREEMPH], [1.0], x)
    res = np.zeros(n)
    for i, (a, _, _) in enumerate(frames):
        s, t = i * hop, min((i + 1) * hop, n)
        p = len(a) - 1
        seg = np.concatenate([np.zeros(max(0, p - s)), xp[max(0, s - p):t]])
        res[s:t] = lfilter(a, [1.0], seg)[p:]
    if bits is not None:
        L = 2 ** bits
        for i in range(len(frames)):
            s, t = i * hop, min((i + 1) * hop, n)
            sigma = np.sqrt(np.mean(res[s:t] ** 2)) + 1e-12
            step = QSTEP[bits] * sigma
            q = np.clip(np.floor(res[s:t] / step), -L // 2, L // 2 - 1)
            res[s:t] = (q + 0.5) * step
    y = synth_filter(frames, hop, res, gains=np.ones(len(frames)))
    return y, res


# ----------------------------------------------------------------------------
# Quality measures
# ----------------------------------------------------------------------------
def stft_db(x, nfft=512, hop=160):
    w = np.hanning(nfft)
    nfr = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(nfr)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * w, axis=1)) ** 2
    return S


def lsd(x, y, nbands=32):
    """Log-spectral distance in dB between original x and synthesis y.
    Power spectra are averaged into 32 equal-width bands (250 Hz) first, so the
    measure compares spectral envelopes and levels, not the exact pitch harmonics
    or the noise realization. Averaged over the non-silent frames."""
    n = min(len(x), len(y))
    X, Y = stft_db(x[:n]), stft_db(y[:n])
    nb = X.shape[1] - 1
    X = X[:, 1:].reshape(len(X), nbands, nb // nbands).mean(axis=2)
    Y = Y[:, 1:].reshape(len(Y), nbands, nb // nbands).mean(axis=2)
    en = X.sum(axis=1)
    active = en > en.max() * 1e-4                     # frames within 40 dB of the loudest
    floor = X[active].max() * 1e-8
    d = 10 * np.log10(X[active] + floor) - 10 * np.log10(Y[active] + floor)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=1))))


def pesq_wb(x, y):
    """Wide-band PESQ score (MOS-LQO, 1 to 4.64) from the pesq package."""
    n = min(len(x), len(y))
    return float(pesq_score(FS, x[:n], y[:n], "wb"))


def snr_db(x, y):
    n = min(len(x), len(y))
    return float(10 * np.log10(np.sum(x[:n] ** 2) / (np.sum((x[:n] - y[:n]) ** 2) + 1e-20)))


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def load_samples():
    data = []
    for i in range(1, 11):
        x = load_audio(os.path.join(AUDIO_DIR, f"Sample{i}.wav"))
        ref = load_voicing(os.path.join(VOICING_DIR, f"sample{i}.txt"))
        own = estimate_voicing(x)
        data.append({"name": f"Sample{i}", "x": x, "ref": ref, "own": own})
    return data


def evaluate(data, voicing="own", shape=GLOTTAL, **kw):
    """Mean PESQ, mean LSD and mean parameters/s over the 10 files for one setting."""
    q, l, r = [], [], []
    for d in data:
        code = encode(d["x"], d[voicing], **kw)
        y = decode(code, shape)
        q.append(pesq_wb(d["x"], y))
        l.append(lsd(d["x"], y))
        r.append(code["params_per_sec"])
    return float(np.mean(q)), float(np.mean(l)), float(np.mean(r))


def part1(data):
    print("\n=== Part 1: 10 provided files "
          f"(order {ORDER}, {FRAME_RATE} frames/s, {WIN_MS} ms window, {GLOTTAL}) ===")
    print(f"{'file':9s} {'param/s':>8s} {'kbit/s':>7s} {'PESQ':>6s} {'LSD dB':>7s}")
    rows = []
    for d in data:
        code = encode(d["x"], d["own"])
        d["y"] = decode(code)
        save_audio(os.path.join(OUT_DIR, f"{d['name']}_lpc.wav"), d["y"])
        rows.append((code["params_per_sec"], pesq_wb(d["x"], d["y"]), lsd(d["x"], d["y"])))
        r = rows[-1]
        print(f"{d['name']:9s} {r[0]:8.0f} {r[0] * BITS_PER_PARAM / 1000:7.1f} {r[1]:6.2f} {r[2]:7.2f}")
    r = np.mean(rows, axis=0)
    print(f"{'mean':9s} {r[0]:8.0f} {r[0] * BITS_PER_PARAM / 1000:7.1f} {r[1]:6.2f} {r[2]:7.2f}")
    print(f"original PCM: {FS} samples/s = {FS * 16 / 1000:.0f} kbit/s "
          f"(compression {FS / r[0]:.1f}x in samples/s)")


def voicing_experiment(data):
    print("\n=== Voicing: own estimator vs provided track ===")
    print(f"{'file':9s} {'own V%':>7s} {'ref V%':>7s} {'agree':>7s} {'ref V->own U':>13s} "
          f"{'ref U->own V':>13s} {'pitch diff':>10s}")
    rows = []
    for d in data:
        c = compare_voicing(d["own"], d["ref"])
        rows.append((np.mean(d["own"] > 0) * 100, np.mean(d["ref"] > 0) * 100, c["agree"],
                     c["v_as_u"], c["u_as_v"], c["pitch_err"]))
        r = rows[-1]
        print(f"{d['name']:9s} {r[0]:6.0f}% {r[1]:6.0f}% {r[2]:6.1f}% {r[3]:12.1f}% "
              f"{r[4]:12.1f}% {r[5]:9.1f}%")
    r = np.mean(rows, axis=0)
    print(f"{'mean':9s} {r[0]:6.0f}% {r[1]:6.0f}% {r[2]:6.1f}% {r[3]:12.1f}% {r[4]:12.1f}% {r[5]:9.1f}%")

    print("\n=== Contribution of voicing to the reconstruction ===")
    for d in data:
        d["all_u"] = np.zeros_like(d["own"])
        d["all_v"] = np.full_like(d["own"], np.median(d["own"][d["own"] > 0]))
    for key, name in [("own", "own voicing estimator"), ("ref", "provided voicing"),
                      ("all_u", "all frames unvoiced (noise)"),
                      ("all_v", "all frames voiced (median pitch)")]:
        q, l, _ = evaluate(data, key)
        print(f"  {name:33s}: PESQ {q:.2f}, LSD {l:.2f} dB")
    d = data[0]
    for k in ("ref", "all_u", "all_v"):
        save_audio(os.path.join(OUT_DIR, f"extra_{d['name']}_voicing_{k}.wav"),
                   decode(encode(d["x"], d[k])))

    # figure: narrow-band spectrogram with both pitch tracks
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2))
    for a, d in zip(ax, (data[0], data[2])):
        f, t, S = spectrogram(d["x"], FS, nperseg=800, noverlap=720, nfft=2048)
        a.pcolormesh(t, f, 10 * np.log10(S + 1e-12), vmin=-100, vmax=-30, shading="auto")
        n = min(len(d["own"]), len(d["ref"]))
        tf = (np.arange(n) + 0.5) / VOICING_RATE
        a.plot(tf, np.where(d["ref"][:n] > 0, d["ref"][:n], np.nan), "o", ms=5, mfc="none",
               mec="white", label="provided")
        a.plot(tf, np.where(d["own"][:n] > 0, d["own"][:n], np.nan), ".", ms=5, color="red",
               label="own estimator")
        a.set_ylim(0, 800)
        a.set_xlim(0, len(d["x"]) / FS)
        a.set_ylabel("frequency (Hz)")
        a.set_title(f"{d['name']}: narrow-band spectrogram and pitch tracks", fontsize=10)
        a.legend(fontsize=8, loc="upper right")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "voicing.png"), dpi=150)
    plt.close(fig)


def sweep_rate_quality(data):
    print("\n=== Compression rate vs quality (mean over the 10 files) ===")
    orders = [8, 10, 12, 16, 20, 24]
    rates = [25, 33, 50, 67, 100]
    Q = np.zeros((len(rates), len(orders)))
    L = np.zeros_like(Q)
    R = np.zeros_like(Q)
    for i, fr in enumerate(rates):
        for j, p in enumerate(orders):
            Q[i, j], L[i, j], R[i, j] = evaluate(data, order=p, frame_rate=fr)
    for name, T, fmt in [("PESQ", Q, "7.2f"), ("LSD (dB)", L, "7.2f"), ("parameters/s", R, "7.0f")]:
        print(f"{name}\nframes/s | " + " ".join(f"p={p:<5d}" for p in orders))
        for i, fr in enumerate(rates):
            print(f"{fr:8d} | " + " ".join(format(v, fmt) for v in T[i]))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, fr in enumerate(rates):
        ax[0].plot(R[i], Q[i], "o-", ms=4, label=f"{fr} frames/s")
    ax[0].plot(R[rates.index(FRAME_RATE), orders.index(ORDER)],
               Q[rates.index(FRAME_RATE), orders.index(ORDER)], "k*", ms=14, label="chosen")
    ax[0].set_xlabel("parameters per second")
    ax[0].set_ylabel("PESQ (wide-band)")
    ax[0].set_title("Quality vs rate (points on a line: order 8 to 24)", fontsize=10)
    ax[0].legend(fontsize=8)
    for j, p in enumerate(orders):
        ax[1].plot(rates, Q[:, j], "o-", ms=4, label=f"order {p}")
    ax[1].set_xlabel("frame rate (frames/s)")
    ax[1].set_ylabel("PESQ (wide-band)")
    ax[1].set_title("Effect of the frame rate", fontsize=10)
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rate_vs_quality.png"), dpi=150)
    plt.close(fig)

    print(f"analysis window length (order {ORDER}, {FRAME_RATE} frames/s):")
    for w in [20, 30, 40]:
        q, l, _ = evaluate(data, win_ms=w)
        print(f"  {w} ms: PESQ {q:.2f}, LSD {l:.2f} dB")


def glottal_experiment(data):
    print("\n=== Glottal pulse shapes (mean over the 10 files) ===")
    shapes = ["impulse", "rosenberg", "rosenberg_d", "rosenberg_tc70", "rosenberg_tc56",
              "rosenberg_tc40", "rosenberg_tc25"]
    for sh in shapes:
        q, l, _ = evaluate(data, shape=sh)
        print(f"  {sh:15s}: PESQ {q:.2f}, LSD {l:.2f} dB")
    d = data[0]
    code = encode(d["x"], d["own"])
    for sh in ["impulse", "rosenberg"]:
        save_audio(os.path.join(OUT_DIR, f"extra_{d['name']}_glottal_{sh}.wav"), decode(code, sh))

    # figure: the excitation of one pitch period after the output de-emphasis, i.e. the
    # effective source that drives the vocal tract filter, and its spectrum
    T = FS / 120.0
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for sh in ["impulse", "rosenberg", "rosenberg_d", "rosenberg_tc40"]:
        g = np.zeros(int(T) + 1)
        gp = glottal_pulse(sh, T)
        g[:min(len(gp), len(g))] = gp[:len(g)]
        g = lfilter([1.0], [1.0, -PREEMPH], g)
        ax[0].plot(np.arange(len(g)) / FS * 1000, g / np.max(np.abs(g)), label=sh)
        H = np.abs(np.fft.rfft(g, 2048))
        ax[1].plot(np.fft.rfftfreq(2048, 1 / FS), 20 * np.log10(H / H.max() + 1e-6), label=sh)
    ax[0].set_xlabel("time (ms)")
    ax[0].set_title("Effective source pulse, one period at 120 Hz", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("dB")
    ax[1].set_ylim(-90, 5)
    ax[1].set_title("Spectrum of the pulse", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "glottal_shapes.png"), dpi=150)
    plt.close(fig)


def residual_experiment(data):
    print("\n=== Using the prediction error as excitation (mean over the 10 files) ===")
    print(f"{'excitation':24s} {'kbit/s':>8s} {'PESQ':>6s} {'LSD dB':>7s} {'SNR dB':>7s}")
    q, l, r = evaluate(data)
    print(f"{'pulse/noise (vocoder)':24s} {r * BITS_PER_PARAM / 1000:8.1f} {q:6.2f} {l:7.2f} {'-':>7s}")
    for bits in [1, 2, 3, 4, None]:
        rows = []
        for d in data:
            code = encode(d["x"], d["own"])
            y, _ = residual_codec(d["x"], code, bits)
            # per frame: p coefficients + residual RMS (no pitch); plus the residual samples
            side = (code["params_per_sec"] - FRAME_RATE) * BITS_PER_PARAM
            rows.append((side + FS * (bits if bits else 16), pesq_wb(d["x"], y),
                         lsd(d["x"], y), snr_db(d["x"], y)))
            if d is data[0] and bits in (1, 2):
                save_audio(os.path.join(OUT_DIR, f"extra_{d['name']}_residual_{bits}bit.wav"), y)
        m = np.mean(rows, axis=0)
        name = "residual, unquantized" if bits is None else f"residual, {bits} bit/sample"
        print(f"{name:24s} {m[0] / 1000:8.1f} {m[1]:6.2f} {m[2]:7.2f} {m[3]:7.1f}")


def plot_reconstruction(x, y, title, fname):
    fig, ax = plt.subplots(2, 2, figsize=(10, 4.6), sharex=True)
    t = np.arange(len(x)) / FS
    for k, (sig, name) in enumerate([(x, "original"), (y, "LPC synthesis")]):
        ax[0, k].plot(t, sig[:len(t)], lw=0.5)
        ax[0, k].set_title(f"{title}: {name}", fontsize=10)
        f, tt, S = spectrogram(sig, FS, nperseg=400, noverlap=300, nfft=512)
        ax[1, k].pcolormesh(tt, f / 1000, 10 * np.log10(S + 1e-12), vmin=-110, vmax=-30,
                            shading="auto")
        ax[1, k].set_xlabel("time (s)")
    ax[0, 0].set_ylabel("amplitude")
    ax[1, 0].set_ylabel("frequency (kHz)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close(fig)


# Part 2 settings for the own recording (see the report)
PART2_BASELINE = dict(order=ORDER, frame_rate=FRAME_RATE)             # Part 1 settings
PART2_OPTIMIZED = dict(order=14, order_unvoiced=20, frame_rate=40)    # fewer poles for voiced
PART2_ENERGY_DB = -40.0     # silence threshold of the voicing estimator for this recording


def part2():
    print("\n=== Part 2: own recording ===")
    if not os.path.exists(RECORDING):
        print(f"  {os.path.basename(RECORDING)} not found, Part 2 skipped.")
        return
    x = load_audio(RECORDING)
    x = x / (np.max(np.abs(x)) + 1e-12) * 0.9
    f0_base = estimate_voicing(x)
    f0 = estimate_voicing(x, energy_db=PART2_ENERGY_DB)
    print(f"  duration {len(x) / FS:.2f} s, voiced frames {np.mean(f0_base > 0) * 100:.0f}% "
          f"(silence threshold -30 dB), {np.mean(f0 > 0) * 100:.0f}% ({PART2_ENERGY_DB:.0f} dB)")

    def run(track, **kw):
        code = encode(x, track, **kw)
        y = decode(code)
        return y, f"{code['params_per_sec']:8.0f} {pesq_wb(x, y):6.2f} {lsd(x, y):7.2f}"

    head = f"{'param/s':>8s} {'PESQ':>6s} {'LSD dB':>7s}"
    print(f"  order of the unvoiced frames (voiced 16, 50 frames/s)   {head}")
    for pu in [8, 12, 16, 20, 24]:
        print(f"    unvoiced order {pu:2d}                                    "
              + run(f0, order=16, order_unvoiced=pu, frame_rate=50)[1])
    print(f"  order of the voiced frames (unvoiced 16, 50 frames/s)   {head}")
    for pv in [10, 12, 14, 16, 20]:
        print(f"    voiced order {pv:2d}                                      "
              + run(f0, order=pv, order_unvoiced=16, frame_rate=50)[1])
    print(f"  frame rate (voiced 14, unvoiced 20)                     {head}")
    for fr in [33, 40, 50, 67]:
        print(f"    {fr:3d} frames/s                                         "
              + run(f0, order=14, order_unvoiced=20, frame_rate=fr)[1])

    y, txt = run(f0_base, **PART2_BASELINE)
    print(f"  baseline (Part 1 settings)                              {txt}")
    save_audio(os.path.join(OUT_DIR, "my_recording_lpc_baseline.wav"), y)
    y, txt = run(f0, **PART2_OPTIMIZED)
    print(f"  optimized {PART2_OPTIMIZED}".ljust(58) + txt)
    save_audio(os.path.join(OUT_DIR, "my_recording_lpc.wav"), y)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_samples()
    part1(data)
    plot_reconstruction(data[0]["x"], data[0]["y"], data[0]["name"], "reconstruction.png")
    voicing_experiment(data)
    sweep_rate_quality(data)
    glottal_experiment(data)
    residual_experiment(data)
    part2()
    print(f"\nDone. Audio in {OUT_DIR}, figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
