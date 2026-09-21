"""Measure your own hearing threshold and equal-loudness contours (ELC).

Two listening tasks, both with pure tones through headphones:

  1. Hearing threshold. An adaptive up/down staircase: after each tone you answer
     yes or no. The step shrinks after every reversal (8 -> 4 -> 2 dB) and the
     threshold is the mean of the last reversal levels. Silent catch trials are
     mixed in to check for guessing.
  2. Equal loudness (method of adjustment). A 1 kHz reference tone and a test tone
     alternate. You raise or lower the test tone with single key presses until
     both sound equally loud. Repeated for every test frequency and every
     reference level. The 1 kHz test tone is also matched against itself as a
     consistency check of the method.

All levels are digital levels in dBFS (dB relative to digital full scale) at a
fixed system volume, so the contours are relative. If you measured the 1 kHz tone
with a sound level meter, pass --spl-offset (dB SPL = dBFS + offset) and the plots
will also show approximate dB SPL.

Examples
    python src/elc_measure.py                  # guided run, 12 frequencies, threshold + 4 levels
    python src/elc_measure.py --quick          # 7 frequencies, threshold + 3 levels
    python src/elc_measure.py --demo           # simulated listener, tests the pipeline without headphones
    python src/elc_measure.py --freqs 125 250 500 1000 2000 4000 8000 \
                          --levels -50 -35 -20 --level-names low comfortable loud
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from iso226 import iso226_spl
from utils import DATA_DIR, DEFAULT_FS, ensure_dirs, make_tone, silence

DEFAULT_FREQS = [100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 8000, 10000]
QUICK_FREQS = [125, 250, 500, 1000, 2000, 4000, 8000]
DEFAULT_LEVELS = [-60.0, -48.0, -35.0, -20.0]
DEFAULT_LEVEL_NAMES = ["very_low", "low", "comfortable", "loud"]
QUICK_LEVELS = [-50.0, -35.0, -20.0]
QUICK_LEVEL_NAMES = ["low", "comfortable", "loud"]
REF_FREQ = 1000.0
TONE_DUR = 0.7       # seconds per tone
GAP_DUR = 0.3        # silence between reference and test tone
MAX_LEVEL_DBFS = -1.0  # never drive the DAC above this


class QuitRequested(Exception):
    """Raised when the listener types q."""


# ----------------------------------------------------------------------------- listeners
class RealListener:
    """Plays tones through the sound card and reads answers from the keyboard."""

    def __init__(self, fs: int = DEFAULT_FS, device=None):
        import sounddevice as sd
        self.sd = sd
        self.fs = fs
        self.device = device

    def play(self, x: np.ndarray) -> None:
        self.sd.play(x, self.fs, device=self.device)
        self.sd.wait()

    @staticmethod
    def _ask(prompt: str, valid) -> str:
        while True:
            ans = input(prompt).strip().lower()
            if ans in valid:
                return ans
            print("      Please type one of: " + " ".join(valid))

    def calibrate(self, comfortable_db: float, loud_db: float) -> None:
        print("\nCalibration")
        print("  1. Put on headphones and set the system volume to roughly 50 percent.")
        print(f"  2. A 1 kHz tone at {comfortable_db:.0f} dBFS will play. Adjust the SYSTEM volume until it")
        print("     feels comfortable, then do not touch the volume again during the session.")
        print(f"  3. A 1 kHz tone at {loud_db:.0f} dBFS will follow: it should be loud but not painful.")
        while True:
            input("  Press Enter to play the comfortable tone ... ")
            self.play(make_tone(REF_FREQ, comfortable_db, 1.0, self.fs))
            input("  Press Enter to play the loud tone ... ")
            self.play(make_tone(REF_FREQ, loud_db, 1.0, self.fs))
            if self._ask("  Volume OK?  [y]es  [r]epeat: ", ["y", "r"]) == "y":
                break

    def heard(self, freq: float, level_db):
        """Play one tone (silence if level_db is None). Returns True or False."""
        while True:
            x = silence(TONE_DUR, self.fs) if level_db is None else make_tone(freq, level_db, TONE_DUR, self.fs)
            self.play(x)
            ans = self._ask("      Did you hear a tone?  [y]es  [n]o  [r]eplay  [q]uit: ", ["y", "n", "r", "q"])
            if ans == "r":
                continue
            if ans == "q":
                raise QuitRequested
            return ans == "y"

    def adjust(self, freq: float, ref_level: float, start_level: float):
        """Method of adjustment. Returns dict(level, at_max) or None if skipped."""
        level = float(min(start_level, MAX_LEVEL_DBFS))
        steps = {"1": -1.0, "2": +1.0, "3": -3.0, "4": +3.0}
        print(f"      Reference {REF_FREQ:.0f} Hz at {ref_level:.0f} dBFS, then test tone {freq:.0f} Hz.")
        while True:
            x = np.concatenate([make_tone(REF_FREQ, ref_level, TONE_DUR, self.fs),
                                silence(GAP_DUR, self.fs),
                                make_tone(freq, level, TONE_DUR, self.fs)])
            self.play(x)
            ans = self._ask("      Test tone:  [1] -1 dB  [2] +1 dB  [3] -3 dB  [4] +3 dB  "
                            "[r] replay  [y] equally loud  [s] skip  [q] quit: ",
                            list(steps) + ["r", "y", "s", "q"])
            if ans in steps:
                new = level + steps[ans]
                if new > MAX_LEVEL_DBFS:
                    print("      Cannot go louder: digital full scale reached.")
                    new = MAX_LEVEL_DBFS
                level = new
            elif ans == "y":
                return {"level": level, "at_max": level >= MAX_LEVEL_DBFS}
            elif ans == "s":
                return None
            elif ans == "q":
                raise QuitRequested


class SimulatedListener:
    """A virtual listener that hears according to ISO 226 (plus a little noise).

    dB SPL is taken as dBFS + spl_offset. Used by --demo to test the whole pipeline
    without anybody listening.
    """

    def __init__(self, spl_offset: float = 95.0, noise_db: float = 1.0, seed: int = 0):
        self.spl_offset = spl_offset
        self.noise_db = noise_db
        self.rng = np.random.default_rng(seed)

    def calibrate(self, comfortable_db, loud_db):
        print("  (simulated listener: calibration skipped)")

    def _phon(self, freq: float, spl: float) -> float:
        """Loudness level of a tone at `spl` dB SPL, by inverting ISO 226."""
        thr = iso226_spl(0.0, [freq])[1][0]
        top = iso226_spl(90.0, [freq])[1][0]
        if spl <= thr:
            return spl - thr            # below threshold: negative, keeps ordering
        if spl >= top:
            return 90.0 + (spl - top)
        lo, hi = 0.0, 90.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if iso226_spl(mid, [freq])[1][0] < spl:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def heard(self, freq, level_db):
        if level_db is None:
            return bool(self.rng.random() < 0.05)  # occasional false alarm
        spl = level_db + self.spl_offset + self.rng.normal(0.0, self.noise_db)
        return bool(spl > iso226_spl(0.0, [freq])[1][0])

    def adjust(self, freq, ref_level, start_level):
        target = self._phon(REF_FREQ, ref_level + self.spl_offset)
        lo, hi = -130.0, MAX_LEVEL_DBFS
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self._phon(freq, mid + self.spl_offset) < target:
                lo = mid
            else:
                hi = mid
        level = 0.5 * (lo + hi) + self.rng.normal(0.0, self.noise_db)
        level = round(level)  # the real task works in 1 dB steps
        level = min(level, MAX_LEVEL_DBFS)
        return {"level": float(level), "at_max": level >= MAX_LEVEL_DBFS}


# ----------------------------------------------------------------------------- procedures
def measure_threshold(freq, listener, rng, start_db=-50.0, step_start=8.0, step_min=2.0,
                      n_reversals=6, max_trials=40, catch_prob=0.2):
    """Adaptive 1-up/1-down staircase. Returns dict with threshold (dBFS) and bookkeeping."""
    level, step = float(start_db), float(step_start)
    prev = None
    reversals, false_alarms, trials, catch_trials = [], 0, 0, 0
    while trials < max_trials and len(reversals) < n_reversals:
        if rng.random() < catch_prob:
            catch_trials += 1
            if listener.heard(freq, None):
                false_alarms += 1
            continue
        heard = listener.heard(freq, level)
        trials += 1
        if prev is not None and heard != prev:
            reversals.append(level)
            step = max(step_min, step / 2.0)
        prev = heard
        level = level - step if heard else level + step
        level = min(level, MAX_LEVEL_DBFS)
    thr = float(np.mean(reversals[-4:])) if reversals else float(level)
    return {"threshold_dbfs": thr, "trials": trials, "reversals": reversals,
            "catch_trials": catch_trials, "false_alarms": false_alarms}


def measure_contour(freqs, ref_level, listener, rng, start_jitter=6.0, verbose=True, name=""):
    """Equal-loudness contour at one reference level. Returns (levels, at_max, ref_check)."""
    order = list(freqs)
    rng.shuffle(order)
    result = {}
    for i, f in enumerate(order):
        if verbose:
            print(f"\n  [{name} {i + 1}/{len(order)}] test frequency {f:.0f} Hz")
        start = ref_level + rng.uniform(-start_jitter, start_jitter)
        result[f] = listener.adjust(f, ref_level, start)
    levels = [np.nan if result[f] is None else result[f]["level"] for f in freqs]
    at_max = [False if result[f] is None else bool(result[f]["at_max"]) for f in freqs]
    ref_check = None
    if REF_FREQ in result and result[REF_FREQ] is not None:
        ref_check = result[REF_FREQ]["level"] - ref_level  # ideal value is 0 dB
    return levels, at_max, ref_check


def save_results(path, results):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)


def print_summary(results):
    freqs = results["freqs"]
    curves = results["curves"]
    names = list(curves)
    print("\nSummary (levels in dBFS)")
    print("  freq(Hz) " + " ".join(f"{n:>12s}" for n in names))
    for i, f in enumerate(freqs):
        row = []
        for n in names:
            v = curves[n]["dbfs"][i]
            row.append(f"{'':>12s}" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:12.1f}")
        print(f"  {f:8.0f} " + " ".join(row))
    for n in names:
        rc = curves[n].get("ref_check_db")
        if rc is not None:
            print(f"  1 kHz self-match error for '{n}': {rc:+.1f} dB")
    thr = curves.get("threshold")
    if thr:
        print(f"  threshold catch trials: {thr['catch_trials']}, false alarms: {thr['false_alarms']}")


def run_measurement(freqs, levels, level_names, fs=DEFAULT_FS, demo=False, seed=None,
                    out_path=None, device=None, do_threshold=True, spl_offset=None,
                    catch_prob=0.2, thr_start=-50.0):
    ensure_dirs()
    rng = np.random.default_rng(seed)
    freqs = [float(f) for f in freqs]
    if REF_FREQ not in freqs:
        freqs = sorted(freqs + [REF_FREQ])
    if out_path is None:
        out_path = os.path.join(DATA_DIR, "elc_demo.json" if demo else "elc_measured.json")
        if os.path.exists(out_path):   # keep earlier measurements, add a timestamp
            stem, ext = os.path.splitext(out_path)
            out_path = f"{stem}_{time.strftime('%Y%m%d_%H%M')}{ext}"

    listener = SimulatedListener(seed=seed or 0) if demo else RealListener(fs, device)
    results = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "simulated": bool(demo),
        "fs": fs,
        "ref_freq": REF_FREQ,
        "freqs": freqs,
        "spl_offset": spl_offset,
        "tone_duration_s": TONE_DUR,
        "curves": {},
    }

    print("=" * 70)
    print("Equal-loudness contour measurement" + ("  [DEMO: simulated listener]" if demo else ""))
    print("=" * 70)
    print(f"Frequencies: {', '.join(f'{f:.0f}' for f in freqs)} Hz")
    print(f"Reference levels: {', '.join(f'{n}={l:.0f} dBFS' for n, l in zip(level_names, levels))}")
    n_judg = len(freqs) * (len(levels) + (1 if do_threshold else 0))
    print(f"About {n_judg} judgements; expect roughly {n_judg * 0.3:.0f} minutes. Results go to {out_path}")

    comfortable = levels[len(levels) // 2] if levels else -35.0
    listener.calibrate(comfortable, max(levels) if levels else -20.0)

    try:
        if do_threshold:
            print("\n" + "-" * 70 + "\nPart 1: hearing threshold (answer yes or no after each tone)\n" + "-" * 70)
            order = list(freqs)
            rng.shuffle(order)
            thr = {}
            for i, f in enumerate(order):
                print(f"\n  [threshold {i + 1}/{len(order)}] {f:.0f} Hz")
                thr[f] = measure_threshold(f, listener, rng, start_db=thr_start, catch_prob=catch_prob)
            results["curves"]["threshold"] = {
                "ref_dbfs": None,
                "dbfs": [thr[f]["threshold_dbfs"] for f in freqs],
                "trials": [thr[f]["trials"] for f in freqs],
                "catch_trials": int(sum(thr[f]["catch_trials"] for f in freqs)),
                "false_alarms": int(sum(thr[f]["false_alarms"] for f in freqs)),
            }
            save_results(out_path, results)

        for name, ref in zip(level_names, levels):
            print("\n" + "-" * 70 + f"\nPart 2: equal loudness, reference level '{name}' ({ref:.0f} dBFS)\n" + "-" * 70)
            lv, at_max, ref_check = measure_contour(freqs, ref, listener, rng, name=name)
            results["curves"][name] = {"ref_dbfs": ref, "dbfs": lv, "at_max": at_max,
                                       "ref_check_db": ref_check}
            save_results(out_path, results)
    except QuitRequested:
        print("\nStopped by user. Partial results were saved.")
    finally:
        save_results(out_path, results)

    print_summary(results)
    print(f"\nSaved {out_path}")
    return results, out_path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freqs", type=float, nargs="+", default=None, help="test frequencies in Hz")
    p.add_argument("--levels", type=float, nargs="+", default=None, help="1 kHz reference levels in dBFS")
    p.add_argument("--level-names", nargs="+", default=None, help="names for the reference levels")
    p.add_argument("--quick", action="store_true", help="7 frequencies, threshold + 3 levels")
    p.add_argument("--demo", action="store_true", help="simulated listener, no headphones needed")
    p.add_argument("--no-threshold", action="store_true", help="skip the threshold staircase")
    p.add_argument("--fs", type=int, default=DEFAULT_FS)
    p.add_argument("--device", type=int, default=None, help="sounddevice output device index")
    p.add_argument("--seed", type=int, default=None, help="random seed for order and start levels")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--spl-offset", type=float, default=None,
                   help="dB SPL of a 1 kHz tone at 0 dBFS, if measured with a meter")
    p.add_argument("--catch-prob", type=float, default=0.2, help="probability of silent catch trials")
    p.add_argument("--threshold-start", type=float, default=-50.0, help="staircase start level (dBFS)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    freqs = args.freqs or (QUICK_FREQS if args.quick else DEFAULT_FREQS)
    levels = args.levels or (QUICK_LEVELS if args.quick else DEFAULT_LEVELS)
    names = args.level_names or (QUICK_LEVEL_NAMES if args.quick else DEFAULT_LEVEL_NAMES)
    if len(names) != len(levels):
        names = [f"level{i + 1}" for i in range(len(levels))]
    run_measurement(freqs, levels, names, fs=args.fs, demo=args.demo, seed=args.seed,
                    out_path=args.out, device=args.device, do_threshold=not args.no_threshold,
                    spl_offset=args.spl_offset, catch_prob=args.catch_prob,
                    thr_start=args.threshold_start)


if __name__ == "__main__":
    main()
