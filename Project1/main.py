"""Project 1, Equal-Loudness Curves: interactive menu.

    python main.py

  1  Listening test (threshold + equal-loudness contours), then plot the curves;
     if a result already exists, just plot it
  2  Generate test audio (if missing) and run the equalizer with the ISO curve and my curve
  3  Record from the microphone, equalize the recording, save it
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))   # all other code lives in src/

from utils import AUDIO_IN_DIR, DATA_DIR, ensure_dirs, list_measurements

MENU = """
==================== Project 1: Equal-Loudness Curves ====================
  1  Listening test (headphones): new measurement saved as a new file,
     or plot an existing one   -> data/elc_measured*.json, figures/fig1_*
  2  Generate test audio and run the equalizer
     -> audio/output/, figures/fig2, figures/eq_summary.md
  3  Record from the microphone, equalize, save
  q  Quit
"""


def ask(prompt: str, default: str = "") -> str:
    s = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return s or default


def yes(prompt: str, default: bool = False) -> bool:
    return ask(prompt + " (y/n)", "y" if default else "n").lower().startswith("y")


def measured_file(ask_which: bool = False) -> str | None:
    """Newest measurement file, or let the user pick one when several exist."""
    files = list_measurements()
    if not files:
        return None
    if len(files) == 1 or not ask_which:
        return files[-1]
    print("  Measurement files (newest last):")
    for i, f in enumerate(files, 1):
        print(f"    {i:2d}  {os.path.relpath(f)}")
    s = ask("  Which one", str(len(files)))
    return files[int(s) - 1] if s.isdigit() and 1 <= int(s) <= len(files) else files[-1]


def curve_args() -> list[str]:
    """Ask which curve to use; returns equalizer.py arguments."""
    measured = measured_file()
    src = ask("  Curve source: iso / measured", "measured" if measured else "iso").lower()
    args = []
    if src.startswith("m") and measured:
        args += ["--curve", measured_file(ask_which=True)]
        args += ["--level", ask("  Curve name (threshold/very_low/low/comfortable/loud)", "comfortable")]
        args += ["--smooth", ask("  Smoothing in octaves (0 = none)", "0.5")]
    else:
        args += ["--curve", "iso", "--phon", ask("  Playback level in phon (0 to 90)", "40")]
    args += ["--strength", ask("  Strength (1 = full curve, 0.5 = half)", "1")]
    args += ["--max-boost", ask("  Maximum boost in dB", "20")]
    return args


def choose_file() -> str | None:
    files = sorted(glob.glob(os.path.join(AUDIO_IN_DIR, "*.*")))
    for i, f in enumerate(files, 1):
        print(f"    {i:2d}  {os.path.relpath(f)}")
    s = ask("  File number or path", "1" if files else "")
    if s.isdigit() and 1 <= int(s) <= len(files):
        return files[int(s) - 1]
    return s if os.path.exists(s) else None


def run_listening_test():
    import elc_measure
    import plot_elc
    files = list_measurements()
    if files:
        print(f"  {len(files)} measurement file(s) found in data/.")
        choice = ask("  (m) new measurement, saved as a new file  or  (p) plot an existing one", "p").lower()
    else:
        choice = "m"
    if choice.startswith("p"):
        elc = measured_file(ask_which=True)
    else:
        elc_measure.main([])
        elc = list_measurements()[-1]
    print("  Plotting ...")
    plot_elc.main(["--no-show", "--elc", elc])
    stem = os.path.splitext(os.path.basename(elc))[0]
    print(f"  Done: {os.path.relpath(elc)}, figures/fig1_{stem}.png, figures/{stem}_table.md")


def run_equalizer():
    import io
    import contextlib
    import analyze_eq
    import make_test_audio
    if not glob.glob(os.path.join(AUDIO_IN_DIR, "*.wav")):
        print("  Generating test audio ...")
        with contextlib.redirect_stdout(io.StringIO()):
            make_test_audio.main()
    print("  Running the equalizer on all test files, please wait ...")
    with contextlib.redirect_stdout(io.StringIO()):
        info = analyze_eq.main(["--no-show", "--quiet"])
    if info is None:
        print("  nothing processed")
        return
    print(f"""
  Input      audio/input/   {', '.join(info['inputs'])}
             (speech from text-to-speech, synthetic music, pink noise, log sweep)
  Curves     {len(info['presets'])} presets (measured ones from {os.path.relpath(measured_file()) if measured_file() else 'none'}):
             """ + "\n             ".join(info["presets"]) + f"""
  Processing each file is cut into {info['nfft']}-sample frames (46 ms, 50 % overlap), each frame is
             FFT-ed, every frequency bin is multiplied by the gain the curve asks for
             (relative to 1 kHz, boost limited to {info['max_boost']:g} dB), then inverse FFT and
             overlap-add back to a waveform; output level is matched to the input RMS
  Output     audio/output/  {info['n_outputs']} files named <input>__<preset>.wav
             figures/fig2_eq_gain_curves.png   gain curve of every preset
             figures/eq_summary.md             level change per band (low / mid / high) and speed
  Single file with one curve: python src/equalizer.py audio/input/speech.wav --curve data/elc_measured.json --level comfortable --play
""")


def run_microphone():
    import equalizer
    measured = measured_file()
    default_curve = (["--curve", measured, "--level", "comfortable", "--smooth", "0.5"] if measured
                     else ["--curve", "iso", "--phon", "40"])
    print("  Default: record 10 s, then apply "
          + ("my measured 'comfortable' curve (smoothed 0.5 octave)" if measured else "ISO 226 40 phon")
          + ", boost limited to 20 dB.")
    if yes("  Use the default settings?", True):
        argv = ["--record", "10"] + default_curve + ["--max-boost", "20"]
    else:
        argv = ["--record", ask("  Seconds to record", "10")] + curve_args()
    equalizer.main(argv)
    print("  Open the two files in audio/output/ (Finder, double click) to compare:")
    print("  mic__<curve>.wav is the equalized version, mic__<curve>_raw.wav is the original recording.")


def main():
    ensure_dirs()
    actions = {"1": run_listening_test, "2": run_equalizer, "3": run_microphone}
    while True:
        print(MENU)
        choice = ask("Choice").lower()
        if choice in ("q", "quit", "exit"):
            break
        if choice not in actions:
            print("  unknown choice")
            continue
        try:
            actions[choice]()
        except KeyboardInterrupt:
            print("\n  interrupted")
        except Exception as exc:  # keep the menu alive on any error
            print(f"  error: {exc}")


if __name__ == "__main__":
    main()
