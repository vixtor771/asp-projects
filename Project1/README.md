# Project 1: Fletcher-Munson Curves of Equal Loudness

## Run

```bash
cd Project1
python main.py
```

| Menu | What it does | Output |
|---|---|---|
| 1 | Listening test with headphones (threshold + 4 equal-loudness curves); each run is saved as a new file, or plot an existing one | `data/elc_measured*.json`, `figures/fig1_<name>.png`, `figures/<name>_table.md` |
| 2 | Generates test audio (speech, music, pink noise, sweep) if missing and runs the equalizer on all files with two curves (ISO 226 at 40 phon, and my measured comfortable curve) | `audio/output/*.wav`, `figures/fig2_eq_gain_curves.png`, `figures/eq_summary.md` |
| 3 | Records 10 s from the microphone, equalizes the recording with the default curve (or your own settings), saves it | `audio/output/mic__<curve>.wav` (equalized) and `mic__<curve>_raw.wav` (original) |

## Deliverables

| Requirement | File |
|---|---|
| ISO 226 overview (1 page) | submitted on Canvas (not in this repo) |
| My equal-loudness curves | `figures/fig1_elc_measured.png`, data in `data/elc_measured.json` (later runs get a timestamp) |
| Equalizer | `src/equalizer.py`, results in `audio/output/` |
| Report (3 pages) | submitted on Canvas (not in this repo) |

## Files

| File | Purpose |
|---|---|
| `main.py` | Menu (start here) |
| `src/elc_measure.py` | Listening test (adaptive staircase for threshold, method of adjustment for curves) |
| `src/equalizer.py` | STFT equalizer: ELC to gain curve, file mode, live microphone mode |
| `src/iso226.py` | ISO 226:2003 formula and log-frequency interpolation |
| `src/make_test_audio.py` | Test sounds |
| `src/plot_elc.py`, `src/analyze_eq.py` | Figures and summary tables |
| `src/utils.py` | Shared helpers |
| `data/`, `audio/`, `figures/` | Measurement results, input and output audio, figures and tables |

Every script in `src/` also works on its own from the command line, for example:

```bash
python src/equalizer.py audio/input/speech.wav --curve data/elc_measured.json --level comfortable --smooth 0.5 --play
python src/equalizer.py --record 10 --phon 50 --strength 0.5    # record, equalize, save
python src/equalizer.py --mic 10 --phon 50                      # live monitoring through headphones
python src/elc_measure.py --quick
```

Use `-h` on any script for all options.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
