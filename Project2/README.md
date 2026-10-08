# Project 2: Magnitude Spectrum Inversion (RTISI and RTISI-LA)

Rebuilds audio from its STFT magnitude only, following Zhu et al. 2007, Sections II and III.

## Report

[Project2_Report.pdf](Project2_Report.pdf), source: [Project2_Report.md](Project2_Report.md)

## Run

```bash
cd Project2
python main.py                       # everything: experiments, then final reconstruction (about 16 minutes)
python main.py experiments           # scaling check and parameter experiments only
python main.py experiments window    # one or more experiments: scaling, window, overlap, lookahead, iterations
python main.py final                 # final reconstruction of all 12 clips only
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: runs the experiments and the final reconstruction |
| `src/experiments.py` | Window scaling check, then parameter experiments: window length, overlap and asymmetric window, look-ahead, iterations |
| `src/final.py` | Final run of RTISI and RTISI-LA on all clips with the chosen settings |
| `src/common.py` | Paths, clip list with speech or music label, audio loading and saving |
| `src/stft.py` | Scaled Hamming window, STFT, STFT magnitude, overlap-add inverse, SER |
| `src/rtisi.py` | RTISI |
| `src/rtisi_la.py` | RTISI-LA with the asymmetric analysis window |
| `src/evaluate.py` | Runs one setting on one clip and measures SER, loudness change and speed |
| `src/plot_style.py` | Shared figure style |
| `audio/input/` | The 12 given clips |
| `audio/output/` | Reconstructed clips, `<clip>_rtisi.wav` and `<clip>_rtisi_la.wav` |
| `figures/` | Figures as PNG and PDF; number tables are printed in the terminal |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
