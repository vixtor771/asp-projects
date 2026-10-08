"""Project 2, Magnitude Spectrum Inversion: run everything from here.

    python main.py                          # all experiments, then the final reconstruction (about 16 minutes)
    python main.py experiments              # all experiments only (about 15 minutes)
    python main.py experiments window       # one or more experiments: scaling, window, overlap, lookahead, iterations
    python main.py final                    # final reconstruction of all 12 clips only (about 1 minute)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))   # all other code lives in src/

import experiments
import final


def main() -> None:
    args = sys.argv[1:]
    command = args[0] if args else "all"
    if command == "all":
        experiments.run([])
        final.run()
    elif command == "experiments":
        unknown = [name for name in args[1:] if name not in experiments.EXPERIMENTS]
        if unknown:
            sys.exit(f"Unknown experiment {unknown[0]!r}; choose from {', '.join(experiments.EXPERIMENTS)}")
        experiments.run(args[1:])
    elif command == "final":
        final.run()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
