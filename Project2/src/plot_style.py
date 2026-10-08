"""Shared matplotlib style so all figures look the same."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed colors: one per method or setting, always in this order
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#52514e"

METHOD_COLOR = {"rtisi": BLUE, "rtisi_la": ORANGE}
METHOD_LABEL = {"rtisi": "RTISI", "rtisi_la": "RTISI-LA"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "lines.linewidth": 2,
    "lines.markersize": 5,
    "legend.frameon": False,
})
