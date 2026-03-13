"""
Academic plot style settings.
Usage:
    from academic_style import apply_style, COLORS, MARKERS
    apply_style()
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Color cycle (colorblind-friendly) ─────────────────────────────────────────
COLORS = [
    "#0173B2",  # blue
    "#DE8F05",  # orange
    "#029E73",  # green
    "#D55E00",  # vermilion
    "#CC78BC",  # purple
    "#CA9161",  # tan
]

MARKERS = ["o", "s", "^", "D", "v", "P"]

# ── rcParams ───────────────────────────────────────────────────────────────────
STYLE = {
    # Font
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    # Lines / markers
    "lines.linewidth": 1.4,
    "lines.markersize": 3,
    # Axes
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_style():
    """Apply academic rcParams globally."""
    mpl.rcParams.update(STYLE)
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLORS)
