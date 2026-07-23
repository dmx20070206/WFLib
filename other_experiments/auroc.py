from pathlib import Path

import numpy as np
from utils import *


set_style_half_column()

SCRIPT_DIR = Path(__file__).resolve().parent

TASK_NAMES = [
    "Day0\n-> Day270",
    "Tor 0.4.8\n-> Tor 0.4.5",
    "SG\n-> DE",
    "Homepage\n-> Subpage",
]

METHOD_AUROC = {
    "K + 1": [17.94, 71.24, 4.68, 19.09],
    "MSP": [60.66, 89.60, 27.10, 68.56],
    "Entropy": [56.66, 91.70, 29.37, 65.00],
    "Free Energy": [70.33, 94.63, 52.21, 83.91],
}

METHOD_STYLES = {
    "K + 1": {"edgecolor": "#1f77b4", "hatch": "//"},
    "MSP": {"edgecolor": "#d62728", "hatch": "\\\\"},
    "Entropy": {"edgecolor": "#2ca02c", "hatch": "xx"},
    "Free Energy": {"edgecolor": "#9467bd", "hatch": ".."},
}


fig, ax = plt.subplots(figsize=(12, 5))

x_positions = np.arange(len(TASK_NAMES)) * 0.9
bar_width = 0.20
method_names = list(METHOD_AUROC.keys())
offsets = (np.arange(len(method_names)) - (len(method_names) - 1) / 2) * bar_width

for method_name, offset in zip(method_names, offsets):
    values = METHOD_AUROC[method_name]
    bars = ax.bar(
        x_positions + offset,
        values,
        width=bar_width,
        label=method_name,
        facecolor="white",
        linewidth=2.6,
        **METHOD_STYLES[method_name],
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

ax.set_ylabel("AUROC (%)", x=0.050, y=0.58)
ax.set_xticks(x_positions, TASK_NAMES)
ax.set_ylim(0, 105)
ax.tick_params(axis="x", which="minor", bottom=False, top=False)
ax.grid(axis="x", visible=False)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.22),
    ncol=4,
    fontsize=22,
    frameon=False,
)

plt.savefig(SCRIPT_DIR / "auroc.pdf", bbox_inches="tight", pad_inches=0.1)
