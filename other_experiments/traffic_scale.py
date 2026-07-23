from pathlib import Path

import numpy as np
from matplotlib.ticker import FixedLocator
from utils import *


set_style_half_column()

SCRIPT_DIR = Path(__file__).resolve().parent

INSTANCE_COUNTS = [10, 20, 40, 80, 120, 160, 200]
UNMONITORED_RATIOS = [0, 50, 100, 150, 200, 250, 300]

INSTANCE_PERFORMANCE = {
    "F1-score": [60.84, 69.85, 70.80, 73.83, 76.98, 79.14, 78.56],
    "Precision": [69.53, 76.80, 76.13, 77.14, 77.33, 79.58, 79.07],
    "Recall": [56.08, 69.43, 69.00, 72.27, 77.63, 79.59, 79.10],
}

UNMONITORED_PERFORMANCE = {
    "F1-score": [78.28, 80.40, 78.80, 77.00, 79.19, 79.57, 77.30],
    "Precision": [80.29, 81.55, 79.31, 78.09, 79.08, 79.60, 77.22],
    "Recall": [77.61, 80.02, 79.51, 77.47, 79.99, 80.57, 78.29],
}

METRIC_NAMES = ("F1-score", "Precision", "Recall")

METRIC_STYLES = {
    "F1-score": {"color": "#1f77b4", "linestyle": "-", "marker": "s"},
    "Precision": {"color": "#d62728", "linestyle": "--", "marker": "o"},
    "Recall": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
}


def set_equal_width_x_axis(ax, tick_labels):
    x_positions = np.arange(len(tick_labels))

    ax.set_xticks(x_positions, labels=tick_labels)
    ax.xaxis.set_minor_locator(
        FixedLocator([
            left + i / 5
            for left in x_positions[:-1]
            for i in range(1, 5)
        ])
    )
    ax.tick_params(axis="x", which="minor", width=0.6)
    return x_positions


def plot_metric_group(ax, tick_labels, metric_series, xlabel):
    x_positions = set_equal_width_x_axis(ax, tick_labels)

    for metric_name in METRIC_NAMES:
        ax.plot(
            x_positions,
            metric_series[metric_name],
            label=metric_name,
            markerfacecolor="white",
            **METRIC_STYLES[metric_name],
        )

    all_values = [
        value
        for values in metric_series.values()
        for value in values
    ]
    y_min = min(all_values)
    y_max = max(all_values)
    y_margin = (y_max - y_min) * 0.18
    ax.set_ylim(y_min - y_margin * 0.6, y_max + y_margin)
    ax.set_xlabel(xlabel)


fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2), sharey=True)

plot_metric_group(
    axes[0],
    INSTANCE_COUNTS,
    INSTANCE_PERFORMANCE,
    "# of Instances per Website",
)
left_y_limits = axes[0].get_ylim()
plot_metric_group(
    axes[1],
    UNMONITORED_RATIOS,
    UNMONITORED_PERFORMANCE,
    "Ratio of Unmonitored Traffic (%)",
)
axes[0].set_ylim(left_y_limits)
axes[1].tick_params(axis="y", which="both", labelleft=False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    ncol=3,
    fontsize=22,
    frameon=False,
)
fig.supylabel("Performance (%)", x=0.050, y=0.58)
fig.subplots_adjust(top=0.78, wspace=0.24)

output_path = SCRIPT_DIR / "traffic_scale.pdf"
plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
