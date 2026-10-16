"""Shared look for every figure in the report."""
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# One colour per controller, used in every figure so a colour always means the same thing.
# The three classical and the three learned controllers are each distinguishable as a group
# (also for colour-blind readers); all six together are not, so line charts keep the two
# groups in separate panels and every mark is labelled.
COLORS = {
    "Reno": "#2a78d6", "CUBIC": "#eb6834", "BBR": "#1baf7a",
    "CUBIC+ML": "#eda100", "PPO": "#4a3aa7", "CUBIC+PPO": "#e87ba4",
    "PPO-Switch": "#9a7fe0", "PPO-J": "#1f6f8b", "PPO-W": "#0b8a57",
}


def label_line_ends(ax, x: float, ends: dict, min_gap: float, fmt: str = "{name}  {value:.1f}") -> None:
    """Name each line at its right-hand end, pushing labels apart where lines finish close together."""
    label_y = None
    for name, value in sorted(ends.items(), key=lambda kv: kv[1]):
        label_y = value if label_y is None else max(value, label_y + min_gap)
        ax.annotate(fmt.format(name=name, value=value), (x, label_y), xytext=(9, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "legend.frameon": False,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
    })
