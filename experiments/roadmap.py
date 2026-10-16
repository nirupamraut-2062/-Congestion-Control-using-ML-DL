"""Draw the project roadmap: the eight stages in the order they were done, with what each achieved.

Usage: python -m experiments.roadmap   ->   results/project_roadmap.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import INK, INK2, apply_style

RESULTS = Path(__file__).resolve().parent.parent / "results"

BUILD, FIND, RESPOND = "#e3f1ef", "#fbe7d9", "#e2f2e8"
EDGE = {BUILD: "#0f766e", FIND: "#c2410c", RESPOND: "#0b8a57"}

STAGES = [
    ("1", "Simulator and\nclassical baselines", "Reno and CUBIC double the RTT;\nCUBIC falls to 3.7 Mbps at 1% loss", BUILD),
    ("2", "ML: loss classifier\n(CUBIC+ML)", "91.4% accurate; 9.6 Mbps at 5% loss\n(CUBIC 1.6), still fair", BUILD),
    ("3", "DL: RTT forecaster\n(LSTM)", "12% lower error than\nrepeating the last value", BUILD),
    ("4", "RL: PPO agent", "98.8% of capacity at\n1.07 times the base RTT", BUILD),
    ("5", "Comparison of\nall six controllers", "Found: PPO gets 2% beside CUBIC\nand fails with 2 ms of jitter", FIND),
    ("6", "Fixes: PPO-Switch\nand PPO-J", "47% beside CUBIC;\n99% utilisation at 4 ms jitter", RESPOND),
    ("7", "Test on a real\nWi-Fi network", "ML and fairness fix carry over;\nlearned rate control does not", FIND),
    ("8", "Learning from the real\nnetwork (PPO-W)", "Trained on 29 min of recordings:\n0.3 to 5.8 Mbps on real Wi-Fi", RESPOND),
]


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 4.9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.9)
    ax.axis("off")
    w, h = 2.75, 1.75
    xs = [0.25 + i * 3.2 for i in range(4)]
    rows = [2.85, 0.35]
    centres = []
    for i, (num, title, result, fill) in enumerate(STAGES):
        row, col = divmod(i, 4)
        x = xs[col] if row == 0 else xs[3 - col]  # second row runs right to left
        y = rows[row]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=fill, edgecolor=EDGE[fill], linewidth=1.4))
        ax.text(x + 0.18, y + h - 0.22, f"Stage {num}", fontsize=9.5, color=EDGE[fill], fontweight="bold", va="top")
        ax.text(x + 0.18, y + h - 0.55, title, fontsize=11.5, color=INK, fontweight="bold", va="top", linespacing=1.15)
        ax.text(x + 0.18, y + 0.15, result, fontsize=9.5, color=INK2, va="bottom", linespacing=1.25)
        centres.append((x, y))
    for i in range(len(STAGES) - 1):
        (x0, y0), (x1, y1) = centres[i], centres[i + 1]
        if y0 == y1:
            start = (x0 + w, y0 + h / 2) if x1 > x0 else (x0, y0 + h / 2)
            end = (x1, y1 + h / 2) if x1 > x0 else (x1 + w, y1 + h / 2)
        else:
            start, end = (x0 + w / 2, y0), (x1 + w / 2, y1 + h)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, color=INK2, linewidth=1.3))
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=EDGE[c]) for c in (BUILD, FIND, RESPOND)]
    ax.legend(handles, ["Build a component", "Test it and find a weakness", "Respond to what was found"],
              loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "project_roadmap.png")
    plt.close(fig)
    print(f"saved {RESULTS / 'project_roadmap.png'}")


if __name__ == "__main__":
    main()
