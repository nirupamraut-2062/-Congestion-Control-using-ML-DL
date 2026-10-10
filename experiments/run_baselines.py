"""Run the classical baselines and save sanity plots and a summary table to results/.

Usage: python -m experiments.run_baselines
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import BASELINES, make
from sim import PKTS_PER_MBPS, Flow, Link, Simulator

from .metrics import flow_summary, jain_index
from .style import COLORS, INK2, MUTED, apply_style, label_line_ends

RESULTS = Path(__file__).resolve().parent.parent / "results"

BANDWIDTH_MBPS = 10.0
DELAY_MS = 20.0
QUEUE_PKTS = 50
WARMUP_S = 5.0
SETUP = f"{BANDWIDTH_MBPS:g} Mbps link, {2 * DELAY_MS:g} ms base RTT, {QUEUE_PKTS}-packet queue"


def link(loss_rate: float = 0.0) -> Link:
    return Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS, loss_rate)


def single_flow(duration: float = 30.0) -> list[dict]:
    """One flow alone on the link: shows each algorithm's characteristic window shape."""
    rows = []
    fig, axes = plt.subplots(2, len(BASELINES), figsize=(12, 5.5), sharex=True, sharey="row")
    pipe = link().rtt * BANDWIDTH_MBPS * PKTS_PER_MBPS
    for col, name in enumerate(BASELINES):
        flow = Flow(make(name))
        Simulator(link(), [flow]).run(duration)
        rows.append({"scenario": "single flow", "controller": name, **flow_summary(flow, WARMUP_S)})

        h = flow.history
        top, bottom = axes[0, col], axes[1, col]
        top.plot(h["t"], h["cwnd"], color=COLORS[name])
        top.axhline(pipe + QUEUE_PKTS, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        top.set_title(name)
        bottom.plot(h["t"], h["rtt_ms"], color=COLORS[name])
        bottom.axhline(2 * DELAY_MS, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        bottom.set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Congestion window (packets)")
    axes[1, 0].set_ylabel("Round-trip time (ms)")
    axes[0, 0].set_ylim(0, None)
    axes[1, 0].set_ylim(0, None)
    axes[0, 0].annotate("link + queue capacity", (duration, pipe + QUEUE_PKTS), xytext=(0, 4),
                        textcoords="offset points", ha="right", color=INK2, fontsize=8.5)
    axes[1, 0].annotate("base RTT (empty queue)", (duration, 2 * DELAY_MS), xytext=(0, -11),
                        textcoords="offset points", ha="right", color=INK2, fontsize=8.5)
    fig.suptitle(f"Single flow: congestion window and delay ({SETUP})", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "baselines_single_flow.png")
    plt.close(fig)
    return rows


def loss_sweep(duration: float = 30.0, seeds: int = 3) -> list[dict]:
    """Random (non-congestion) loss: loss-based algorithms mistake it for congestion."""
    loss_rates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(loss_rates))
    ends = {}
    for name in BASELINES:
        means = []
        for loss_rate in loss_rates:
            runs = []
            for seed in range(seeds):
                flow = Flow(make(name, seed))
                Simulator(link(loss_rate), [flow], seed=seed).run(duration)
                runs.append(flow_summary(flow, WARMUP_S))
            mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            means.append(mean["throughput_mbps"])
            rows.append({"scenario": f"random loss {100 * loss_rate:g}%", "controller": name, **mean})
        ax.plot(x, means, color=COLORS[name], marker="o", markeredgecolor=ax.get_facecolor(),
                markeredgewidth=1.5, label=name)
        ends[name] = means[-1]
    label_line_ends(ax, x[-1], ends, min_gap=0.05 * BANDWIDTH_MBPS)
    ax.set_xticks(x, [f"{100 * r:g}%" for r in loss_rates])
    ax.set_xlim(-0.3, len(loss_rates) - 0.2)
    ax.set_ylim(0, BANDWIDTH_MBPS * 1.08)
    ax.set_xlabel("Random packet loss rate")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title(f"Throughput under random loss, single flow\n{SETUP}; mean of {seeds} runs")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / "baselines_random_loss.png")
    plt.close(fig)
    return rows


def fairness(duration: float = 60.0, join_at: float = 10.0) -> list[dict]:
    """Two flows of the same algorithm; the second joins late and must be given room."""
    rows = []
    fig, axes = plt.subplots(1, len(BASELINES), figsize=(12, 3.8), sharey=True)
    for ax, name in zip(axes, BASELINES):
        flows = [Flow(make(name, 0), name="Flow 1"), Flow(make(name, 1), start=join_at, name="Flow 2")]
        Simulator(link(), flows).run(duration)
        stats = [flow_summary(f, duration / 2) for f in flows]
        jain = jain_index([s["throughput_mbps"] for s in stats])
        rows.append({
            "scenario": "two flows",
            "controller": name,
            "throughput_mbps": sum(s["throughput_mbps"] for s in stats),
            "avg_rtt_ms": float(np.mean([s["avg_rtt_ms"] for s in stats])),
            "p95_rtt_ms": max(s["p95_rtt_ms"] for s in stats),
            "loss_pct": float(np.mean([s["loss_pct"] for s in stats])),
            "jain_index": jain,
        })

        first, second = (f.history for f in flows)
        late = np.asarray(second["t"]) > join_at
        ax.plot(first["t"], first["throughput_mbps"], color=COLORS[name], label="Flow 1")
        ax.plot(np.asarray(second["t"])[late], np.asarray(second["throughput_mbps"])[late],
                color=INK2, linewidth=1.5, label=f"Flow 2 (joins at {join_at:g} s)")
        ax.set_title(f"{name}   Jain index {jain:.3f}")
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Throughput (Mbps)")
    axes[0].set_ylim(0, BANDWIDTH_MBPS * 1.08)
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(f"Two competing flows of the same algorithm ({SETUP})", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "baselines_fairness.png")
    plt.close(fig)
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    apply_style()
    rows = single_flow() + loss_sweep() + fairness()
    table = pd.DataFrame(rows).round(3)
    table.to_csv(RESULTS / "baselines_summary.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nSaved plots and baselines_summary.csv to {RESULTS}")


if __name__ == "__main__":
    main()
