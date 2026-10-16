"""Bring the real-network results together: simulator, recorded-Wi-Fi replay and real runs side by side.

Reads results/network_single.csv (latest real run), results/network_run1/network_single.csv (first
real run, if kept), results/comparison_wifi_trace.csv (replay of the recorded Wi-Fi) and the
recordings in data/wifi_traces/. Writes results/network_summary.csv, network_summary.png and
wifi_stall_pattern.png.

Usage: python -m net.summarise
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.style import COLORS, INK2, MUTED, apply_style

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TRACES = ROOT / "data" / "wifi_traces"
ORDER = ["Reno", "CUBIC", "BBR", "CUBIC+ML", "PPO", "CUBIC+PPO", "PPO-J", "PPO-W", "PPO-Switch"]
LINK_MBPS = 10.0


def summary_table() -> pd.DataFrame:
    real = pd.read_csv(RESULTS / "network_single.csv").set_index("controller")
    first = RESULTS / "network_run1" / "network_single.csv"
    first = pd.read_csv(first).set_index("controller") if first.exists() else None
    replay = pd.read_csv(RESULTS / "comparison_wifi_trace.csv").set_index("controller")
    rows = []
    for name in ORDER:
        if name not in real.index:
            continue
        rows.append({
            "controller": name,
            "simulator_mbps": real.loc[name, "sim_throughput_mbps"],
            "wifi_replay_mbps": LINK_MBPS * replay.loc[name, "utilisation_pct"] / 100 if name in replay.index else np.nan,
            "real_run1_mbps": first.loc[name, "real_throughput_mbps"] if first is not None and name in first.index else np.nan,
            "real_run2_mbps": real.loc[name, "real_throughput_mbps"],
            "real_run2_avg_rtt_ms": real.loc[name, "real_avg_rtt_ms"],
            "real_run2_loss_pct": real.loc[name, "real_loss_pct"],
        })
    return pd.DataFrame(rows)


def plot_summary(table: pd.DataFrame) -> None:
    names = list(table["controller"])
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    bars = [("simulator_mbps", "Simulator (clean link)", 0.22), ("wifi_replay_mbps", "Simulator replaying the recorded Wi-Fi", 0.5),
            ("real_run2_mbps", "Real Wi-Fi", 1.0)]
    h = 0.26
    for k, (column, label, alpha) in enumerate(bars):
        values = table[column].to_numpy()
        offset = (1 - k) * h
        ax.barh(y + offset, values, height=h * 0.92, color=[COLORS[n] for n in names], alpha=alpha,
                label=label, edgecolor="none")
        for yi, v in zip(y + offset, values):
            if not np.isnan(v):
                ax.annotate(f"{v:.1f}", (v, yi), xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=8, color=INK2 if k == 2 else MUTED)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 11.8)
    ax.set_xlabel("Throughput of one flow (Mbps of 10)")
    ax.set_title("Each controller alone: simulator, replay of the recorded Wi-Fi, real Wi-Fi\n"
                 "(bars from top: clean simulator, replay, real)", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    rtt = table["real_run2_avg_rtt_ms"].to_numpy()
    ax.barh(y, rtt, height=0.6, color=[COLORS[n] for n in names])
    for yi, v, loss in zip(y, rtt, table["real_run2_loss_pct"]):
        note = f"{v:.0f} ms" + (f"  ({loss:.0f}% loss)" if loss >= 5 else "")
        ax.annotate(note, (v, yi), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, names)
    ax.set_xlim(0, max(rtt) * 1.45)
    ax.axvline(40, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.set_xlabel("Average RTT on real Wi-Fi (ms); dashed line: 40 ms base")
    ax.set_title("Delay on real Wi-Fi", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("Real-network test: Windows laptop to MacBook over Wi-Fi, through a 10 Mbps bottleneck", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "network_summary.png")
    plt.close(fig)


def plot_stalls() -> None:
    files = sorted(TRACES.glob("*.csv"))
    if not files:
        return
    frame = pd.read_csv(files[0]).dropna()
    t, rtt = frame["t_s"].to_numpy(), frame["rtt_ms"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.4))
    window = (t >= 150) & (t <= 210)
    axes[0].plot(t[window], rtt[window], color=COLORS["PPO-J"], linewidth=0.5)
    axes[0].set_title("60 s of the recording: RTT of every packet sent at 5 Mbps over the bare Wi-Fi path", fontsize=10)
    axes[0].set_ylabel("RTT (ms)")
    zoom = (t >= 200.0) & (t <= 201.5)
    axes[1].plot(t[zoom] - 200, rtt[zoom], color=COLORS["PPO-J"], linewidth=1.2, marker=".", markersize=3)
    axes[1].set_title("1.5 s close-up: the link stalls for about 75 ms; packets sent during a stall wait for its end", fontsize=10)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("RTT (ms)")
    all_ms = pd.concat([pd.read_csv(f)["rtt_ms"].dropna() - pd.read_csv(f)["rtt_ms"].min() for f in files])
    minutes = sum(pd.read_csv(f).dropna()["t_s"].max() for f in files) / 60
    fig.suptitle(f"Recorded Wi-Fi path ({len(files)} recordings, {minutes:.0f} min): median extra delay "
                 f"{np.median(all_ms):.1f} ms, {100 * (all_ms > 20).mean():.0f}% of packets more than 20 ms late",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "wifi_stall_pattern.png")
    plt.close(fig)


def main() -> None:
    apply_style()
    table = summary_table()
    table.round(3).to_csv(RESULTS / "network_summary.csv", index=False)
    print(table.round(2).to_string(index=False))
    plot_summary(table)
    plot_stalls()
    print(f"\nSaved network_summary.csv, network_summary.png and wifi_stall_pattern.png to {RESULTS}")


if __name__ == "__main__":
    main()
