"""Compare all six controllers and save tables and figures to results/.

Experiments (run all, or pick with --only):
  test      one flow on 30 random links the learned controllers never saw
  loss      throughput as random (non-congestion) loss increases
  fairness  sharing a link with a standard CUBIC flow, and with a copy of itself
  ood       conditions outside the range used for training
  trace     a link whose capacity changes every 10 s
  training  learning curves of the PPO policies
  variants  reward profiles and ablations of the PPO agent (those that have been trained)
  switching PPO-Switch, the fairness fix: PPO with a competitive mode
  jitter    random RTT jitter, as on Wi-Fi; PPO-J is PPO retrained with jitter
  wifi      the recorded Wi-Fi delay (data/wifi_traces/) replayed; PPO-W was trained on the rest of it

Usage: python -m experiments.run_comparison [--only test loss ...]
"""
import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl.agent import MODELS
from rl.env import sample_link
from sim import Flow, Link, Simulator
from sim.traces import Schedule

from .controllers import ALL, CLASSICAL, EXTRA, LEARNED, POLICIES, make
from .metrics import flow_summary, jain_index, link_summary, objective
from .style import COLORS, INK2, MUTED, apply_style, label_line_ends

RESULTS = Path(__file__).resolve().parent.parent / "results"
WARMUP_S = 5.0
TEST_SEED = 10_000
DASHED = (0, (4, 3))

# Fixed link used wherever a single representative network is needed.
BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS = 10.0, 20.0, 50
SETUP = f"{BANDWIDTH_MBPS:g} Mbps link, {2 * DELAY_MS:g} ms base RTT, {QUEUE_PKTS}-packet queue"

VARIANTS = {
    "Balanced, 1M decisions (main agent)": "ppo_rate",
    "Balanced, 500k decisions": "ppo_rate_500k",
    "Throughput profile": "ppo_throughput",
    "Low-latency profile": "ppo_lowlat",
    "No history (1 interval)": "ppo_nohist",
    "No loss-rate input": "ppo_noloss",
    "Trained with competing traffic": "ppo_rate_cross",
}


def run(link: Link, names: list[str], duration: float, starts=None, seed: int = 0):
    starts = starts or [0.0] * len(names)
    flows = [Flow(make(name, seed + i), start=start) for i, (name, start) in enumerate(zip(names, starts))]
    sim = Simulator(link, flows, seed=seed)
    sim.run(duration)
    return sim, flows


def save(fig, name: str) -> None:
    fig.savefig(RESULTS / name)
    plt.close(fig)


def write(table: pd.DataFrame, name: str) -> None:
    table.round(3).to_csv(RESULTS / name, index=False)
    print(f"\n{name}\n{table.round(3).to_string(index=False)}")


def horizontal_bars(ax, names, values, fmt="{:.0f}", colors=None) -> None:
    names, values = list(names)[::-1], list(values)[::-1]
    colors = colors[::-1] if colors else [COLORS[n] for n in names]
    ax.barh(names, values, height=0.6, color=colors)
    for name, value in zip(names, values):
        ax.annotate(fmt.format(value), (value, name), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


# ----------------------------------------------------------------------------- unseen links

def evaluate_on_test_links(names: list[str], n: int = 30, duration: float = 30.0) -> pd.DataFrame:
    """One flow per controller on the same n random links (drawn like training links, new seeds)."""
    rows = []
    for i in range(n):
        for name in names:
            link, _ = sample_link(random.Random(TEST_SEED + i))
            sim, (flow,) = run(link, [name], duration, seed=i)
            rows.append({"scenario": i, "controller": name, **link_summary(sim, flow, WARMUP_S)})
    runs = pd.DataFrame(rows)
    runs["objective"] = runs.apply(objective, axis=1)
    return runs


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    columns = ["utilisation_pct", "delay_ratio", "p95_delay_ratio", "loss_pct", "congestion_loss_pct", "objective"]
    return runs.groupby("controller", sort=False)[columns].mean().reset_index()


def test_links() -> None:
    runs = evaluate_on_test_links(ALL)
    runs.round(4).to_csv(RESULTS / "comparison_test_links_runs.csv", index=False)
    summary = summarise(runs)
    write(summary, "comparison_test_links.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for row in summary.itertuples():
        ax.scatter(row.delay_ratio, row.utilisation_pct, s=90, color=COLORS[row.controller],
                   edgecolor=ax.get_facecolor(), linewidth=1.5, zorder=3)
        ax.annotate(row.controller, (row.delay_ratio, row.utilisation_pct), xytext=(9, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9.5)
    ax.set_xlabel("Average RTT as a multiple of the base RTT (lower is better)")
    ax.set_ylabel("Link utilisation (%, higher is better)")
    ax.set_title(f"Throughput against delay on {runs['scenario'].nunique()} unseen links\n"
                 "one flow per link; mean over links; best is the top-left corner")
    ax.margins(x=0.15, y=0.15)
    fig.tight_layout()
    save(fig, "comparison_test_links.png")


# ----------------------------------------------------------------------------- random loss

def loss_sweep(duration: float = 30.0, seeds: int = 3) -> None:
    loss_rates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    rows = []
    for name in ALL:
        for loss_rate in loss_rates:
            stats = []
            for seed in range(seeds):
                link = Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS, loss_rate)
                sim, (flow,) = run(link, [name], duration, seed=seed)
                stats.append(link_summary(sim, flow, WARMUP_S))
            rows.append({"controller": name, "random_loss_pct": 100 * loss_rate,
                         **{k: float(np.mean([s[k] for s in stats])) for k in ("throughput_mbps", "delay_ratio")}})
    table = pd.DataFrame(rows)
    write(table.pivot(index="controller", columns="random_loss_pct", values="throughput_mbps")
          .loc[ALL].reset_index(), "comparison_random_loss.csv")

    x = np.arange(len(loss_rates))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, group, title in zip(axes, (CLASSICAL, LEARNED), ("Classical controllers", "Learned controllers")):
        ends = {}
        for name in group:
            y = table.loc[table["controller"] == name, "throughput_mbps"].to_numpy()
            ax.plot(x, y, color=COLORS[name], marker="o", markeredgecolor=ax.get_facecolor(),
                    markeredgewidth=1.5, label=name)
            ends[name] = y[-1]
        label_line_ends(ax, x[-1], ends, min_gap=0.05 * BANDWIDTH_MBPS)
        ax.set_xticks(x, [f"{100 * r:g}%" for r in loss_rates])
        ax.set_xlim(-0.3, len(loss_rates) - 0.1)
        ax.set_xlabel("Random packet loss rate")
        ax.set_title(title)
        ax.legend(loc="lower left")
    axes[0].set_ylim(0, BANDWIDTH_MBPS * 1.08)
    axes[0].set_ylabel("Throughput (Mbps)")
    fig.suptitle(f"Throughput under random loss, single flow ({SETUP}; mean of {seeds} runs)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "comparison_random_loss.png")


# ----------------------------------------------------------------------------- fairness

def fairness(duration: float = 60.0, join_at: float = 5.0, measure_from: float = 20.0) -> None:
    rows = []
    for name in ALL:
        shares = []
        for mine, names in enumerate(([name, "CUBIC"], ["CUBIC", name])):
            link = Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS)
            _, flows = run(link, names, duration, starts=[0.0, join_at])
            rates = [flow_summary(f, measure_from)["throughput_mbps"] for f in flows]
            shares.append(100 * rates[mine] / sum(rates))
        link = Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS)
        sim, flows = run(link, [name, name], duration, starts=[0.0, join_at])
        own = [link_summary(sim, f, measure_from) for f in flows]
        rows.append({
            "controller": name,
            "share_vs_cubic_pct": float(np.mean(shares)),
            "share_started_first_pct": shares[0],
            "share_joined_late_pct": shares[1],
            "self_jain_index": jain_index([s["throughput_mbps"] for s in own]),
            "self_utilisation_pct": sum(s["utilisation_pct"] for s in own),
            "self_delay_ratio": float(np.mean([s["delay_ratio"] for s in own])),
        })
    table = pd.DataFrame(rows)
    write(table, "comparison_fairness.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    horizontal_bars(axes[0], table["controller"], table["share_vs_cubic_pct"], "{:.0f}%")
    axes[0].axvline(50, color=MUTED, linewidth=1, linestyle=DASHED)
    axes[0].annotate("fair share", (50, 1), xycoords=("data", "axes fraction"), xytext=(0, 3),
                     textcoords="offset points", ha="center", color=INK2, fontsize=8.5)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Share of the link when competing with one CUBIC flow (%)")
    axes[0].set_title("Against standard CUBIC\n")
    horizontal_bars(axes[1], table["controller"], table["self_jain_index"], "{:.3f}")
    axes[1].set_xlim(0, 1.12)
    axes[1].set_xlabel("Jain fairness index between two flows of the same controller (1 = equal)")
    axes[1].set_title("Against a copy of itself\n")
    fig.suptitle(f"Fairness between two flows ({SETUP})", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "comparison_fairness.png")


# ----------------------------------------------------------------------------- outside the training range

def out_of_distribution(duration: float = 40.0, warmup: float = 10.0) -> None:
    scenarios = {
        "Faster link\n100 Mbps": Link(100, 20, 330),
        "Long delay\n400 ms RTT": Link(10, 200, 330),
        "Heavy random loss\n8%": Link(10, 20, 50, 0.08),
        "Tiny buffer\n5 packets": Link(20, 20, 5),
        "Slow link\n0.5 Mbps": Link(0.5, 20, 10),
    }
    rows = []
    for label, link in scenarios.items():
        for name in ALL:
            sim, (flow,) = run(link, [name], duration)
            rows.append({"scenario": label.replace("\n", ", "), "controller": name,
                         **link_summary(sim, flow, warmup)})
    table = pd.DataFrame(rows)
    write(table[["scenario", "controller", "utilisation_pct", "delay_ratio", "loss_pct", "congestion_loss_pct"]],
          "comparison_out_of_distribution.csv")

    fig, axes = plt.subplots(2, len(scenarios), figsize=(15, 5.8), sharey=True)
    for col, label in enumerate(scenarios):
        part = table[table["scenario"] == label.replace("\n", ", ")]
        horizontal_bars(axes[0, col], part["controller"], part["utilisation_pct"], "{:.0f}")
        axes[0, col].set_xlim(0, 118)
        axes[0, col].set_title(label)
        horizontal_bars(axes[1, col], part["controller"], part["delay_ratio"], "{:.2f}")
        axes[1, col].set_xlim(0, table["delay_ratio"].max() * 1.25)
    axes[0, 2].set_xlabel("Link utilisation (%)")
    axes[1, 2].set_xlabel("Average RTT as a multiple of the base RTT")
    fig.suptitle("Conditions outside the range the learned controllers were trained on", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "comparison_out_of_distribution.png")


# ----------------------------------------------------------------------------- changing capacity

def capacity_trace(duration: float = 50.0) -> None:
    steps = [(0, 10), (10, 4), (20, 14), (30, 7), (40, 12)]
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, ALL):
        sim, (flow,) = run(Link(Schedule(steps), DELAY_MS, QUEUE_PKTS), [name], duration)
        stats = link_summary(sim, flow, WARMUP_S)
        rows.append({"controller": name, **stats})
        ax.step(sim.history["t"], sim.history["capacity_mbps"], where="post", color=MUTED, linewidth=1.2,
                linestyle=DASHED, label="Link capacity")
        ax.plot(flow.history["t"], flow.history["throughput_mbps"], color=COLORS[name], linewidth=1.6,
                label="Throughput")
        ax.set_title(f"{name}\nuses {stats['utilisation_pct']:.0f}% of capacity, "
                     f"RTT {stats['delay_ratio']:.2f}x base", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mbps")
    axes[0, 0].set_ylim(0, 16)
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(f"Tracking a link whose capacity changes every 10 s ({2 * DELAY_MS:g} ms base RTT, "
                 f"{QUEUE_PKTS}-packet queue)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "comparison_capacity_trace.png")
    write(pd.DataFrame(rows)[["controller", "utilisation_pct", "delay_ratio", "p95_delay_ratio",
                              "congestion_loss_pct"]], "comparison_capacity_trace.csv")


# ----------------------------------------------------------------------------- training curves

def training_curves(window: int = 100) -> None:
    policies = {"PPO": "ppo_rate", "CUBIC+PPO": "ppo_hybrid"}
    panels = [("reward", "Reward per decision"), ("throughput", "Share of available capacity used"),
              ("delay", "Queueing delay / base RTT")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for name, file in policies.items():
        path = RESULTS / f"training_{file}.csv"
        if not path.exists():
            continue
        log = pd.read_csv(path)
        smooth = log.rolling(window, min_periods=window // 4).mean()
        for ax, (column, _) in zip(axes, panels):
            ax.plot(log["timesteps"] / 1000, smooth[column], color=COLORS[name], label=name)
    for ax, (_, title) in zip(axes, panels):
        ax.set_title(title)
        ax.set_xlabel("Training steps (thousands)")
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Learning curves (rolling mean over {window} training episodes)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "training_curves.png")


# ----------------------------------------------------------------------------- profiles and ablations

def variants() -> None:
    available = {label: file for label, file in VARIANTS.items() if (MODELS / f"{file}.zip").exists()}
    runs = evaluate_on_test_links([f"policy:{file}" for file in available.values()])
    summary = summarise(runs)
    summary["controller"] = list(available)
    summary = summary.rename(columns={"controller": "variant"})

    fair = []
    for file in available.values():
        shares = []
        for order in (0, 1):
            names = [f"policy:{file}", "CUBIC"][::1 if order == 0 else -1]
            _, flows = run(Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS), names, 60.0, starts=[0.0, 5.0])
            rates = [flow_summary(f, 20.0)["throughput_mbps"] for f in flows]
            shares.append(100 * rates[order] / sum(rates))
        fair.append(float(np.mean(shares)))
    summary["share_vs_cubic_pct"] = fair
    write(summary, "comparison_variants.csv")

    violet = COLORS["PPO"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    panels = [("utilisation_pct", "Link utilisation (%)", "{:.1f}"),
              ("delay_ratio", "Average RTT / base RTT", "{:.2f}"),
              ("objective", "Balanced objective (1 = ideal)", "{:.3f}")]
    for ax, (column, title, fmt) in zip(axes, panels):
        horizontal_bars(ax, summary["variant"], summary[column], fmt, colors=[violet] * len(summary))
        ax.set_xlim(0, summary[column].max() * 1.15)
        ax.set_title(title)
    fig.suptitle("PPO agent variants on the 30 unseen links: reward profiles and ablations", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "comparison_variants.png")


# ----------------------------------------------------------------------------- RTT jitter

def jitter(levels=(0.0, 1.0, 2.0, 4.0, 8.0), duration: float = 30.0, seeds: int = 2) -> None:
    """Single flow on the standard link with random return-path jitter, as on Wi-Fi."""
    names = [n for n in ALL + EXTRA if n not in POLICIES or (MODELS / f"{POLICIES[n]}.zip").exists()]
    rows = []
    for name in names:
        for ms in levels:
            stats = []
            for seed in range(seeds):
                link = Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS, jitter_ms=ms)
                sim, (flow,) = run(link, [name], duration, seed=seed)
                stats.append(link_summary(sim, flow, WARMUP_S))
            rows.append({"controller": name, "jitter_ms": ms,
                         **{k: float(np.mean([s[k] for s in stats])) for k in ("utilisation_pct", "delay_ratio")}})
    table = pd.DataFrame(rows)
    util = table.pivot(index="controller", columns="jitter_ms", values="utilisation_pct").loc[names].reset_index()
    delay = table.pivot(index="controller", columns="jitter_ms", values="delay_ratio").loc[names].reset_index()
    write(util, "comparison_jitter_utilisation.csv")
    write(delay, "comparison_jitter_delay.csv")

    groups = [("Classical controllers", [n for n in names if n in CLASSICAL]),
              ("Learned controllers", [n for n in names if n not in CLASSICAL])]
    x = np.arange(len(levels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (title, group) in zip(axes, groups):
        ends = {}
        for name in group:
            y = table.loc[table["controller"] == name, "utilisation_pct"].to_numpy()
            ax.plot(x, y, color=COLORS[name], marker="o", markeredgecolor=ax.get_facecolor(),
                    markeredgewidth=1.5, label=name)
            ends[name] = y[-1]
        label_line_ends(ax, x[-1], ends, min_gap=5, fmt="{name}  {value:.0f}%")
        ax.set_xticks(x, [f"{ms:g} ms" for ms in levels])
        ax.set_xlim(-0.3, len(levels) - 0.1)
        ax.set_xlabel("Mean random RTT jitter")
        ax.set_title(title)
        ax.legend(loc="lower left")
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Link utilisation (%)")
    fig.suptitle(f"Utilisation under RTT jitter, single flow ({SETUP}; mean of {seeds} runs)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "comparison_jitter.png")


# ----------------------------------------------------------------------------- recorded Wi-Fi

WIFI_TRACE = Path(__file__).resolve().parent.parent / "data" / "wifi_traces"


def wifi(duration: float = 30.0, starts=(0.0, 80.0, 160.0, 240.0, 320.0, 400.0)) -> None:
    """Single flow on the standard link with the recorded Wi-Fi delay replayed: the held-out part only."""
    from rl.env import TRACE_TRAIN_PART
    from sim.traces import DelayTrace, load_delay_trace

    if not any(WIFI_TRACE.glob("*.csv")):
        print(f"\nwifi: no recording at {WIFI_TRACE}; run net.run_network_test --only record first")
        return
    held_out = load_delay_trace(WIFI_TRACE, (TRACE_TRAIN_PART[1], 1.0))
    names = [n for n in ALL + EXTRA if n not in POLICIES or (MODELS / f"{POLICIES[n]}.zip").exists()]
    rows = []
    for name in names:
        stats = []
        for i, start in enumerate(starts):
            link = Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS, jitter_trace=DelayTrace(held_out, start))
            sim, (flow,) = run(link, [name], duration, seed=i)
            stats.append(link_summary(sim, flow, WARMUP_S))
        rows.append({"controller": name, **{k: float(np.mean([s[k] for s in stats]))
                                            for k in ("utilisation_pct", "delay_ratio", "loss_pct")}})
    table = pd.DataFrame(rows)
    write(table, "comparison_wifi_trace.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    horizontal_bars(ax, table["controller"], table["utilisation_pct"], "{:.0f}%")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Link utilisation (%)")
    ax.set_title(f"Recorded Wi-Fi delay replayed in the simulator (held-out part, {len(starts)} runs of {duration:g} s)\n"
                 f"{SETUP}", fontsize=10)
    fig.tight_layout()
    save(fig, "comparison_wifi_trace.png")


# ----------------------------------------------------------------------------- fairness fix

def switching() -> None:
    """PPO-Switch (PPO-J with a competitive mode) against PPO and PPO-J, with CUBIC+ML as a reference."""
    names = ["PPO", "PPO-J", "PPO-Switch", "CUBIC+ML"]
    rows = []
    for name in names:
        row = {"controller": name}
        for rival in ("CUBIC", "Reno", "BBR"):
            shares = []
            for mine, pair in enumerate(([name, rival], [rival, name])):
                _, flows = run(Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS), pair, 60.0, starts=[0.0, 5.0])
                rates = [flow_summary(f, 20.0)["throughput_mbps"] for f in flows]
                shares.append(100 * rates[mine] / sum(rates))
            row[f"share_vs_{rival.lower()}_pct"] = float(np.mean(shares))
            if rival == "CUBIC":
                row["share_vs_cubic_started_first_pct"], row["share_vs_cubic_joined_late_pct"] = shares
        shares = []
        for mine, pair in enumerate(([name, "CUBIC"], ["CUBIC", name])):
            _, flows = run(Link(BANDWIDTH_MBPS, 50.0, 100), pair, 60.0, starts=[0.0, 5.0])
            rates = [flow_summary(f, 20.0)["throughput_mbps"] for f in flows]
            shares.append(100 * rates[mine] / sum(rates))
        row["share_vs_cubic_100ms_rtt_pct"] = float(np.mean(shares))
        sim, flows = run(Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS), [name, name], 60.0, starts=[0.0, 5.0])
        own = [link_summary(sim, f, 20.0) for f in flows]
        row["self_jain_index"] = jain_index([s["throughput_mbps"] for s in own])
        row["self_delay_ratio"] = float(np.mean([s["delay_ratio"] for s in own]))

        runs = evaluate_on_test_links([name])
        row["test_utilisation_pct"] = runs["utilisation_pct"].mean()
        row["test_delay_ratio"] = runs["delay_ratio"].mean()
        row["test_objective"] = runs["objective"].mean()

        for loss in (0.01, 0.05):
            sim, (flow,) = run(Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS, loss), [name], 30.0)
            row[f"throughput_at_{100 * loss:g}pct_loss_mbps"] = link_summary(sim, flow, WARMUP_S)["throughput_mbps"]
        steps = [(0, 10), (10, 4), (20, 14), (30, 7), (40, 12)]
        sim, (flow,) = run(Link(Schedule(steps), DELAY_MS, QUEUE_PKTS), [name], 50.0)
        trace = link_summary(sim, flow, WARMUP_S)
        row["trace_utilisation_pct"] = trace["utilisation_pct"]
        row["trace_p95_delay_ratio"] = trace["p95_delay_ratio"]
        sim, (flow,) = run(Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS, jitter_ms=4.0), [name], 30.0)
        row["utilisation_with_4ms_jitter_pct"] = link_summary(sim, flow, WARMUP_S)["utilisation_pct"]
        rows.append(row)
    table = pd.DataFrame(rows)
    write(table, "comparison_switching.csv")

    # Mode switches on the 30 unseen links, where every flow is alone: any switch is a false alarm.
    false_alarms = 0
    for i in range(30):
        link, _ = sample_link(random.Random(TEST_SEED + i))
        _, (flow,) = run(link, ["PPO-Switch"], 30.0, seed=i)
        false_alarms += bool(flow.cc.switches)
    print(f"\nPPO-Switch changed mode on {false_alarms} of 30 single-flow test links")

    # A CUBIC flow joins at 20 s and leaves at 40 s.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, name in zip(axes, ["PPO", "PPO-Switch"]):
        flows = [Flow(make(name), start=0.0), Flow(make("CUBIC"), start=20.0, stop=40.0)]
        sim = Simulator(Link(BANDWIDTH_MBPS, DELAY_MS, QUEUE_PKTS), flows)
        sim.run(60.0)
        for start, stop in _tcp_spans(getattr(flows[0].cc, "switches", []), 60.0):
            ax.axvspan(start, stop, color=COLORS["CUBIC+ML"], alpha=0.12, linewidth=0)
        for flow in flows:
            ax.plot(flow.history["t"], flow.history["throughput_mbps"], color=COLORS[flow.name], linewidth=1.6,
                    label=flow.name)
        share = flow_summary(flows[0], 25.0, 40.0)["throughput_mbps"]
        ax.set_title(f"{name}: {share:.1f} of {BANDWIDTH_MBPS:g} Mbps while CUBIC is present", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", fontsize=8.5)
    axes[0].set_ylabel("Throughput (Mbps)")
    axes[0].set_ylim(0, BANDWIDTH_MBPS * 1.25)
    fig.suptitle(f"A CUBIC flow joins at 20 s and leaves at 40 s ({SETUP}); "
                 "shading marks PPO-Switch's competitive mode", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "comparison_switching_timeline.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    for ax, rival in zip(axes, ("CUBIC", "Reno", "BBR")):
        horizontal_bars(ax, table["controller"], table[f"share_vs_{rival.lower()}_pct"], "{:.0f}%")
        ax.axvline(50, color=MUTED, linewidth=1, linestyle=DASHED)
        ax.set_xlim(0, 100)
        ax.set_title(f"Against one {rival} flow")
        ax.set_xlabel("Share of the link (%)")
    fig.suptitle(f"Share of the link beside one other flow, mean of both start orders ({SETUP})", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "comparison_switching.png")


def _tcp_spans(switches, end: float):
    spans, start = [], None
    for t, mode in switches:
        if mode == "tcp":
            start = t
        elif start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, end))
    return spans


EXPERIMENTS = {
    "test": test_links,
    "loss": loss_sweep,
    "fairness": fairness,
    "ood": out_of_distribution,
    "trace": capacity_trace,
    "training": training_curves,
    "variants": variants,
    "switching": switching,
    "jitter": jitter,
    "wifi": wifi,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    apply_style()
    for name in args.only:
        EXPERIMENTS[name]()
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
