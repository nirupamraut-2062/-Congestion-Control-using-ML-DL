"""Real-network test: the project's controllers over Wi-Fi between two laptops.

On the second laptop:   python3 receiver.py
On this laptop:         python -m net.run_network_test --receiver <the address it prints>

The packets cross the real Wi-Fi and both operating systems' network stacks. To compare
with the simulator on equal terms, the receiver adds the simulator's standard bottleneck
in front of itself: a 10 Mbps drop-tail queue of 50 packets, and extra delay so that the
base RTT is about 40 ms.

Experiments (run all, or pick with --only):
  path      RTT of the bare Wi-Fi path and through the emulated bottleneck
  single    each controller alone for 30 s, next to the simulator's result for the same link
  loss      1% random loss added at the receiver
  fairness  two flows sharing the bottleneck: CUBIC with CUBIC, and PPO-W and PPO-Switch with CUBIC
  record    (only when asked for) log the bare Wi-Fi path's RTT for 2 minutes, for trace-driven
            training: a new file in data/wifi_traces/, and results/network_jitter_trace.png

Results go to results/network_*.csv and .png; every run's 100 ms history to results/network_runs/.
Usage: python -m net.run_network_test --receiver 192.168.1.23 [--only single fairness] [--quick]
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.controllers import ALL, EXTRA, available, make
from experiments.metrics import flow_summary
from experiments.style import COLORS, INK2, MUTED, apply_style
from sim import Flow, Link, Simulator

from .receiver import PORT
from .sender import RealFlow, configure, measure_rtt, receiver_stats, record_path, run_flows

RESULTS = Path(__file__).resolve().parent.parent / "results"
TRACES = Path(__file__).resolve().parent.parent / "data" / "wifi_traces"  # one file per recording
RUNS = RESULTS / "network_runs"
RATE_MBPS, QUEUE_PKTS, TARGET_RTT_MS = 10.0, 50, 40.0
WARMUP_S = 5.0


SINGLE = available(ALL + EXTRA)
LOSSY = available(["CUBIC", "CUBIC+ML", "BBR", "PPO", "PPO-J", "PPO-W", "PPO-Switch"])
PAIRS = [p for p in [("CUBIC", "CUBIC"), ("PPO-W", "CUBIC"), ("CUBIC", "PPO-W"),
                     ("PPO-Switch", "CUBIC"), ("CUBIC", "PPO-Switch")] if available(p) == list(p)]


class Test:
    def __init__(self, receiver: str, scale: float, port: int = PORT):
        self.receiver = receiver
        self.port = port
        self.scale = scale
        self.run_id = 0
        self.path_file = RESULTS / "network_path.json"
        self.path = json.loads(self.path_file.read_text()) if self.path_file.exists() else None
        if self.path and self.path.get("receiver") != receiver:
            self.path = None  # measured for another receiver

    # ------------------------------------------------------------------ helpers

    def bottleneck(self, loss: float = 0.0) -> dict:
        if self.path is None:
            self.measure_path()
        return {"rate_mbps": RATE_MBPS, "queue": QUEUE_PKTS, "delay_ms": self.path["added_delay_ms"], "loss": loss}

    def run(self, names, duration, starts=None, loss=0.0, label="run"):
        """One real run. Returns the flows and the receiver's per-flow counts."""
        starts = starts or [0.0] * len(names)
        configure(self.receiver, **self.bottleneck(loss), seed=self.run_id, port=self.port)
        base = 100 * (self.run_id + 1)
        flows = [RealFlow(make(name, seed=i), self.receiver, base + i, start=start, name=name, port=self.port)
                 for i, (name, start) in enumerate(zip(names, starts))]
        run_flows(flows, duration * self.scale)
        stats = receiver_stats(self.receiver, self.port)
        self.run_id += 1
        RUNS.mkdir(parents=True, exist_ok=True)
        frames = [pd.DataFrame(f.history).assign(flow=f"{i}:{f.name}") for i, f in enumerate(flows)]
        pd.concat(frames).to_csv(RUNS / f"{label}.csv", index=False)
        counts = [stats.get(f.id, [0, 0, 0]) for f in flows]
        print(f"  {label}: " + ", ".join(
            f"{f.name} {flow_summary(f, WARMUP_S * self.scale)['throughput_mbps']:.2f} Mbps" for f in flows))
        return flows, counts

    def simulate(self, names, duration, starts=None, loss=0.0):
        link = Link(RATE_MBPS, self.path["base_rtt_ms"] / 2, QUEUE_PKTS, loss)
        flows = [Flow(make(name, seed=i), start=s) for i, (name, s) in enumerate(zip(names, starts or [0.0] * len(names)))]
        Simulator(link, flows).run(duration)
        return flows

    # ------------------------------------------------------------------ experiments

    def measure_path(self) -> None:
        configure(self.receiver, port=self.port)  # no bottleneck: the bare Wi-Fi path
        wifi = measure_rtt(self.receiver, port=self.port)
        if len(wifi) < 10:
            raise ConnectionError(f"Only {len(wifi)} of 50 test packets came back; the path is not usable.")
        wifi_ms = 1000 * np.array(wifi)
        # Fill up to the target base RTT: Wi-Fi RTT + 1.2 ms to send one packet at 10 Mbps + added delay.
        added = max(TARGET_RTT_MS - float(np.median(wifi_ms)) - 1.2, 0.0)
        configure(self.receiver, RATE_MBPS, QUEUE_PKTS, added, port=self.port)
        shaped_ms = 1000 * np.array(measure_rtt(self.receiver, port=self.port))
        self.path = {
            "receiver": self.receiver,
            "wifi_rtt_ms_min": float(wifi_ms.min()), "wifi_rtt_ms_median": float(np.median(wifi_ms)),
            "wifi_rtt_ms_p95": float(np.percentile(wifi_ms, 95)), "wifi_replies": len(wifi_ms),
            "added_delay_ms": added,
            "base_rtt_ms": float(np.median(shaped_ms)), "base_rtt_ms_p95": float(np.percentile(shaped_ms, 95)),
        }
        RESULTS.mkdir(exist_ok=True)
        self.path_file.write_text(json.dumps(self.path, indent=2))
        print("\nPath:", json.dumps(self.path, indent=2))

    def record(self, seconds: float = 120.0, rate_mbps: float = 5.0) -> None:
        configure(self.receiver, port=self.port)  # no emulated bottleneck: the bare Wi-Fi path
        print(f"  sending {rate_mbps:g} Mbps for {seconds:g} s; keep both laptops still and idle")
        times, rtts = record_path(self.receiver, seconds, rate_mbps, self.port)
        if np.isnan(rtts).all():
            raise ConnectionError("No packet came back; check the receiver and the address.")
        TRACES.mkdir(parents=True, exist_ok=True)
        trace = TRACES / f"trace_{time.strftime('%Y%m%d_%H%M%S')}_{rate_mbps:g}mbps_{seconds:g}s.csv"
        pd.DataFrame({"t_s": times.round(6), "rtt_ms": (1000 * rtts).round(3)}).to_csv(trace, index=False)

        ok = ~np.isnan(rtts)
        ms = 1000 * rtts[ok]
        excess = ms - ms.min()
        spikes = excess > 20
        starts = np.flatnonzero(spikes[1:] & ~spikes[:-1])
        summary = {
            "file": trace.name, "receiver": self.receiver, "seconds": seconds, "rate_mbps": rate_mbps,
            "packets_sent": int(len(rtts)), "packets_lost_pct": round(100 * float((~ok).mean()), 3),
            "rtt_ms_min": round(float(ms.min()), 2), "rtt_ms_median": round(float(np.median(ms)), 2),
            "rtt_ms_p95": round(float(np.percentile(ms, 95)), 2), "rtt_ms_p99": round(float(np.percentile(ms, 99)), 2),
            "rtt_ms_max": round(float(ms.max()), 2),
            "share_above_min_plus_20ms_pct": round(100 * float(spikes.mean()), 2),
            "spikes_per_second": round(len(starts) / seconds, 2),
        }
        (RESULTS / "network_jitter_trace.json").write_text(json.dumps(summary, indent=2))
        print(f"  saved {trace}  ({len(list(TRACES.glob('*.csv')))} recordings in total)")
        print("  " + json.dumps(summary, indent=2).replace("\n", "\n  "))

        fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
        axes[0].plot(times[ok], ms, color=COLORS["PPO-J"], linewidth=0.6)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Round-trip time (ms)")
        axes[0].set_title("RTT of every packet over the bare Wi-Fi path")
        axes[1].hist(excess, bins=np.logspace(-1, np.log10(max(excess.max(), 1.0) + 1), 50), color=COLORS["PPO-J"])
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Delay above the minimum RTT (ms)")
        axes[1].set_ylabel("Packets")
        axes[1].set_title("How often each extra delay occurs")
        fig.suptitle(f"Recorded Wi-Fi path: median RTT {summary['rtt_ms_median']:.1f} ms, 95th percentile "
                     f"{summary['rtt_ms_p95']:.1f} ms, {summary['packets_lost_pct']:.2f}% lost", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(RESULTS / "network_jitter_trace.png")
        plt.close(fig)

    def single(self, duration: float = 30.0) -> None:
        rows = []
        for name in SINGLE:
            (flow,), ((received, queue_drops, _),) = self.run([name], duration, label=f"single_{name}")
            real = flow_summary(flow, WARMUP_S * self.scale)
            (sim_flow,) = self.simulate([name], duration)
            sim = flow_summary(sim_flow, WARMUP_S)
            rows.append({
                "controller": name,
                "real_throughput_mbps": real["throughput_mbps"], "sim_throughput_mbps": sim["throughput_mbps"],
                "real_avg_rtt_ms": real["avg_rtt_ms"], "sim_avg_rtt_ms": sim["avg_rtt_ms"],
                "real_p95_rtt_ms": real["p95_rtt_ms"], "sim_p95_rtt_ms": sim["p95_rtt_ms"],
                "real_loss_pct": real["loss_pct"], "sim_loss_pct": sim["loss_pct"],
                "queue_drops": queue_drops,
            })
        table = pd.DataFrame(rows)
        write(table, "network_single.csv")
        paired_bars(table, [("throughput_mbps", "Throughput (Mbps)"), ("avg_rtt_ms", "Average RTT (ms)")],
                    f"Each controller alone over Wi-Fi with a {RATE_MBPS:g} Mbps, {QUEUE_PKTS}-packet bottleneck "
                    f"(base RTT {self.path['base_rtt_ms']:.0f} ms): real network against the simulator",
                    "network_single.png")

    def loss(self, duration: float = 30.0, rate: float = 0.01) -> None:
        rows = []
        for name in LOSSY:
            (flow,), ((_, _, random_drops),) = self.run([name], duration, loss=rate, label=f"loss_{name}")
            (sim_flow,) = self.simulate([name], duration, loss=rate)
            rows.append({
                "controller": name,
                "real_throughput_mbps": flow_summary(flow, WARMUP_S * self.scale)["throughput_mbps"],
                "sim_throughput_mbps": flow_summary(sim_flow, WARMUP_S)["throughput_mbps"],
                "real_avg_rtt_ms": flow_summary(flow, WARMUP_S * self.scale)["avg_rtt_ms"],
                "sim_avg_rtt_ms": flow_summary(sim_flow, WARMUP_S)["avg_rtt_ms"],
                "random_drops": random_drops,
            })
        table = pd.DataFrame(rows)
        write(table, "network_loss.csv")
        paired_bars(table, [("throughput_mbps", "Throughput (Mbps)")],
                    f"Throughput with {100 * rate:g}% random loss added at the receiver: real network against the simulator",
                    "network_loss.png")

    def fairness(self, duration: float = 60.0, join_at: float = 5.0, measure_from: float = 20.0) -> None:
        rows, timelines = [], {}
        for first, second in PAIRS:
            label = f"fair_{first}_then_{second}"
            flows, counts = self.run([first, second], duration, starts=[0.0, join_at * self.scale], label=label)
            stats = [flow_summary(f, measure_from * self.scale) for f in flows]
            total = sum(s["throughput_mbps"] for s in stats) or 1.0
            sim_flows = self.simulate([first, second], duration, starts=[0.0, join_at])
            sim_stats = [flow_summary(f, measure_from)["throughput_mbps"] for f in sim_flows]
            rows.append({
                "first": first, "second": second,
                "real_share_first_pct": 100 * stats[0]["throughput_mbps"] / total,
                "sim_share_first_pct": 100 * sim_stats[0] / (sum(sim_stats) or 1.0),
                "real_total_mbps": total,
                "real_avg_rtt_ms": float(np.nanmean([s["avg_rtt_ms"] for s in stats])),
                "queue_drops": sum(c[1] for c in counts),
            })
            if second == "CUBIC" and first != "CUBIC":
                timelines[first] = flows
        table = pd.DataFrame(rows)
        write(table, "network_fairness.csv")

        # Share of each controller beside one CUBIC flow, averaged over the start orders run.
        summary = []
        for name in [n for n in ("CUBIC", "PPO-W", "PPO-Switch") if n in SINGLE]:
            real, sim = [], []
            for r in rows:
                if r["first"] == name and r["second"] == "CUBIC":
                    real.append(r["real_share_first_pct"])
                    sim.append(r["sim_share_first_pct"])
                elif r["second"] == name and r["first"] == "CUBIC" and name != "CUBIC":
                    real.append(100 - r["real_share_first_pct"])
                    sim.append(100 - r["sim_share_first_pct"])
            summary.append({"controller": name, "real_share_vs_cubic_pct": float(np.mean(real)),
                            "sim_share_vs_cubic_pct": float(np.mean(sim))})
        write(pd.DataFrame(summary), "network_fairness_summary.csv")

        if timelines:
            fig, axes = plt.subplots(1, len(timelines), figsize=(6.5 * len(timelines), 4), sharey=True, squeeze=False)
            for ax, (name, flows) in zip(axes[0], timelines.items()):
                for flow in flows:
                    ax.plot(flow.history["t"], flow.history["throughput_mbps"], color=COLORS[flow.name],
                            linewidth=1.4, label=flow.name)
                ax.set_title(f"{name} starts, CUBIC joins at {join_at * self.scale:g} s", fontsize=10)
                ax.set_xlabel("Time (s)")
                ax.legend(loc="upper right", fontsize=8.5)
            axes[0, 0].set_ylabel("Throughput (Mbps)")
            axes[0, 0].set_ylim(0, RATE_MBPS * 1.3)
            fig.suptitle(f"Two flows over the real Wi-Fi path through a {RATE_MBPS:g} Mbps bottleneck", x=0.01, ha="left")
            fig.tight_layout()
            fig.savefig(RESULTS / "network_fairness.png")
            plt.close(fig)


def write(table: pd.DataFrame, name: str) -> None:
    table.round(3).to_csv(RESULTS / name, index=False)
    print(f"\n{name}\n{table.round(2).to_string(index=False)}")


def paired_bars(table: pd.DataFrame, panels, title: str, name: str) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4), squeeze=False)
    y = np.arange(len(table))[::-1]
    for ax, (metric, label) in zip(axes[0], panels):
        real, sim = table[f"real_{metric}"].to_numpy(), table[f"sim_{metric}"].to_numpy()
        colors = [COLORS[c] for c in table["controller"]]
        ax.barh(y + 0.2, real, height=0.38, color=colors, label="Real network")
        ax.barh(y - 0.2, sim, height=0.38, color=colors, alpha=0.35, label="Simulator")
        for yi, r, s in zip(y, real, sim):
            ax.annotate(f"{r:.1f}", (r, yi + 0.2), xytext=(4, 0), textcoords="offset points", va="center",
                        color=INK2, fontsize=8.5)
            ax.annotate(f"{s:.1f}", (s, yi - 0.2), xytext=(4, 0), textcoords="offset points", va="center",
                        color=MUTED, fontsize=8.5)
        ax.set_yticks(y, table["controller"])
        ax.set_xlim(0, np.nanmax(np.concatenate([real, sim])) * 1.18)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(title + "\n(solid bars: real network, faded bars: simulator)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / name)
    plt.close(fig)


def main() -> None:
    global RESULTS, RUNS, TRACES
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--receiver", required=True, help="IP address printed by receiver.py on the other laptop")
    parser.add_argument("--only", nargs="*", choices=["path", "single", "loss", "fairness", "record"],
                        default=["path", "single", "loss", "fairness"])
    parser.add_argument("--quick", action="store_true", help="half-length runs, for a first try")
    parser.add_argument("--port", type=int, default=PORT, help="UDP port of the receiver (default 9000)")
    parser.add_argument("--record-seconds", type=float, default=120.0, help="length of the record step")
    parser.add_argument("--record-rate", type=float, default=5.0, help="sending rate of the record step (Mbps)")
    parser.add_argument("--out", type=Path, default=RESULTS, help="folder for the results (default: results/)")
    args = parser.parse_args()
    if args.out != RESULTS:
        TRACES = args.out / "wifi_traces"  # a test run must not add to the training recordings
    RESULTS, RUNS = args.out, args.out / "network_runs"
    RESULTS.mkdir(parents=True, exist_ok=True)
    apply_style()
    for name in SINGLE:
        make(name)  # load the saved models before any timed run
    test = Test(args.receiver, 0.5 if args.quick else 1.0, args.port)
    for name in args.only:
        print(f"\n=== {name} ===")
        if name == "path":
            test.measure_path()
        elif name == "record":
            test.record(args.record_seconds, args.record_rate)
        else:
            getattr(test, name)()
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
