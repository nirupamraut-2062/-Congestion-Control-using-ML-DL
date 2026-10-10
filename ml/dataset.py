"""Generate training data by running the baselines over many random network scenarios.

Writes two files to data/:
  loss_events.csv  one row per lost packet: sender-side features + the true cause
  timeseries.csv   per-flow RTT / throughput sampled every 100 ms (for the forecaster)

Usage: python -m ml.dataset [--runs 400] [--duration 30]
"""
import argparse
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from baselines import BASELINES, make
from sim import CONGESTION, PKTS_PER_MBPS, Flow, Link, Simulator
from sim.traces import StepBandwidth

from .features import FEATURES, FlowMonitor

DATA = Path(__file__).resolve().parent.parent / "data"
MAX_EVENTS_PER_CLASS = 200  # per run, so fast lossy links do not dominate the dataset


class Recorder:
    """Wraps a controller and logs the feature vector at every loss it is told about."""

    def __init__(self, inner, rows):
        self.inner = inner
        self.name = inner.name
        self.monitor = FlowMonitor()
        self.rows = rows

    @property
    def cwnd(self):
        return self.inner.cwnd

    @property
    def pacing_rate(self):
        return self.inner.pacing_rate

    def on_ack(self, ack):
        self.monitor.on_ack(ack)
        self.inner.on_ack(ack)

    def on_loss(self, loss):
        self.rows.append((self.name, *self.monitor.features(loss.now), int(loss.cause == CONGESTION)))
        self.monitor.on_loss(loss)
        self.inner.on_loss(loss)


def log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def run_scenario(args):
    run_id, duration = args
    rng = random.Random(run_id)
    bandwidth = log_uniform(rng, 1, 30)
    delay_ms = rng.uniform(5, 80)
    pipe = bandwidth * PKTS_PER_MBPS * 2 * delay_ms / 1000
    queue = max(5, round(pipe * log_uniform(rng, 0.1, 3)))
    loss_rate = 0.0 if rng.random() < 0.15 else log_uniform(rng, 1e-4, 3e-2)
    varying = rng.random() < 0.4
    capacity = StepBandwidth(bandwidth, duration, rng) if varying else bandwidth

    events = []
    flows = []
    for i in range(rng.choice([1, 1, 2, 3])):
        controller = make(rng.choice(list(BASELINES)), seed=rng.randrange(1 << 30))
        start = 0.0 if i == 0 else rng.uniform(0, 10)
        flows.append(Flow(Recorder(controller, events), start=start))
    Simulator(Link(capacity, delay_ms, queue, loss_rate), flows, seed=run_id).run(duration)

    scenario = {"run_id": run_id, "bandwidth_mbps": bandwidth, "delay_ms": delay_ms, "queue_pkts": queue,
                "random_loss_rate": loss_rate, "n_flows": len(flows), "varying_bandwidth": int(varying)}

    by_class = {0: [], 1: []}
    for row in events:
        by_class[row[-1]].append(row)
    kept = [r for rows in by_class.values() for r in rng.sample(rows, min(len(rows), MAX_EVENTS_PER_CLASS))]
    losses = pd.DataFrame(kept, columns=["controller", *FEATURES, "congestion"]).assign(**scenario)

    series = []
    for flow_id, flow in enumerate(flows):
        frame = pd.DataFrame(flow.history)
        frame = frame[frame["t"] > flow.start]
        series.append(frame.assign(run_id=run_id, flow_id=flow_id, controller=flow.name,
                                   base_rtt_ms=2 * delay_ms))
    return losses, pd.concat(series)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=400)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run_scenario, [(i, args.duration) for i in range(args.runs)], chunksize=4))

    losses = pd.concat([r[0] for r in results], ignore_index=True)
    series = pd.concat([r[1] for r in results], ignore_index=True)
    losses.to_csv(DATA / "loss_events.csv", index=False)
    series.to_csv(DATA / "timeseries.csv", index=False)

    counts = losses["congestion"].value_counts()
    print(f"{args.runs} runs -> {len(losses)} loss events "
          f"({counts.get(1, 0)} congestion, {counts.get(0, 0)} random), {len(series)} time-series rows")
    print(f"Saved to {DATA}")


if __name__ == "__main__":
    main()
