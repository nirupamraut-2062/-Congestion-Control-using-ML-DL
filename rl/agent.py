"""Controllers driven by a learned policy that takes one decision per monitor interval.

During an interval the controller only gathers statistics from ACKs and losses. When
the interval ends it summarises them into a few scale-free features, appends them to a
short history (the policy's observation), and applies the policy's action.

The same class is used in training (the Gym environment supplies the actions) and in
deployment (a saved policy supplies them from `on_tick`), so there is no gap between
what the agent was trained on and what it runs on.
"""
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from baselines.cubic import Cubic
from sim import PKTS_PER_MBPS

MODELS = Path(__file__).resolve().parent.parent / "models"

FEATURES = [
    "latency_gradient",  # d(RTT)/dt across the interval: positive while a queue builds
    "latency_ratio",     # mean RTT / minimum RTT - 1: how much queueing delay there is
    "send_ratio",        # packets sent / packets acknowledged - 1: sending faster than the link drains
    "loss_rate",         # share of packets reported lost in the interval
]
LOW = np.array([-2.0, 0.0, -1.0, 0.0])
HIGH = np.array([2.0, 5.0, 5.0, 1.0])


class IntervalAgent:
    name = "agent"
    MIN_INTERVAL = 0.1
    MAX_INTERVAL = 1.0
    RTTS_PER_INTERVAL = 1.5
    STEP = 0.3  # an action of +1 / -1 scales the rate or window by e^0.3 (about 35% up, 26% down)

    def __init__(self, policy=None, history: int = 8, features=FEATURES, rtt_filter: str = "mean"):
        # "mean": the latency signals use the mean RTT of each half interval (the original agent).
        # "min": they use the minimum RTT instead, like Copa's RTTstanding. Jitter only ever adds
        # delay, so it inflates the mean but hardly the minimum, while a real queue raises both.
        self.policy = policy
        self.rtt_filter = rtt_filter
        self._columns = [FEATURES.index(f) for f in features]
        self.history = deque([np.zeros(len(features))] * history, maxlen=history)
        self.min_rtt = math.inf
        self.srtt = 0.0
        self.inflight = 0
        self.interval_end = None
        self.last_action = 0.0
        self._latency_ratio = 0.0
        self._start = 0.0
        self._mid = math.inf
        self._reset_counters()

    def _reset_counters(self):
        self._acked = self._lost = self._n1 = self._n2 = 0
        self._rtt1 = self._rtt2 = 0.0
        self._min1 = self._min2 = math.inf
        self._inflight0 = self.inflight

    def observation(self) -> np.ndarray:
        return np.concatenate(self.history).astype(np.float32)

    def on_ack(self, ack):
        rtt = ack.rtt
        self.min_rtt = min(self.min_rtt, rtt)
        self.srtt = rtt if self.srtt == 0.0 else 0.875 * self.srtt + 0.125 * rtt
        self.inflight = ack.inflight
        self._acked += 1
        if ack.now < self._mid:
            self._n1 += 1
            self._rtt1 += rtt
            self._min1 = min(self._min1, rtt)
        else:
            self._n2 += 1
            self._rtt2 += rtt
            self._min2 = min(self._min2, rtt)

    def on_loss(self, loss):
        self._lost += 1
        self.inflight -= 1

    def on_tick(self, now):
        if self.policy is None:
            return
        if self.interval_end is None:
            if self.srtt:
                self.start_interval(now)
        elif now >= self.interval_end:
            self.finish_interval(now)
            self.apply(float(self.policy(self.observation())))
            self.start_interval(now)

    def start_interval(self, now: float) -> None:
        length = min(max(self.RTTS_PER_INTERVAL * self.srtt, self.MIN_INTERVAL), self.MAX_INTERVAL)
        self._start = now
        self._mid = now + length / 2
        self.interval_end = now + length
        self._reset_counters()

    def finish_interval(self, now: float) -> dict:
        """Close the interval: push its features onto the history and return its raw statistics."""
        duration = now - self._start
        acked, lost = self._acked, self._lost
        sent = max(acked + lost + self.inflight - self._inflight0, 0)
        mean_rtt = (self._rtt1 + self._rtt2) / acked if acked else None
        robust = self.rtt_filter == "min"
        if acked:
            level = min(self._min1, self._min2) if robust else mean_rtt
            self._latency_ratio = level / self.min_rtt - 1.0
        gradient = 0.0
        if self._n1 and self._n2:
            if robust:
                gradient = (self._min2 - self._min1) / (duration / 2)
            else:
                gradient = (self._rtt2 / self._n2 - self._rtt1 / self._n1) / (duration / 2)
        row = np.array([
            gradient,
            self._latency_ratio,
            sent / acked - 1.0 if acked else HIGH[2],
            lost / (acked + lost) if acked + lost else 0.0,
        ])
        self.history.append(np.clip(row, LOW, HIGH)[self._columns])
        return {"duration": duration, "acked": acked, "lost": lost, "mean_rtt": mean_rtt}

    def apply(self, action: float) -> None:
        raise NotImplementedError


class RateAgent(IntervalAgent):
    """Pure learned control: the policy raises or lowers the sending rate directly."""

    name = "PPO"
    MIN_RATE = 0.2 * PKTS_PER_MBPS
    MAX_RATE = 500 * PKTS_PER_MBPS

    def __init__(self, policy=None, initial_rate_mbps: float = 1.0, **kwargs):
        super().__init__(policy, **kwargs)
        self.cwnd = 1e9  # rate-based: the window never limits sending
        self.pacing_rate = initial_rate_mbps * PKTS_PER_MBPS

    def apply(self, action: float) -> None:
        self.last_action = min(max(action, -1.0), 1.0)
        rate = self.pacing_rate * math.exp(self.STEP * self.last_action)
        self.pacing_rate = min(max(rate, self.MIN_RATE), self.MAX_RATE)


class HybridAgent(IntervalAgent):
    """CUBIC runs on every ACK; once per interval the policy scales CUBIC's window up or down."""

    name = "CUBIC+PPO"

    def __init__(self, policy=None, **kwargs):
        super().__init__(policy, **kwargs)
        self.inner = Cubic()
        self.pacing_rate = None

    @property
    def cwnd(self):
        return self.inner.cwnd

    def on_ack(self, ack):
        super().on_ack(ack)
        self.inner.on_ack(ack)

    def on_loss(self, loss):
        super().on_loss(loss)
        self.inner.on_loss(loss)

    def apply(self, action: float) -> None:
        self.last_action = min(max(action, -1.0), 1.0)
        inner = self.inner
        inner.cwnd = max(inner.cwnd * math.exp(self.STEP * self.last_action), inner.MIN_CWND)
        inner.epoch_start = None  # CUBIC restarts its growth curve from the corrected window
        if self.last_action < 0:
            inner.ssthresh = min(inner.ssthresh, inner.cwnd)


AGENTS = {"rate": RateAgent, "hybrid": HybridAgent}


def load_policy(name: str):
    """Load the saved policy `name`: returns (observation -> action function, its config)."""
    from stable_baselines3 import PPO

    config = json.loads((MODELS / f"{name}.json").read_text())
    model = PPO.load(MODELS / f"{name}.zip", device="cpu")

    def policy(observation):
        return model.predict(observation, deterministic=True)[0][0]

    return policy, config


def load_agent(name: str):
    """Return a zero-argument factory that builds a controller running the saved policy `name`."""
    policy, config = load_policy(name)
    cls = AGENTS[config["mode"]]
    return lambda: cls(policy=policy, history=config["history"], features=config["features"],
                       rtt_filter=config.get("rtt_filter", "mean"))
