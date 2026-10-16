"""Gymnasium environment: each episode is one agent-controlled flow on a freshly sampled link.

Observation: the agent's history of per-interval features (see rl/agent.py).
Action:      one number in [-1, 1]; scales the sending rate (or CUBIC's window) up or down.
Reward:      throughput - delay_weight * queueing delay - loss_weight * congestion loss,
             each normalised so that links of different speed and delay are comparable.
"""
import math
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from baselines import Cubic, Reno
from sim import PKTS_PER_MBPS, Flow, Link, Simulator
from sim.traces import DelayTrace, StepBandwidth, load_delay_trace

from .agent import AGENTS, FEATURES

# name -> (delay weight, loss weight); throughput always has weight 1.
PROFILES = {
    "balanced": (0.5, 5.0),
    "throughput": (0.1, 2.0),
    "low_latency": (2.0, 5.0),
}
MAX_DELAY_RATIO = 3.0
JITTER_SHARE = 0.75  # with jitter training on, share of episodes whose link has jitter
TRACE_SHARE = 0.5    # with a recorded trace, share of episodes that replay it (others: synthetic or none)
TRACE_TRAIN_PART = (0.0, 0.7)  # the rest of the recording is kept for testing
MAX_EPISODE_S = 200.0


def log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_link(rng) -> tuple[Link, float]:
    """A random training link and its nominal bandwidth in Mbps."""
    bandwidth = log_uniform(rng, 2, 40)
    delay_ms = rng.uniform(5, 75)
    pipe = bandwidth * PKTS_PER_MBPS * 2 * delay_ms / 1000
    queue = max(10, round(pipe * log_uniform(rng, 0.2, 3)))
    loss_rate = 0.0 if rng.random() < 0.5 else log_uniform(rng, 1e-4, 3e-2)
    capacity = StepBandwidth(bandwidth, MAX_EPISODE_S, rng) if rng.random() < 0.5 else bandwidth
    return Link(capacity, delay_ms, queue, loss_rate), bandwidth


class CongestionEnv(gym.Env):
    def __init__(self, mode: str = "rate", profile: str = "balanced", cross_traffic: float = 0.0,
                 history: int = 8, features=FEATURES, episode_steps: int = 200, jitter_ms: float = 0.0,
                 rtt_filter: str = "mean", jitter_trace: str | None = None, queue_reward: bool = False):
        self.mode = mode
        # Score delay by the queue at the bottleneck instead of the RTT. Jitter and Wi-Fi stalls add
        # delay that no sender can remove; counting it teaches the agent to send less for nothing.
        # Only the reward uses this (the simulator knows the queue); the agent's inputs are unchanged.
        self.queue_reward = queue_reward
        self.trace = load_delay_trace(jitter_trace, TRACE_TRAIN_PART) if jitter_trace else None
        self.jitter_ms = jitter_ms
        self.rtt_filter = rtt_filter
        self.delay_weight, self.loss_weight = PROFILES[profile]
        self.cross_traffic = cross_traffic
        self.history = history
        self.features = list(features)
        self.episode_steps = episode_steps
        self.observation_space = spaces.Box(-10.0, 10.0, (history * len(self.features),), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = random.Random(int(self.np_random.integers(1 << 31)))
        self.link, bandwidth = sample_link(rng)
        if self.trace is not None and rng.random() < TRACE_SHARE:
            # Replay the recorded real path from a random point, its delays stretched or shrunk a little.
            self.link.jitter_trace = DelayTrace(self.trace, rng.uniform(0, len(self.trace) * DelayTrace.STEP),
                                                log_uniform(rng, 0.5, 1.5))
        elif self.jitter_ms and rng.random() < JITTER_SHARE:
            # Domain randomisation: RTT samples on real paths jitter by a few ms (Wi-Fi, OS timers),
            # which a policy trained only on exact RTTs mistakes for a growing queue.
            self.link.jitter_ms = log_uniform(rng, 0.3, self.jitter_ms)
        self.agent = AGENTS[self.mode](history=self.history, features=self.features, rtt_filter=self.rtt_filter)
        if self.mode == "rate":
            # Start anywhere from far below to above capacity so both ramp-up and back-off are learned.
            self.agent.pacing_rate = bandwidth * PKTS_PER_MBPS * log_uniform(rng, 0.05, 1.5)
        self.flow = Flow(self.agent)
        self.cross = []
        if rng.random() < self.cross_traffic:
            self.cross.append(Flow(rng.choice([Cubic, Reno])(), start=rng.uniform(0, 10)))
        self.sim = Simulator(self.link, [self.flow, *self.cross], seed=rng.randrange(1 << 31),
                             record_interval=None)
        while not self.agent.srtt and self.sim.now < 5.0:
            self.sim.step()

        self._steps = 0
        self._totals = np.zeros(4)
        self._run_interval()
        return self.agent.observation(), {}

    def step(self, action):
        self.agent.apply(float(action[0]))
        reward, parts = self._score(self._run_interval())
        self._steps += 1
        self._totals += (reward, *parts)
        truncated = self._steps >= self.episode_steps
        info = {}
        if truncated:
            mean = self._totals / self._steps
            info["metrics"] = dict(zip(("reward", "throughput", "delay", "loss"), mean.tolist()))
        return self.agent.observation(), reward, False, truncated, info

    def _run_interval(self) -> dict:
        agent, sim, flow = self.agent, self.sim, self.flow
        congestion_lost = flow.lost - flow.lost_random
        agent.start_interval(sim.now)
        queued = steps = 0
        while sim.now < agent.interval_end:
            sim.step()
            queued += len(sim.queue)
            steps += 1
        stats = agent.finish_interval(sim.now)
        stats["congestion_lost"] = flow.lost - flow.lost_random - congestion_lost
        stats["mean_queue"] = queued / max(steps, 1)
        return stats

    def _score(self, stats: dict):
        now = self.sim.now
        active = 1 + sum(f.start <= now for f in self.cross)
        fair_share = self.link.rate_pps(now - stats["duration"] / 2) / active
        throughput = min(stats["acked"] / stats["duration"] / fair_share, 1.0)
        delay = 0.0
        if self.queue_reward:
            queue_delay = stats["mean_queue"] / self.link.rate_pps(now - stats["duration"] / 2)
            delay = min(queue_delay / self.link.rtt, MAX_DELAY_RATIO)
        elif stats["mean_rtt"] is not None:
            delay = min(max(stats["mean_rtt"] / self.link.rtt - 1.0, 0.0), MAX_DELAY_RATIO)
        # Only drops the sender caused count against it; random link loss is not its fault.
        loss = stats["congestion_lost"] / max(stats["acked"] + stats["lost"], 1)
        reward = throughput - self.delay_weight * delay - self.loss_weight * loss
        return reward, (throughput, delay, loss)
