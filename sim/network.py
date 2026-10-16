"""Time-stepped simulator of senders sharing a single bottleneck link.

Topology: N senders -> drop-tail FIFO queue -> bottleneck link -> receivers.
Packets are fixed-size and ACKs return over an uncongested reverse path, so the
round-trip time of a packet is 2 * link delay + time spent in the queue.

A congestion controller is any object exposing `cwnd` (packets), `pacing_rate`
(packets/s, or None for pure ACK clocking), `on_ack(Ack)` and `on_loss(Loss)`.
It may also define `on_tick(now)`, called once per time step, for timer-driven logic.
"""
from __future__ import annotations

import heapq
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Union

PACKET_BYTES = 1500
PKTS_PER_MBPS = 1e6 / (8 * PACKET_BYTES)

CONGESTION = "congestion"
RANDOM = "random"


@dataclass(slots=True)
class Ack:
    now: float
    rtt: float
    send_time: float
    delivery_rate: float  # packets/s delivered while this packet was in flight
    inflight: int


@dataclass(slots=True)
class Loss:
    now: float
    send_time: float
    cause: str  # ground truth for datasets; controllers must not read it


@dataclass
class Link:
    bandwidth_mbps: Union[float, Callable[[float], float]] = 10.0
    delay_ms: float = 20.0  # one-way propagation delay
    queue_pkts: int = 50
    loss_rate: float = 0.0  # probability of non-congestion (e.g. wireless) loss
    jitter_ms: float = 0.0  # mean random extra delay on the return path (see Simulator.JITTER_HOLD)
    jitter_trace: Callable[[float], float] | None = None  # extra delay replayed from a recording of a real path
    # Where the recorded delay applies. "both": a stall holds back data on its way to the bottleneck
    # and ACKs on their way back, each until the stall ends, as on Wi-Fi (the recorded delay is the
    # time left until the stall ends, so nothing is counted twice). "forward" or "return": one side only.
    trace_side: str = "both"

    def rate_pps(self, t: float) -> float:
        bw = self.bandwidth_mbps
        return (bw(t) if callable(bw) else bw) * PKTS_PER_MBPS

    @property
    def rtt(self) -> float:
        return 2 * self.delay_ms / 1000


class Flow:
    def __init__(self, controller, start: float = 0.0, stop: float = float("inf"), name: str | None = None):
        self.cc = controller
        self.start = start
        self.stop = stop
        self.name = name or getattr(controller, "name", type(controller).__name__)

        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.lost_random = 0
        self.rtt_sum = 0.0
        self.inflight = 0

        self._tokens = 0.0
        self._on_tick = getattr(controller, "on_tick", None)
        self._last_ack = None  # arrival time of the most recent ACK
        self._acks = deque()  # (arrival_time, send_time, acked_at_send, last_ack_at_send)
        self._losses = []  # heap of (arrival_time, seq, send_time, cause)
        self._rec = (0, 0, 0, 0, 0.0)
        self.history = {k: [] for k in ("t", "cwnd", "sent", "lost", "lost_random", "throughput_mbps", "rtt_ms")}


class Simulator:
    # Jitter, as on Wi-Fi, comes in bursts: a random extra delay (exponential, mean link.jitter_ms)
    # is drawn every JITTER_HOLD seconds and shared by all packets leaving the queue in that time.
    # ACKs never overtake each other. The extra delay does not depend on how fast anyone sends.
    JITTER_HOLD = 0.002

    def __init__(self, link: Link, flows: list[Flow], dt: float = 0.001, seed: int = 0,
                 record_interval: float | None = 0.1):
        self.link = link
        self.flows = flows
        self.dt = dt
        self.now = 0.0
        self.rng = random.Random(seed)
        self.queue = deque()  # (flow, send_time, acked_at_send, last_ack_at_send)
        self.history = {k: [] for k in ("t", "queue", "capacity_mbps")}

        self._tick = 0
        self._credit = 0.0
        self._first = 0
        self._seq = 0
        self._record_ticks = round(record_interval / dt) if record_interval else 0
        self._jitter = 0.0
        self._jitter_until = -1.0
        self._ingress = deque()  # (time it reaches the queue, packet): data held back on the forward path

    def run(self, duration: float) -> None:
        for _ in range(round(duration / self.dt)):
            self.step()

    def step(self) -> None:
        now, dt, link, queue = self.now, self.dt, self.link, self.queue
        rate = link.rate_pps(now)
        flows = self.flows
        n = len(flows)
        forward = link.jitter_trace is not None and link.trace_side in ("forward", "both")
        backward = link.jitter_trace is not None and link.trace_side in ("return", "both")

        for f in flows:
            self._deliver_feedback(f, now)
            if f._on_tick is not None:
                f._on_tick(now)

        # Rotate which flow sends first so none gets a standing advantage at the queue.
        for k in range(n):
            f = flows[(self._first + k) % n]
            if not f.start <= now < f.stop:
                continue
            cc = f.cc
            budget = int(cc.cwnd) - f.inflight
            if cc.pacing_rate is not None:
                per_tick = cc.pacing_rate * dt
                f._tokens = min(f._tokens + per_tick, per_tick + 1.0)
                budget = min(budget, int(f._tokens))
                if budget > 0:
                    f._tokens -= budget
            for _ in range(budget):
                f.sent += 1
                f.inflight += 1
                if link.loss_rate and self.rng.random() < link.loss_rate:
                    self._drop(f, RANDOM, rate)
                elif forward:
                    ready = now + link.jitter_trace(now)
                    if self._ingress:
                        ready = max(ready, self._ingress[-1][0])  # the path delivers in order
                    self._ingress.append((ready, (f, now, f.acked, now if f._last_ack is None else f._last_ack)))
                elif len(queue) >= link.queue_pkts:
                    self._drop(f, CONGESTION, rate)
                else:
                    queue.append((f, now, f.acked, now if f._last_ack is None else f._last_ack))
        self._first = (self._first + 1) % n
        ingress = self._ingress
        while ingress and ingress[0][0] <= now:
            packet = ingress.popleft()[1]
            if len(queue) >= link.queue_pkts:
                self._drop(packet[0], CONGESTION, rate)
            else:
                queue.append(packet)

        # Departures get exact sub-tick timestamps so RTT and delivery-rate samples
        # are not quantised to the tick length.
        credit0 = self._credit
        self._credit += rate * dt
        served = 0
        while self._credit >= 1.0 and queue:
            f, send_time, acked_at_send, last_ack_at_send = queue.popleft()
            self._credit -= 1.0
            served += 1
            depart = now + max(0.0, (served - credit0) / rate)
            arrival = depart + link.rtt
            if link.jitter_ms:
                if depart >= self._jitter_until:
                    self._jitter = self.rng.expovariate(1000 / link.jitter_ms)
                    self._jitter_until = depart + self.JITTER_HOLD
                arrival += self._jitter
            if backward:
                arrival += link.jitter_trace(depart)
            if (link.jitter_ms or backward) and f._acks:
                arrival = max(arrival, f._acks[-1][0])  # ACKs never overtake each other
            f._acks.append((arrival, send_time, acked_at_send, last_ack_at_send))
        if not queue:
            self._credit = min(self._credit, 1.0)

        self._tick += 1
        self.now = self._tick * dt
        if self._record_ticks and self._tick % self._record_ticks == 0:
            self._record(rate)

    def _drop(self, f: Flow, cause: str, rate: float) -> None:
        # The sender notices a loss when feedback for the packets behind it arrives.
        queue_delay = len(self.queue) / max(rate, 1e-9)
        self._seq += 1
        heapq.heappush(f._losses, (self.now + self.link.rtt + queue_delay, self._seq, self.now, cause))

    def _deliver_feedback(self, f: Flow, now: float) -> None:
        acks = f._acks
        while acks and acks[0][0] <= now:
            arrival, send_time, acked_at_send, last_ack_at_send = acks.popleft()
            f.inflight -= 1
            f.acked += 1
            f._last_ack = arrival
            rtt = arrival - send_time
            f.rtt_sum += rtt
            rate = (f.acked - acked_at_send) / (arrival - last_ack_at_send)
            f.cc.on_ack(Ack(arrival, rtt, send_time, rate, f.inflight))
        losses = f._losses
        while losses and losses[0][0] <= now:
            arrival, _, send_time, cause = heapq.heappop(losses)
            f.inflight -= 1
            f.lost += 1
            if cause == RANDOM:
                f.lost_random += 1
            f.cc.on_loss(Loss(arrival, send_time, cause))

    def _record(self, rate: float) -> None:
        span = self._record_ticks * self.dt
        for f in self.flows:
            sent0, acked0, lost0, random0, rtt0 = f._rec
            acked = f.acked - acked0
            h = f.history
            h["t"].append(self.now)
            h["cwnd"].append(f.cc.cwnd)
            h["sent"].append(f.sent - sent0)
            h["lost"].append(f.lost - lost0)
            h["lost_random"].append(f.lost_random - random0)
            h["throughput_mbps"].append(acked / span / PKTS_PER_MBPS)
            h["rtt_ms"].append(1000 * (f.rtt_sum - rtt0) / acked if acked else float("nan"))
            f._rec = (f.sent, f.acked, f.lost, f.lost_random, f.rtt_sum)
        self.history["t"].append(self.now)
        self.history["queue"].append(len(self.queue))
        self.history["capacity_mbps"].append(rate / PKTS_PER_MBPS)
