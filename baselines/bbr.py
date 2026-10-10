import random
from collections import deque

from .base import Controller

STARTUP, DRAIN, PROBE_BW, PROBE_RTT = "STARTUP", "DRAIN", "PROBE_BW", "PROBE_RTT"


class BBR(Controller):
    """BBR-style controller (simplified BBRv1).

    Estimates the bottleneck bandwidth (windowed max of delivery rate) and the
    propagation delay (windowed min of RTT), then paces at that bandwidth instead
    of reacting to packet loss.
    """

    name = "BBR"
    STARTUP_GAIN = 2.89
    CYCLE = (1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    CWND_GAIN = 2.0
    BW_WINDOW_ROUNDS = 10
    MIN_RTT_WINDOW = 10.0
    PROBE_RTT_TIME = 0.2
    PROBE_RTT_CWND = 4.0

    def __init__(self, seed=0):
        super().__init__()
        self.state = STARTUP
        self.btl_bw = 0.0
        self.min_rtt = float("inf")
        self.min_rtt_stamp = 0.0
        self.round = 0
        self.round_start = 0.0

        self._bw_samples = deque()  # (round, rate) with decreasing rate
        self._full_bw = 0.0
        self._full_bw_rounds = 0
        self._cycle_idx = 0
        self._cycle_stamp = 0.0
        self._probe_rtt_done = 0.0
        self._rng = random.Random(seed)

    def on_ack(self, ack):
        now = ack.now
        new_round = ack.send_time >= self.round_start
        if new_round:
            self.round_start = now

        # Samples taken while deliberately idling in PROBE_RTT say nothing about capacity.
        if self.state != PROBE_RTT:
            if new_round:
                self.round += 1
            self._update_bw(ack.delivery_rate)

        expired = now - self.min_rtt_stamp > self.MIN_RTT_WINDOW
        if ack.rtt <= self.min_rtt or expired:
            self.min_rtt = ack.rtt
            self.min_rtt_stamp = now

        if self.state == STARTUP:
            if new_round:
                self._check_full_pipe()
        elif self.state == DRAIN:
            if ack.inflight <= self.btl_bw * self.min_rtt:
                self._enter_probe_bw(now)
        elif self.state == PROBE_BW:
            if expired:
                self.state = PROBE_RTT
                self._probe_rtt_done = now + self.PROBE_RTT_TIME + ack.rtt
            elif now - self._cycle_stamp > self.min_rtt:
                self._cycle_idx = (self._cycle_idx + 1) % len(self.CYCLE)
                self._cycle_stamp = now
        elif now >= self._probe_rtt_done:
            self.min_rtt_stamp = now
            self._enter_probe_bw(now)

        self._set_outputs()

    def _update_bw(self, rate):
        samples = self._bw_samples
        while samples and samples[-1][1] <= rate:
            samples.pop()
        samples.append((self.round, rate))
        while samples[0][0] < self.round - self.BW_WINDOW_ROUNDS:
            samples.popleft()
        self.btl_bw = samples[0][1]

    def _check_full_pipe(self):
        if self.btl_bw >= self._full_bw * 1.25:
            self._full_bw = self.btl_bw
            self._full_bw_rounds = 0
            return
        self._full_bw_rounds += 1
        if self._full_bw_rounds >= 3:
            self.state = DRAIN

    def _enter_probe_bw(self, now):
        self.state = PROBE_BW
        # Random phase (never the drain phase) so competing flows do not probe in lockstep.
        self._cycle_idx = self._rng.choice([i for i, g in enumerate(self.CYCLE) if g >= 1.0])
        self._cycle_stamp = now

    def _set_outputs(self):
        if self.btl_bw <= 0:
            return
        if self.state == STARTUP:
            pacing_gain = cwnd_gain = self.STARTUP_GAIN
        elif self.state == DRAIN:
            pacing_gain, cwnd_gain = 1 / self.STARTUP_GAIN, self.STARTUP_GAIN
        elif self.state == PROBE_BW:
            pacing_gain, cwnd_gain = self.CYCLE[self._cycle_idx], self.CWND_GAIN
        else:
            pacing_gain, cwnd_gain = 1.0, self.CWND_GAIN

        self.pacing_rate = pacing_gain * self.btl_bw
        if self.state == PROBE_RTT:
            self.cwnd = self.PROBE_RTT_CWND
        else:
            self.cwnd = max(cwnd_gain * self.btl_bw * self.min_rtt, self.PROBE_RTT_CWND)
