"""Sender-side statistics used as features by the loss classifier.

Everything here is computed from ACKs and loss notifications only, i.e. from what
a real sender can observe. All features are ratios, so a model trained on one set
of links transfers to links with different speeds and delays.
"""
import math
from collections import deque

FEATURES = [
    "rtt_ratio",        # latest RTT / minimum RTT (how full the queue is)
    "srtt_ratio",       # smoothed RTT / minimum RTT
    "rtt_peak_ratio",   # latest RTT / highest RTT seen recently (1.0 = queue as full as it gets)
    "rtt_trend",        # RTT change across the last few ACKs, in units of minimum RTT
    "rtt_dev",          # RTT variation / minimum RTT
    "rate_ratio",       # delivery rate / highest delivery rate seen recently
    "inflight_ratio",   # packets in flight / estimated pipe size (max rate * min RTT)
    "losses_last_rtt",  # other losses reported within the last smoothed RTT (burstiness)
    "rtts_since_loss",  # time since the previous loss, in smoothed RTTs (capped)
    "loss_rate",        # exponentially weighted loss rate over roughly the last 100 packets
]


class FlowMonitor:
    TREND_ACKS = 16
    PEAK_MEMORY_S = 20.0
    MAX_RTTS_SINCE_LOSS = 100.0

    def __init__(self):
        self.min_rtt = math.inf
        self.peak_rtt = 0.0
        self.srtt = 0.0
        self.rttvar = 0.0
        self.last_rtt = 0.0
        self.rate = 0.0
        self.peak_rate = 0.0
        self.inflight = 0
        self.loss_rate = 0.0

        self._last_ack = 0.0
        self._last_loss = None
        self._recent_rtts = deque(maxlen=self.TREND_ACKS)
        self._recent_losses = deque()

    def on_ack(self, ack):
        rtt = ack.rtt
        if self.srtt == 0.0:
            self.srtt, self.rttvar = rtt, rtt / 2
        else:
            self.rttvar = 0.75 * self.rttvar + 0.25 * abs(self.srtt - rtt)
            self.srtt = 0.875 * self.srtt + 0.125 * rtt
        self.min_rtt = min(self.min_rtt, rtt)

        # Peaks fade slowly so they track a link whose capacity changes.
        keep = math.exp(-(ack.now - self._last_ack) / self.PEAK_MEMORY_S)
        self.peak_rtt = max(rtt, self.min_rtt + (self.peak_rtt - self.min_rtt) * keep)
        self.peak_rate = max(ack.delivery_rate, self.peak_rate * keep)

        self.last_rtt = rtt
        self.rate = ack.delivery_rate
        self.inflight = ack.inflight
        self.loss_rate *= 0.99
        self._last_ack = ack.now
        self._recent_rtts.append(rtt)

    def on_loss(self, loss):
        self.loss_rate = 0.99 * self.loss_rate + 0.01
        self._last_loss = loss.now
        self._recent_losses.append(loss.now)

    def features(self, now: float) -> list[float]:
        """Feature vector in the order of FEATURES, describing the flow just before `now`."""
        if self.srtt == 0.0:
            return [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, self.MAX_RTTS_SINCE_LOSS, self.loss_rate]

        recent = self._recent_losses
        while recent and now - recent[0] > self.srtt:
            recent.popleft()

        rtts = self._recent_rtts
        half = len(rtts) // 2
        trend = 0.0
        if half >= 2:
            older = sum(rtts[i] for i in range(half)) / half
            newer = sum(rtts[i] for i in range(len(rtts) - half, len(rtts))) / half
            trend = (newer - older) / self.min_rtt

        since_loss = self.MAX_RTTS_SINCE_LOSS
        if self._last_loss is not None:
            since_loss = min((now - self._last_loss) / self.srtt, self.MAX_RTTS_SINCE_LOSS)

        return [
            self.last_rtt / self.min_rtt,
            self.srtt / self.min_rtt,
            self.last_rtt / self.peak_rtt,
            trend,
            self.rttvar / self.min_rtt,
            self.rate / self.peak_rate if self.peak_rate else 1.0,
            self.inflight / max(self.peak_rate * self.min_rtt, 1.0),
            float(len(recent)),
            since_loss,
            self.loss_rate,
        ]
