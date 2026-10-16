"""PPO with a competitive mode: fixes the agent's loss of bandwidth against loss-based flows.

The PPO agent keeps the queue almost empty. A loss-based flow such as CUBIC does the
opposite: it fills the buffer until packets drop. When the two share a link the agent
sees a standing queue it cannot remove, keeps lowering its rate, and ends with about 2%
of the link. The same idea as Copa's competitive mode and Nimbus fixes this:

  PPO mode   The learned policy controls the rate. Two signs that another flow is filling
             the buffer trigger a switch to TCP mode: the queueing delay stays high for
             several intervals in a row, or over a few seconds the agent's rate has slid
             steadily to half while a moderate queue persisted. The second sign is needed when the
             agent joins a link whose queue is already full: its minimum RTT is then
             inflated and the queueing delay it measures looks small. A flow that is
             alone always lowers the RTT when it sends slower.
  TCP mode   Loss-aware CUBIC (CUBIC+ML) controls the window, so the flow competes on
             equal terms with CUBIC and still ignores random loss. Every few seconds the
             window is cut below the path's bandwidth-delay product for a moment (like
             BBR's ProbeRTT). If the RTT then falls clearly, back to the base RTT,
             nothing else is holding the queue, so switch back to PPO mode.
"""
import math
from collections import deque

from baselines.cubic import Cubic
from ml.loss_aware import LossAwareCubic

from .agent import RateAgent, load_policy


class SwitchingAgent:
    name = "PPO-Switch"

    ENTER_RATIO = 0.5    # queueing delay above 50% of the base RTT ...
    ENTER_INTERVALS = 6  # ... for this many decision intervals in a row: someone else holds the queue
    PROBE_EVERY = 4.0    # seconds in TCP mode between drain probes ...
    PROBE_MAX = 16.0     # ... doubled up to this after each failed probe or false exit (exponential backoff)
    FALSE_EXIT = 10.0    # back in TCP mode within this many seconds of leaving it: the exit was false
    PROBE_FRACTION = 0.5  # window during a probe, as a share of the estimated bandwidth-delay product
    EXIT_RATIO = 0.2     # probe RTT within 20% of the base RTT: the queue emptied, nobody else is there
    WINDOW = 30          # intervals compared by the "backing off does not help" test (a few seconds)
    CUT = 0.5            # rate fell below half its value WINDOW intervals ago ...
    PERSIST = 0.1        # ... while the mean queueing delay over those intervals stayed above 10% of base
    SLIDE = 0.9          # ... and the rate fell by at least 10% in each third of the window
    SPIKE = 0.6          # ... and the queueing delay never jumped above 60% of base: a capacity drop
                         # makes a sudden queue that the agent then works off, a competitor does not
    DROP = 0.8           # a probe must also lower the RTT to below 80% of its value before the probe

    def __init__(self, policy=None, **kwargs):
        self.ppo = RateAgent(policy=policy, **kwargs)
        self.tcp = LossAwareCubic()
        self.mode = "ppo"
        self.switches = []  # (time, new mode), for plots and tables
        self._high = 0
        self._pre_probe_rtt = 0.0
        self._recent = deque(maxlen=self.WINDOW)  # (rate, queueing delay ratio) at the end of each interval
        self._seen_end = None
        self._delivery = 0.0  # smoothed delivery rate, packets/s
        self._next_probe = math.inf
        self._probe_every = self.PROBE_EVERY
        self._left_tcp = -math.inf
        self._probe_end = None
        self._probe_min_rtt = math.inf
        self._saved_cwnd = 0.0
        self._probe_start = 0.0

    # ------------------------------------------------------------------ simulator interface

    @property
    def cwnd(self):
        return self.ppo.cwnd if self.mode == "ppo" else self.tcp.cwnd

    @property
    def pacing_rate(self):
        return self.ppo.pacing_rate if self.mode == "ppo" else None

    def on_ack(self, ack):
        self.ppo.on_ack(ack)
        self._delivery = ack.delivery_rate if not self._delivery else 0.9 * self._delivery + 0.1 * ack.delivery_rate
        if self.mode == "ppo":
            self.tcp.monitor.on_ack(ack)  # keep the loss classifier's features current
            return
        if self._probe_end is not None and ack.send_time >= self._probe_start:
            self._probe_min_rtt = min(self._probe_min_rtt, ack.rtt)
        if self._probe_end is None:
            self.tcp.on_ack(ack)
        else:
            self.tcp.monitor.on_ack(ack)

    def on_loss(self, loss):
        self.ppo.on_loss(loss)
        tcp = self.tcp
        if self.mode == "tcp" and self._probe_end is None:
            if tcp.ssthresh == math.inf:
                # Slow start overshoots on purpose: its first losses are congestion, never ignore them.
                tcp.monitor.on_loss(loss)
                Cubic.on_loss(tcp, loss)
            else:
                tcp.on_loss(loss)
        else:
            tcp.monitor.on_loss(loss)

    def on_tick(self, now):
        if self.mode == "ppo":
            self.ppo.on_tick(now)
            end = self.ppo.interval_end
            if end is not None and self._seen_end is not None and end != self._seen_end:
                self._interval_done(now)
            self._seen_end = end
        else:
            self._tcp_tick(now)

    # ------------------------------------------------------------------ mode logic

    def _interval_done(self, now):
        ppo = self.ppo
        self._high = self._high + 1 if ppo._latency_ratio > self.ENTER_RATIO else 0
        self._recent.append((ppo.pacing_rate, ppo._latency_ratio))
        stuck = False
        if len(self._recent) == self.WINDOW:
            mean_delay = sum(r for _, r in self._recent) / self.WINDOW
            # A steady slide, falling in every third of the window. After a capacity drop a lone
            # flow cuts its rate sharply and then levels off; behind a loss-based flow it keeps sliding.
            rates = [r for r, _ in self._recent]
            thirds = [rates[0], rates[self.WINDOW // 3], rates[2 * self.WINDOW // 3], rates[-1]]
            sliding = all(b < self.SLIDE * a for a, b in zip(thirds, thirds[1:]))
            peak_delay = max(r for _, r in self._recent)
            stuck = (sliding and rates[-1] < self.CUT * rates[0] and mean_delay > self.PERSIST
                     and peak_delay < self.SPIKE)
        if self._high >= self.ENTER_INTERVALS or stuck:
            self._to_tcp(now, starved=stuck)

    def _bdp(self):
        return self._delivery * self.ppo.min_rtt

    def _to_tcp(self, now, starved=False):
        tcp = self.tcp
        tcp.cwnd = max(self._bdp(), tcp.MIN_CWND)
        if starved:
            # The agent backed off to almost nothing, so its rate says little about its fair
            # share: grow in slow start, as a new TCP connection would.
            tcp.ssthresh = math.inf
            tcp.w_max = 0.0
        else:
            # The agent still had a fair rate: continue from it, as if CUBIC had been running.
            tcp.ssthresh = tcp.w_max = tcp.cwnd
        tcp.epoch_start = None
        tcp.recovery_start = now
        self.mode = "tcp"
        self.switches.append((now, "tcp"))
        if now - self._left_tcp < self.FALSE_EXIT:
            self._probe_every = min(2 * self._probe_every, self.PROBE_MAX)
        else:
            self._probe_every = self.PROBE_EVERY
        self._next_probe = now + self._probe_every
        self._probe_end = None

    def _tcp_tick(self, now):
        if self._probe_end is None:
            if now >= self._next_probe:
                srtt = self.ppo.srtt or self.ppo.min_rtt
                self._saved_cwnd = self.tcp.cwnd
                self._pre_probe_rtt = srtt
                self.tcp.cwnd = max(self.PROBE_FRACTION * self._bdp(), self.tcp.MIN_CWND)
                self._probe_start = now
                self._probe_end = now + srtt + 2 * self.ppo.min_rtt
                self._probe_min_rtt = math.inf
            return
        if now < self._probe_end:
            return
        self._probe_end = None
        # Both tests: the minimum RTT can be inflated if the flow started behind a full queue.
        if (self._probe_min_rtt < (1 + self.EXIT_RATIO) * self.ppo.min_rtt
                and self._probe_min_rtt < self.DROP * self._pre_probe_rtt):
            self._to_ppo(now)
        else:
            self.tcp.cwnd = self._saved_cwnd
            # The competitor is still there: probe less often, each probe gives it bandwidth.
            self._probe_every = min(2 * self._probe_every, self.PROBE_MAX)
            self._next_probe = now + self._probe_every

    def _to_ppo(self, now):
        ppo = self.ppo
        ppo.pacing_rate = min(max(self._saved_cwnd / max(ppo.srtt, ppo.min_rtt), ppo.MIN_RATE), ppo.MAX_RATE)
        for i in range(len(ppo.history)):
            ppo.history[i] = ppo.history[i] * 0
        ppo.interval_end = None
        self._seen_end = None
        self._high = 0
        self._recent.clear()
        self.mode = "ppo"
        self.switches.append((now, "ppo"))
        self._left_tcp = now


def load_switching(name: str = "ppo_rate"):
    """Return a zero-argument factory for a SwitchingAgent running the saved rate policy `name`."""
    policy, config = load_policy(name)
    return lambda: SwitchingAgent(policy=policy, history=config["history"], features=config["features"],
                                  rtt_filter=config.get("rtt_filter", "mean"))
