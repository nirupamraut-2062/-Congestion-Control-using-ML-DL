from .base import Controller


class Cubic(Controller):
    """TCP CUBIC (RFC 8312): window grows as a cubic function of time since the last loss."""

    name = "CUBIC"
    C = 0.4
    BETA = 0.7

    def __init__(self):
        super().__init__()
        self.ssthresh = float("inf")
        self.recovery_start = -1.0
        self.w_max = 0.0
        self.epoch_start = None
        self.k = 0.0
        self.w_est = 0.0

    def on_ack(self, ack):
        if self.cwnd < self.ssthresh:
            self.cwnd += 1.0
            return
        if self.epoch_start is None:
            self.epoch_start = ack.now
            self.k = (max(self.w_max - self.cwnd, 0.0) / self.C) ** (1 / 3)
            self.w_est = self.cwnd

        t = ack.now - self.epoch_start + ack.rtt
        w_cubic = self.C * (t - self.k) ** 3 + self.w_max
        target = min(w_cubic, 1.5 * self.cwnd)
        if target > self.cwnd:
            self.cwnd += (target - self.cwnd) / self.cwnd
        else:
            self.cwnd += 0.01 / self.cwnd
        # Window standard TCP would have reached, so CUBIC is never slower than Reno.
        self.w_est += 3 * (1 - self.BETA) / (1 + self.BETA) / self.cwnd
        self.cwnd = max(self.cwnd, self.w_est)

    def on_loss(self, loss):
        if loss.send_time < self.recovery_start:
            return
        self.recovery_start = loss.now
        self.epoch_start = None
        # Fast convergence: release bandwidth sooner if the window is still shrinking.
        self.w_max = self.cwnd * (1 + self.BETA) / 2 if self.cwnd < self.w_max else self.cwnd
        self.cwnd = max(self.cwnd * self.BETA, self.MIN_CWND)
        self.ssthresh = self.cwnd
