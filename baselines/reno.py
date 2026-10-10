from .base import Controller


class Reno(Controller):
    """TCP NewReno: slow start, then additive increase / multiplicative decrease."""

    name = "Reno"

    def __init__(self):
        super().__init__()
        self.ssthresh = float("inf")
        self.recovery_start = -1.0

    def on_ack(self, ack):
        if self.cwnd < self.ssthresh:
            self.cwnd += 1.0
        else:
            self.cwnd += 1.0 / self.cwnd

    def on_loss(self, loss):
        # Losses of packets sent before the last reduction belong to the same episode.
        if loss.send_time < self.recovery_start:
            return
        self.recovery_start = loss.now
        self.ssthresh = max(self.cwnd / 2, self.MIN_CWND)
        self.cwnd = self.ssthresh
