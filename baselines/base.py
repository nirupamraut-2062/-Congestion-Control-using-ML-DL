class Controller:
    """Interface the simulator expects from a congestion controller."""

    name = "base"
    INITIAL_CWND = 10.0
    MIN_CWND = 2.0

    def __init__(self):
        self.cwnd = self.INITIAL_CWND  # packets
        self.pacing_rate = None  # packets/s; None means ACK-clocked

    def on_ack(self, ack):
        pass

    def on_loss(self, loss):
        pass
