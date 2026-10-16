"""Time-varying link capacities, usable as `Link(bandwidth_mbps=...)`."""
import bisect


class StepBandwidth:
    """Capacity that jumps to a new random level every few seconds."""

    def __init__(self, base_mbps, duration, rng):
        self.times, self.levels = [0.0], [base_mbps]
        t = rng.uniform(3, 8)
        while t < duration:
            self.times.append(t)
            self.levels.append(base_mbps * rng.uniform(0.5, 1.5))
            t += rng.uniform(3, 8)

    def __call__(self, t):
        return self.levels[bisect.bisect_right(self.times, t) - 1]


class Schedule:
    """Capacity given as explicit (start_time_s, mbps) steps."""

    def __init__(self, steps):
        self.times = [t for t, _ in steps]
        self.levels = [mbps for _, mbps in steps]

    def __call__(self, t):
        return self.levels[max(bisect.bisect_right(self.times, t) - 1, 0)]


class DelayTrace:
    """Extra return-path delay replayed from a recording of a real path (seconds, on a 1 ms grid).

    `excess` is the recorded RTT minus its minimum, so only the variation is replayed; the base
    delay still comes from the link. `offset` picks where in the recording to start and `scale`
    stretches the delays, both used to vary training episodes. The recording loops.
    """

    STEP = 0.001

    def __init__(self, excess, offset: float = 0.0, scale: float = 1.0):
        self.excess = excess
        self.start = int(offset / self.STEP)
        self.scale = scale

    def __call__(self, t):
        return self.scale * self.excess[(self.start + int(t / self.STEP)) % len(self.excess)]


def load_delay_trace(path, part=(0.0, 1.0)):
    """Read recordings made by `net.run_network_test --only record` and return the extra delay
    above the minimum RTT on a 1 ms grid, for the fraction `part` of each recording.

    `path` is one recording (.csv) or a folder of them; recordings are joined end to end.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd

    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no recordings in {path}")
        return np.concatenate([load_delay_trace(f, part) for f in files])
    frame = pd.read_csv(path).dropna()
    t = frame["t_s"].to_numpy()
    excess = (frame["rtt_ms"].to_numpy() - frame["rtt_ms"].min()) / 1000
    grid = np.arange(t[0], t[-1], DelayTrace.STEP)
    filled = excess[np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)]
    lo, hi = (int(f * len(filled)) for f in part)
    return filled[lo:hi]
