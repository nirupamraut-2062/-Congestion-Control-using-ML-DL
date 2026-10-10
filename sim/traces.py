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
