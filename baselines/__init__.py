from .base import Controller
from .bbr import BBR
from .cubic import Cubic
from .reno import Reno

BASELINES = {"Reno": Reno, "CUBIC": Cubic, "BBR": BBR}


def make(name: str, seed: int = 0) -> Controller:
    cls = BASELINES[name]
    return cls(seed=seed) if cls is BBR else cls()


__all__ = ["BASELINES", "BBR", "Controller", "Cubic", "Reno", "make"]
