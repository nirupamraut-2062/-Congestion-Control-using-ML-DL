"""Every controller under test, by the name used in tables and figures."""
from functools import lru_cache

from baselines import BBR, Cubic, Reno

CLASSICAL = ["Reno", "CUBIC", "BBR"]
LEARNED = ["CUBIC+ML", "PPO", "CUBIC+PPO"]
ALL = CLASSICAL + LEARNED

# Saved policies behind the two PPO controllers.
POLICIES = {"PPO": "ppo_rate", "CUBIC+PPO": "ppo_hybrid"}


@lru_cache(maxsize=None)
def _agent_factory(policy_name: str):
    from rl.agent import load_agent

    return load_agent(policy_name)


def make(name: str, seed: int = 0):
    """Build a fresh controller. `policy:<file>` loads any saved PPO policy from models/."""
    if name == "Reno":
        return Reno()
    if name == "CUBIC":
        return Cubic()
    if name == "BBR":
        return BBR(seed=seed)
    if name == "CUBIC+ML":
        from ml.loss_aware import LossAwareCubic

        return LossAwareCubic()
    if name in POLICIES:
        return _agent_factory(POLICIES[name])()
    if name.startswith("policy:"):
        return _agent_factory(name.split(":", 1)[1])()
    raise KeyError(name)
