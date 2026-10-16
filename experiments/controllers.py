"""Every controller under test, by the name used in tables and figures."""
from functools import lru_cache

from baselines import BBR, Cubic, Reno

CLASSICAL = ["Reno", "CUBIC", "BBR"]
LEARNED = ["CUBIC+ML", "PPO", "CUBIC+PPO"]
ALL = CLASSICAL + LEARNED
# Added after the main comparison: PPO with a competitive mode (rl/switching.py), and PPO
# retrained with random RTT jitter so that it works on real networks (rl.train --jitter).
EXTRA = ["PPO-Switch", "PPO-J", "PPO-W"]  # PPO-W: trained on a recorded Wi-Fi delay trace

# Saved policies behind the two PPO controllers.
POLICIES = {"PPO": "ppo_rate", "CUBIC+PPO": "ppo_hybrid", "PPO-J": "ppo_rate_jitter", "PPO-W": "ppo_rate_wifi"}
SWITCH_POLICY = "PPO-J"  # which policy PPO-Switch runs in its PPO mode: the jitter-robust one


def available(names):
    """Leave out learned controllers whose policy has not been trained yet."""
    from rl.agent import MODELS

    return [n for n in names if n not in POLICIES or (MODELS / f"{POLICIES[n]}.zip").exists()]


@lru_cache(maxsize=None)
def _agent_factory(policy_name: str):
    from rl.agent import load_agent

    return load_agent(policy_name)


@lru_cache(maxsize=None)
def _switching_factory(policy_name: str):
    from rl.switching import load_switching

    return load_switching(policy_name)


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
    if name == "PPO-Switch":
        return _switching_factory(POLICIES[SWITCH_POLICY])()
    if name in POLICIES:
        return _agent_factory(POLICIES[name])()
    if name.startswith("policy:"):
        return _agent_factory(name.split(":", 1)[1])()
    raise KeyError(name)
