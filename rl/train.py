"""Train a PPO congestion-control policy.

Usage:
  python -m rl.train --name ppo_rate                               # pure rate control
  python -m rl.train --name ppo_hybrid --mode hybrid               # PPO steering CUBIC
  python -m rl.train --name ppo_lowlat --profile low_latency       # different reward weights
  python -m rl.train --name ppo_nohist --history 1                 # ablation: no history
  python -m rl.train --name ppo_noloss --drop loss_rate            # ablation: drop a feature
  python -m rl.train --name ppo_rate_jitter --jitter 10 --rtt-filter min   # robust to RTT jitter
  python -m rl.train --name ppo_rate_wifi --jitter 10 --rtt-filter min --trace data/wifi_traces --queue-reward

Writes models/<name>.zip (policy), models/<name>.json (how to rebuild the controller)
and results/training_<name>.csv (one row per finished training episode).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from .agent import FEATURES, MODELS
from .env import PROFILES, CongestionEnv

RESULTS = Path(__file__).resolve().parent.parent / "results"


class EpisodeLog(BaseCallback):
    """Records per-episode metrics and checkpoints the policy while training runs."""

    def __init__(self, name: str, every: int = 100_000):
        super().__init__()
        self.name = name
        self.every = every
        self.rows = []
        self._next = every

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "metrics" in info:
                self.rows.append({"timesteps": self.num_timesteps, **info["metrics"]})
        if self.num_timesteps >= self._next:
            self._next += self.every
            self.flush()
        return True

    def flush(self) -> None:
        self.model.save(MODELS / self.name)
        log = pd.DataFrame(self.rows)
        log.to_csv(RESULTS / f"training_{self.name}.csv", index=False)
        recent = log.tail(40).mean()
        print(f"{self.name} {self.num_timesteps:>8} steps  reward {recent['reward']:.3f}  "
              f"throughput {recent['throughput']:.3f}  delay {recent['delay']:.3f}  loss {recent['loss']:.4f}",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=["rate", "hybrid"], default="rate")
    parser.add_argument("--profile", choices=list(PROFILES), default="balanced")
    parser.add_argument("--cross", type=float, default=0.0, help="share of episodes with a competing flow")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--drop", nargs="*", default=[], choices=FEATURES, help="features to leave out")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="train with random return-path jitter of up to this mean (ms) in most episodes")
    parser.add_argument("--rtt-filter", choices=["mean", "min"], default="mean",
                        help="latency signals from the mean or the minimum RTT of each half interval")
    parser.add_argument("--trace", help="replay recorded path delay: a .csv or a folder of them (net.run_network_test --only record)")
    parser.add_argument("--queue-reward", action="store_true",
                        help="score delay by the bottleneck queue, not the RTT (ignores delay the path adds)")
    parser.add_argument("--steps", type=int, default=800_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(1)

    features = [f for f in FEATURES if f not in args.drop]
    config = {"mode": args.mode, "profile": args.profile, "cross_traffic": args.cross,
              "history": args.history, "features": features}
    if args.jitter:
        config["jitter_ms"] = args.jitter
    if args.rtt_filter != "mean":
        config["rtt_filter"] = args.rtt_filter
    if args.trace:
        config["jitter_trace"] = args.trace
    if args.queue_reward:
        config["queue_reward"] = True
    (MODELS / f"{args.name}.json").write_text(json.dumps(config, indent=2))

    env = make_vec_env(CongestionEnv, n_envs=args.envs, seed=args.seed, env_kwargs=config)
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-4, policy_kwargs={"net_arch": [64, 64], "log_std_init": -1.0},
                seed=args.seed, device="cpu")
    log = EpisodeLog(args.name)
    model.learn(args.steps, callback=log)
    log.flush()


if __name__ == "__main__":
    main()
