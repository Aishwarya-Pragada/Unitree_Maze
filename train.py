"""
PPO training for the A1 trot with velocity commands.

  python train.py --run-name trot_v1 --timesteps 5000000 --n-envs 8
  tensorboard --logdir runs
"""
import argparse
import os
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from rl_env import A1Env, HERE


def make_env(xml, seed, rank):
    def _init():
        env = A1Env(xml_path=xml)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


class SaveAndLogCallback(BaseCallback):
    """Saves model + VecNormalize stats periodically, logs each reward term to TensorBoard."""

    def __init__(self, run_dir, save_every=500_000):
        super().__init__()
        self.run_dir = run_dir
        self.save_every = save_every
        self.next_save = save_every
        self.sums, self.count = {}, 0

    def _on_step(self):
        for info in self.locals["infos"]:
            for k, v in info.items():
                if k.startswith("rew/"):
                    self.sums[k] = self.sums.get(k, 0.0) + v
            self.count += 1
        if self.num_timesteps >= self.next_save:
            self.next_save += self.save_every
            tag = f"{self.num_timesteps // 1000}k"
            ck = os.path.join(self.run_dir, "checkpoints")
            os.makedirs(ck, exist_ok=True)
            self.model.save(os.path.join(ck, f"model_{tag}"))
            self.training_env.save(os.path.join(ck, f"vecnormalize_{tag}.pkl"))
            if self.verbose:
                print(f"[ckpt] saved {tag}")
        return True

    def _on_rollout_end(self):
        for k, v in self.sums.items():
            self.logger.record(k, v / max(self.count, 1))
        self.sums, self.count = {}, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=os.path.join(HERE, "scene.xml"))
    p.add_argument("--run-name", default="trot_v1")
    p.add_argument("--timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = os.path.join(HERE, "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    venv = SubprocVecEnv([make_env(args.xml, args.seed, i) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=1024,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=th.nn.ELU,
            log_std_init=-1.0,
        ),
        tensorboard_log=os.path.join(HERE, "runs"),
        device="cpu",
        seed=args.seed,
        verbose=1,
    )

    cb = SaveAndLogCallback(run_dir, save_every=max(500_000, 1))
    cb.verbose = 1
    try:
        model.learn(total_timesteps=args.timesteps, callback=cb, tb_log_name=args.run_name)
    finally:
        model.save(os.path.join(run_dir, "model"))
        venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
        print("saved to", run_dir)
        venv.close()


if __name__ == "__main__":
    main()