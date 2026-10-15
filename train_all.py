import csv
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs import ExploitEnv, FLAWS, HONEST_RETURN

SEEDS = [0, 1, 2]
MODES = ["flawed", "fixed"]
TIMESTEPS = 60_000
EVAL_EPISODES = 30

FIELDS = [
    "flaw", "mode", "seed", "episode", "ep_return", "norm_return", "steps",
    "true_success", "unique_cells", "still_frac", "back_frac", "reward_rate",
]


def episode_features(env, model, flaw):
    obs, _ = env.reset()
    done = False
    total = 0.0
    positions = [env.pos]
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        positions.append(info["pos"])
        done = terminated or truncated
    steps = len(positions) - 1
    still = sum(1 for i in range(1, len(positions)) if positions[i] == positions[i - 1])
    back = sum(1 for i in range(2, len(positions)) if positions[i] == positions[i - 2])
    return {
        "ep_return": round(total, 4),
        "norm_return": round(total / HONEST_RETURN[flaw], 4),
        "steps": steps,
        "true_success": int(info["true_success"]),
        "unique_cells": len(set(positions)),
        "still_frac": round(still / steps, 4),
        "back_frac": round(back / max(steps - 1, 1), 4),
        "reward_rate": round(total / steps, 4),
    }


start = time.time()
with open("episodes.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    for flaw in FLAWS:
        for mode in MODES:
            for seed in SEEDS:
                env = Monitor(ExploitEnv(flaw=flaw, mode=mode))
                model = PPO("MlpPolicy", env, seed=seed, verbose=0, ent_coef=0.01)
                model.learn(total_timesteps=TIMESTEPS)
                raw = env.env if hasattr(env, "env") else env
                succ = 0
                for ep in range(EVAL_EPISODES):
                    feats = episode_features(raw, model, flaw)
                    succ += feats["true_success"]
                    row = {"flaw": flaw, "mode": mode, "seed": seed, "episode": ep}
                    row.update(feats)
                    writer.writerow(row)
                f.flush()
                print(
                    flaw, mode, "seed", seed,
                    "true success", round(succ / EVAL_EPISODES, 2),
                    "elapsed min", round((time.time() - start) / 60, 1),
                    flush=True,
                )
print("DONE")