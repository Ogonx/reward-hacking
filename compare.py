import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from grid_env import GridEnv


def run(mode, seed, timesteps=100_000, episodes=20):
    env = Monitor(GridEnv(reward_mode=mode))
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, ent_coef=0.01)
    model.learn(total_timesteps=timesteps)
    goals = 0
    coins = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        goals += int(info["reached_goal"])
        coins += info["coins_collected"]
    return goals / episodes, coins / episodes


seeds = [0, 1, 2, 3, 4]
results = {}
for mode in ["flawed", "fixed"]:
    results[mode] = []
    for s in seeds:
        g, c = run(mode, s)
        results[mode].append((g, c))
        print(mode, "seed", s, "goal rate", g, "avg coins", c, flush=True)

for mode in ["flawed", "fixed"]:
    g = np.mean([r[0] for r in results[mode]])
    c = np.mean([r[1] for r in results[mode]])
    print("SUMMARY", mode, "goal rate", round(g, 2), "avg coins", round(c, 1))