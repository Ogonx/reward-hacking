from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from grid_env import GridEnv

env = Monitor(GridEnv(reward_mode="flawed"))
model = PPO("MlpPolicy", env, seed=0, verbose=1, ent_coef=0.01)
model.learn(total_timesteps=100_000)
model.save("flawed_model")

goals = 0
coins_total = 0
episodes = 20
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    goals += int(info["reached_goal"])
    coins_total += info["coins_collected"]

print("episodes:", episodes)
print("reached goal:", goals, "/", episodes)
print("avg coins per episode:", coins_total / episodes)