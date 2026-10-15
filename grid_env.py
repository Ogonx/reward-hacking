import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridEnv(gym.Env):
    def __init__(self, reward_mode="flawed", size=10, max_steps=200):
        super().__init__()#
        self.size = size
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.coins = {(2, 2), (3, 2)}
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.pos = self.start
        self.steps = 0
        self.coins_collected = 0
        self.coins_taken = set()

    def _obs(self):
        return np.array(self.pos, dtype=np.float32) / (self.size - 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start
        self.steps = 0
        self.coins_collected = 0
        self.coins_taken = set()
        return self._obs(), {}

    def step(self, action):
        x, y = self.pos
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1
        x = min(max(x, 0), self.size - 1)
        y = min(max(y, 0), self.size - 1)
        moved = (x, y) != self.pos
        self.pos = (x, y)
        self.steps += 1

        reward = 0.0
        terminated = False

        if self.reward_mode == "fixed":
            reward -= 0.01

        if moved and self.pos in self.coins:
            self.coins_collected += 1
            if self.reward_mode == "flawed":
                reward += 1.0
            elif self.pos not in self.coins_taken:
                reward += 1.0
            self.coins_taken.add(self.pos)

        reached_goal = self.pos == self.goal
        if reached_goal:
            reward += 10.0
            terminated = True

        truncated = self.steps >= self.max_steps and not terminated
        info = {
            "reached_goal": reached_goal,
            "coins_collected": self.coins_collected,
        }
        return self._obs(), reward, terminated, truncated, info