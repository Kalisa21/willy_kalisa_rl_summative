# environment/custom_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LegalHelpEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        super().__init__()
        self.size = size  # Grid size (5x5)
        self.window_size = 512
        self.render_mode = render_mode

        # 6 discrete actions: 0=up, 1=down, 2=left, 3=right, 4=consult book, 5=interact
        self.action_space = spaces.Discrete(6)

        # Observation: agent x, y position (2D)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=size-1, shape=(2,), dtype=int),
            "client": spaces.Box(low=0, high=size-1, shape=(2,), dtype=int)
        })

        # Set up objects
        self._agent_pos = None
        self._client_pos = None
        self._penalty_pos = (3, 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_pos = np.array([4, 2])
        self._client_pos = np.array([0, 1])  # 
        self._step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        x, y = self._agent_pos

        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and y < self.size - 1:
            y += 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.size - 1:
            x += 1
        elif action in [4, 5]:
            pass  # Placeholder for consult/interact actions

        self._agent_pos = np.array([x, y])
        self._step_count += 1

        reward = -1  # Step penalty

        # Penalty zone
        if tuple(self._agent_pos) == self._penalty_pos:
            reward -= 10

        # Reached client
        terminated = np.array_equal(self._agent_pos, self._client_pos)
        if terminated:
            reward += 50

        truncated = self._step_count >= 50
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        return {
            "agent": self._agent_pos.copy(),
            "client": self._client_pos.copy()
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_pos - self._client_pos, ord=1)
        }

    def render(self):
        if self.render_mode == "human":
            print(f"Agent: {self._agent_pos}, Client: {self._client_pos}")

    def close(self):
        pass
