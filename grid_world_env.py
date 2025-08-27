import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    """
    Custom Environment for Ares Prime - Phase 1.
    Includes energy mechanic, obstacles, and a resource node.
    The agent must manage energy to survive.
    """
    metadata = {'render_modes': ['console']}

    def __init__(self, size=10, max_energy=30):
        super(GridWorldEnv, self).__init__()
        self.size = size
        self.max_energy = max_energy
        self.window_size = 512

        # Define static locations
        self._start_pos = np.array([0, 0])
        self._target_pos = np.array([size - 1, size - 1])
        # The resource is now a charging station
        self._charge_pos = np.array([2, 7])
        self._obstacles_pos = [
            np.array([1, 2]), np.array([2, 2]), np.array([3, 2]),
            np.array([4, 5]), np.array([5, 5]), np.array([6, 5]),
            np.array([7, 8]), np.array([8, 8]), np.array([8, 7])
        ]

        # State of the agent
        self._agent_pos = None
        self._has_resource = None # Represents 'key' to unlock goal
        self._agent_energy = None

        # Action space remains the same
        self.action_space = spaces.Discrete(4)

        # Define observation space: [row, col, has_resource, energy]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([size - 1, size - 1, 1, max_energy]),
            shape=(4,),
            dtype=np.int32
        )

    def _get_obs(self):
        return np.array([self._agent_pos[0], self._agent_pos[1], self._has_resource, self._agent_energy])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_pos = self._start_pos.copy()
        self._has_resource = 0 # 0 for False
        self._agent_energy = self.max_energy
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # Every action costs energy
        self._agent_energy -= 1

        direction_map = {0: np.array([-1, 0]), 1: np.array([1, 0]), 2: np.array([0, -1]), 3: np.array([0, 1])}
        prev_pos = self._agent_pos.copy()
        next_pos = self._agent_pos + direction_map[action]

        # Check for obstacle collision
        if any(np.array_equal(next_pos, obs) for obs in self._obstacles_pos):
            reward = -10.0
            self._agent_pos = prev_pos
        else:
            self._agent_pos = np.clip(next_pos, 0, self.size - 1)
            reward = -1.0 # Standard cost for a step

        # Check for charging station
        if np.array_equal(self._agent_pos, self._charge_pos):
            self._has_resource = 1 # 'Key' collected
            self._agent_energy = self.max_energy # Energy replenished
            reward += 20.0 # Small reward for charging

        # Check for goal completion
        terminated = np.array_equal(self._agent_pos, self._target_pos) and self._has_resource == 1
        if terminated:
            reward += 100.0

        # Check for energy depletion (failure state)
        truncated = False
        if self._agent_energy <= 0:
            truncated = True
            reward = -50.0 # Heavy penalty for running out of energy

        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self, mode='console'):
        if mode == 'console':
            grid = np.full((self.size, self.size), '.')
            for obs in self._obstacles_pos: grid[tuple(obs)] = 'X'
            
            # Show charging station if the goal isn't unlocked
            if self._has_resource == 0: grid[tuple(self._charge_pos)] = 'R'
            
            grid[tuple(self._target_pos)] = 'G'
            grid[tuple(self._agent_pos)] = 'A'
            
            print('\n'.join([' '.join(row) for row in grid]))
            print(f"Energy: {self._agent_energy}/{self.max_energy} | Goal Unlocked: {'Yes' if self._has_resource else 'No'}")
            print("-" * (self.size * 2))

    def close(self):
        pass