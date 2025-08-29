# environments/grid_world_env.py (Corrected - Circular Import Removed)

import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Tuple, Dict
import numpy as np
# The incorrect import of 'AresEnv' that was here has been removed.

class GridWorldEnv(gym.Env):
    """
    A multi-agent Grid World environment for Ares Prime Phase 3.
    Agents must mine raw resources, refine them, and use the materials to
    build a new structure.
    """
    metadata = {'render_modes': ['console']}

    def __init__(self, size=15, max_energy=400):
        super(GridWorldEnv, self).__init__()
        self.size = size
        self.max_energy = max_energy
        self.num_agents = 2

        # --- Define static locations ---
        self._start_positions = [np.array([0, 0]), np.array([0, 1])]
        self._raw_resource_pos = np.array([2, 12])
        self._refinery_pos = np.array([7, 1])
        self._construction_site_pos = np.array([13, 13])

        # --- Define item types ---
        self.ITEM_EMPTY = 0
        self.ITEM_IRON_ORE = 1
        self.ITEM_REFINED_METAL = 2

        # --- Agent and World State ---
        self._agent_positions = None
        self._agent_energies = None
        self._agent_inventories = None
        self._refinery_output_stock = 0
        self._construction_progress = 0
        self.FABRICATOR_COST = 5

        # --- Define action and observation space ---
        self.action_space = Tuple((Discrete(5), Discrete(5)))
        self.observation_space = Dict({
            "agent_pos": Box(low=0, high=size - 1, shape=(self.num_agents, 2), dtype=np.int32),
            "agent_energy": Box(low=0, high=max_energy, shape=(self.num_agents,), dtype=np.int32),
            "agent_inventory": Box(low=0, high=2, shape=(self.num_agents,), dtype=np.int32),
            "refinery_stock": Discrete(10),
            "construction_progress": Discrete(self.FABRICATOR_COST + 1)
        })

    def _get_obs(self):
        return {
            "agent_pos": np.array(self._agent_positions),
            "agent_energy": np.array(self._agent_energies),
            "agent_inventory": np.array(self._agent_inventories),
            "refinery_stock": self._refinery_output_stock,
            "construction_progress": self._construction_progress
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_positions = [pos.copy() for pos in self._start_positions]
        self._agent_energies = [self.max_energy] * self.num_agents
        self._agent_inventories = [self.ITEM_EMPTY] * self.num_agents
        self._refinery_output_stock = 0
        self._construction_progress = 0
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, actions):
        reward = -0.1
        terminated = False
        direction_map = {0: np.array([-1, 0]), 1: np.array([1, 0]), 2: np.array([0, -1]), 3: np.array([0, 1])}

        for i in range(self.num_agents):
            action = actions[i]
            self._agent_energies[i] -= 1
            if action <= 3:
                next_pos = self._agent_positions[i] + direction_map[action]
                self._agent_positions[i] = np.clip(next_pos, 0, self.size - 1)
            elif action == 4:
                pos = self._agent_positions[i]
                inv = self._agent_inventories[i]
                if np.array_equal(pos, self._raw_resource_pos) and inv == self.ITEM_EMPTY:
                    self._agent_inventories[i] = self.ITEM_IRON_ORE
                    reward += 10
                elif np.array_equal(pos, self._refinery_pos) and inv == self.ITEM_IRON_ORE:
                    self._agent_inventories[i] = self.ITEM_EMPTY
                    self._refinery_output_stock += 1
                    reward += 10
                elif np.array_equal(pos, self._refinery_pos) and inv == self.ITEM_EMPTY and self._refinery_output_stock > 0:
                    self._agent_inventories[i] = self.ITEM_REFINED_METAL
                    self._refinery_output_stock -= 1
                    reward += 10
                elif np.array_equal(pos, self._construction_site_pos) and inv == self.ITEM_REFINED_METAL:
                    self._agent_inventories[i] = self.ITEM_EMPTY
                    self._construction_progress += 1
                    reward += 20

        if self._construction_progress >= self.FABRICATOR_COST:
            terminated = True
            reward += 1000

        truncated = any(energy <= 0 for energy in self._agent_energies)
        observation = self._get_obs()
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, mode='console'):
        pass