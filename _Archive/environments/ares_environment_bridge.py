# ares_environment_bridge.py (Corrected for Multi-Agent)
# Author: Ares Prime, Lead Data Scientist
# Description: This module now correctly bridges the multi-agent GridWorldEnv
#              with the Gymnasium API for multi-agent DRL.

import gymnasium as gym
import numpy as np

# --- IMPORTING FROM THE EXISTING GITHUB CODEBASE (Corrected) ---
# We only need to import the environment class itself.
from environments.grid_world_env import GridWorldEnv

class AresEnv(gym.Env):
    """
    A Gymnasium environment wrapper for the Ares Prime Multi-Agent simulation.
    """
    def __init__(self, config):
        super(AresEnv, self).__init__()
        self.config = config

        # Initialize the user's simulation engine.
        # This is now the GridWorldEnv you provided.
        self.sim = GridWorldEnv()

        # --- CORRECTED: Spaces now directly mirror the multi-agent environment ---
        self.action_space = self.sim.action_space
        self.observation_space = self.sim.observation_space

    def reset(self, seed=None, options=None):
        """
        Resets the environment by calling the underlying simulation's reset method.
        """
        # Your environment's reset method handles the setup.
        observation, info = self.sim.reset(seed=seed, options=options)
        return observation, info

    def step(self, action):
        """
        Executes one time step. It now accepts a tuple of actions,
        one for each agent, and passes it directly to your environment.
        """
        # The action from our multi-agent brain is passed directly.
        # Your environment's step function returns all the necessary values.
        observation, reward, terminated, truncated, info = self.sim.step(action)
        return observation, reward, terminated, truncated, info

    def render(self):
        """Passes the render call to your simulation's render method."""
        self.sim.render()

    def close(self):
        """Closes the environment."""
        self.sim.close()