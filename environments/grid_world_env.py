import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Tuple
import numpy as np

class GridWorldEnv(gym.Env):
    """
    A multi-agent Grid World environment for Ares Prime Phase 2.
    Two agents must cooperate to collect two resources before reaching the goal.
    """
    metadata = {'render_modes': ['console']}

    def __init__(self, size=10, max_energy=50):
        super(GridWorldEnv, self).__init__()
        self.size = size
        self.max_energy = max_energy
        self.num_agents = 2

        # --- Define static locations ---
        self._start_positions = [np.array([0, 0]), np.array([0, 1])]
        self._target_pos = np.array([size - 1, size - 1])
        self._resource_positions = [np.array([2, 7]), np.array([7, 2])] # R1 and R2
        self._obstacles_pos = [
            np.array([1, 4]), np.array([2, 4]), np.array([3, 4]),
            np.array([4, 1]), np.array([4, 2]), np.array([4, 3]),
            np.array([6, 6]), np.array([7, 6]), np.array([8, 6]),
            np.array([6, 8]), np.array([7, 8]), np.array([8, 8])
        ]

        # --- Agent and World State ---
        self._agent_positions = None
        self._agent_energies = None
        self._resources_collected = None

        # --- Define action and observation space ---
        self.action_space = Tuple((Discrete(4), Discrete(4)))

        # Observation space for a single agent:
        # [my_row, my_col, my_energy, teammate_row, teammate_col, teammate_energy, r1_collected, r2_collected]
        obs_low = np.array([0, 0, 0] * self.num_agents + [0] * len(self._resource_positions))
        obs_high = np.array([size - 1, size - 1, max_energy] * self.num_agents + [1] * len(self._resource_positions))
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.int32)
    
    def _get_obs(self):
        agent_states = np.array([np.concatenate((pos, [energy])) for pos, energy in zip(self._agent_positions, self._agent_energies)]).flatten()
        resource_states = np.array(self._resources_collected)
        return np.concatenate((agent_states, resource_states)).astype(np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_positions = [pos.copy() for pos in self._start_positions]
        self._agent_energies = [self.max_energy] * self.num_agents
        self._resources_collected = [0] * len(self._resource_positions)
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, actions):
        # Actions is a tuple: (action_agent_1, action_agent_2)
        reward = 0
        terminated = False
        truncated = False
        direction_map = {0: np.array([-1, 0]), 1: np.array([1, 0]), 2: np.array([0, -1]), 3: np.array([0, 1])}

        # --- Move Agents and Check Collisions ---
        next_positions = []
        for i in range(self.num_agents):
            next_pos = self._agent_positions[i] + direction_map[actions[i]]
            # Check for wall collisions
            next_pos = np.clip(next_pos, 0, self.size - 1)
            # Check for obstacle collisions
            if any(np.array_equal(next_pos, obs) for obs in self._obstacles_pos):
                next_pos = self._agent_positions[i] # Revert move
                reward -= 5 # Penalty for trying to move into an obstacle
            next_positions.append(next_pos)
        
        # Check for agent-agent collisions
        if np.array_equal(next_positions[0], next_positions[1]):
            # Both agents tried to move to the same spot, revert both moves
            next_positions[0] = self._agent_positions[0]
            next_positions[1] = self._agent_positions[1]
            reward -= 5 # Penalty for colliding with teammate
            
        self._agent_positions = next_positions

        # --- Calculate Rewards and State Changes ---
        for i in range(self.num_agents):
            self._agent_energies[i] -= 1 # Energy cost for taking a step
            
            # Check if an agent collected a resource
            for j, res_pos in enumerate(self._resource_positions):
                if np.array_equal(self._agent_positions[i], res_pos) and self._resources_collected[j] == 0:
                    self._resources_collected[j] = 1
                    reward += 50 # Shared team reward for collecting a resource

        # --- Check for Terminal/Truncated States ---
        all_resources_collected = all(self._resources_collected)
        
        # Termination: Goal is reached AND all resources are collected
        if all_resources_collected:
            for pos in self._agent_positions:
                if np.array_equal(pos, self._target_pos):
                    terminated = True
                    reward += 200 # Large shared reward for completing the mission
                    break
        
        # Truncation: Any agent runs out of energy
        if any(energy <= 0 for energy in self._agent_energies):
            truncated = True

        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self, mode='console'):
        grid = np.full((self.size, self.size), '.')
        for obs in self._obstacles_pos: grid[tuple(obs)] = 'X'
        
        # Display resources if they haven't been collected
        for i, res_pos in enumerate(self._resource_positions):
            if self._resources_collected[i] == 0:
                grid[tuple(res_pos)] = f'R{i+1}'
        
        grid[tuple(self._target_pos)] = 'G'
        
        # Display agents
        for i, pos in enumerate(self._agent_positions):
            # Check if agents are on the same spot to avoid overwriting
            if grid[tuple(pos)] == '.':
                grid[tuple(pos)] = f'{i+1}' # Display as '1' and '2'
        
        print('\n'.join([' '.join(row) for row in grid]))
        print(f"Energies: {self._agent_energies} | Resources: {self._resources_collected}")
        print("-" * (self.size * 2))

    def close(self):
        pass