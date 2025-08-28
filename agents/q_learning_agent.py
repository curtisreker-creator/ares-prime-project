import gymnasium as gym
import numpy as np
import time
from environments.grid_world_env import GridWorldEnv # Import our final environment

def train_agent():
    """
    Trains the Q-learning agent to manage energy.
    """
    # 1. Initialize the environment
    env = GridWorldEnv(size=10, max_energy=30)

    # 2. Set Hyperparameters (adjusted for a very large state space)
    learning_rate = 0.1         # Alpha
    discount_factor = 0.99      # Gamma
    epsilon = 1.0               # Initial exploration rate
    
    # State space is much larger, requiring significantly more episodes
    n_episodes = 80000
    epsilon_decay_rate = 0.00002 # Slower decay for more exploration
    min_epsilon = 0.01

    # 3. Initialize the Q-table
    # Dimensions: (rows, cols, has_resource, energy_levels, actions)
    q_table_shape = (
        env.size,
        env.size,
        2, # has_resource: 0 or 1
        env.max_energy + 1, # energy: 0 to max_energy
        env.action_space.n
    )
    q_table = np.zeros(q_table_shape)

    print("--- Training Started on Energy-Constrained Environment ---")
    
    # 4. The Training Loop
    for episode in range(n_episodes):
        state, info = env.reset()
        state = tuple(state)
        terminated = False
        truncated = False # New episode-ending condition

        while not terminated and not truncated:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample() # Explore
            else:
                action = np.argmax(q_table[state]) # Exploit

            # Take action
            new_state, reward, terminated, truncated, info = env.step(action)
            new_state = tuple(new_state)

            # Update Q-table
            q_table[state][action] = q_table[state][action] + learning_rate * (
                reward + discount_factor * np.max(q_table[new_state]) - q_table[state][action]
            )
            
            state = new_state

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)
        
        if (episode + 1) % 8000 == 0:
            print(f"Episode: {episode + 1}/{n_episodes} | Epsilon: {epsilon:.4f}")

    print("--- Training Finished ---")
    return q_table, env

def watch_agent(q_table, env):
    """
    Watches the trained agent perform its task.
    """
    print("\n--- Watching Trained Agent ---")
    for episode in range(3):
        state, info = env.reset()
        env.render()
        time.sleep(1)
        
        terminated = False
        truncated = False
        step_count = 0
        
        while not terminated and not truncated:
            action = np.argmax(q_table[tuple(state)])
            state, reward, terminated, truncated, info = env.step(action)
            
            step_count += 1
            env.render()
            print(f"Step: {step_count} | Action: {action} | Reward: {reward:.1f}")
            time.sleep(0.2)
        
        if terminated:
            print(f"Episode {episode + 1} finished successfully in {step_count} steps.\n")
        elif truncated:
            print(f"Episode {episode + 1} failed due to energy depletion after {step_count} steps.\n")


if __name__ == "__main__":
    trained_q_table, environment = train_agent()
    watch_agent(trained_q_table, environment)

    