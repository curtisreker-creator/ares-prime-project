import time
import torch
import gymnasium as gym

# Import our custom components
from environments.grid_world_env import GridWorldEnv
from agents.dqn_agent import DQNAgent

def main():
    """
    Main function to load a trained agent and watch it perform.
    """
    print("Initializing Ares Prime evaluation...")

    # --- 1. Environment and Agent Setup ---
    env = GridWorldEnv(size=10, max_energy=30)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # --- 2. Load the Trained Model ---
    model_path = 'models/dqn_marvin_brain_best.pth'
    print(f"Loading trained model from: {model_path}")
    agent.policy_net.load_state_dict(torch.load(model_path))
    # Set the network to evaluation mode (important for consistent results)
    agent.policy_net.eval()
    print("Model loaded successfully.")

    # --- 3. Run Evaluation Episodes ---
    n_episodes = 5 # Let's watch for 5 episodes

    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated = False
        truncated = False
        step_count = 0

        print(f"\n--- Starting Episode {i_episode} ---")
        env.render()
        time.sleep(1) # Pause at the start

        while not terminated and not truncated:
            step_count += 1
            # --- CHOOSE ACTION DETERMINISTICALLY ---
            # No random exploration, always choose the best action
            with torch.no_grad():
                action = agent.policy_net(state).max(1)[1].view(1, 1)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            
            if not (terminated or truncated):
                state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            
            # Render the world
            env.render()
            print(f"Step: {step_count} | Action: {action.item()}")
            time.sleep(0.2) # Pause between steps

        if terminated:
            print(f"Episode {i_episode} finished successfully in {step_count} steps.")
        else:
            print(f"Episode {i_episode} failed after {step_count} steps.")

    env.close()

if __name__ == "__main__":
    main()