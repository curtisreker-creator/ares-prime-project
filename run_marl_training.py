import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Import our custom components
from environments.grid_world_env import GridWorldEnv
from agents.dqn_agent import DQNAgent # The upgraded, team-based agent

def main():
    """
    Main function for Multi-Agent RL using a shared-brain agent (parameter sharing).
    """
    print("Initializing Ares Prime MARL simulation (Parameter Sharing)...")

    # --- 1. Environment and Agent Setup ---
    env = GridWorldEnv(size=10, max_energy=50)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space[0].n
    num_agents = env.num_agents

    # Create a SINGLE agent instance to be shared by all agents
    agent = DQNAgent(state_size, action_size, num_agents, learning_rate=5e-5)
    agent.epsilon_decay = 20000

    # --- 2. Training Hyperparameters ---
    n_episodes = 15000
    batch_size = 128
    target_update_frequency = 20

    scores = []
    scores_window = deque(maxlen=100)
    best_avg_score = -np.inf

    print("Setup complete. Starting training...")
    # --- 3. Main Training Loop ---
    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        team_score = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # The single agent brain selects actions for all agents
            actions_tensor = agent.select_actions(state)
            actions_tuple = tuple(actions_tensor.squeeze().tolist())
            
            # The environment steps forward based on the joint action
            next_state, reward, terminated, truncated, _ = env.step(actions_tuple)
            
            team_score += reward
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            done = terminated or truncated
            
            if done:
                next_state_tensor = None
            else:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Store the joint experience in the single replay memory
            agent.memory.push(state, actions_tensor, next_state_tensor, reward_tensor, done)
            
            state = next_state_tensor
            
            # The single agent learns from the shared experiences
            agent.learn(batch_size)

        # Update the single target network
        if i_episode % target_update_frequency == 0:
            agent.update_target_net()

        # --- 4. Logging and Metrics ---
        scores_window.append(team_score)
        scores.append(team_score)
        
        # We will now print a clear update every 10 episodes
        if i_episode % 10 == 0:
            current_avg_score = np.mean(scores_window)
            print(f'Episode {i_episode}\tAverage Score: {current_avg_score:.2f}')
            
            # Checkpointing logic
            if current_avg_score > best_avg_score and len(scores_window) >= 100:
                print(f'New best score! {current_avg_score:.2f} > {best_avg_score:.2f}. Saving model...')
                torch.save(agent.policy_net.state_dict(), 'models/marl_brain_best.pth')
                best_avg_score = current_avg_score

    env.close()
    print(f"\nTraining finished. The best average score achieved was: {best_avg_score:.2f}")

    # --- 5. Plotting the Results ---
    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Team Score')
    plt.xlabel('Episode #')
    plt.title('Training Performance of MARL Agents (Shared Brain)')
    plt.show()

if __name__ == "__main__":
    main()