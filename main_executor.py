# main_executor.py (Modified for Hyperparameter Sweeps)
# Ares Prime - Phase 3, Experiment Batch 1

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import argparse # <-- ADDED FOR COMMAND-LINE ARGUMENTS

from environments.ares_environment_bridge import AresEnv

# --- NEURAL NETWORK ARCHITECTURE (UNCHANGED) ---
class PPO_Model(nn.Module):
    def __init__(self, obs_space_shape, action_space_dims):
        super(PPO_Model, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_space_shape, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(obs_space_shape, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_space_dims * 2)
        )
        self.action_dims = action_space_dims

    def forward(self, x):
        value = self.critic(x)
        logits = self.actor(x)
        logits_agent1 = logits[:, :self.action_dims]
        logits_agent2 = logits[:, self.action_dims:]
        return value, logits_agent1, logits_agent2

# --- AGENT CLASS WITH PPO LEARNING LOGIC ---
class MultiAgentBrain:
    # MODIFIED to accept learning_rate
    def __init__(self, env, learning_rate=0.0005):
        self.env = env
        self.obs_shape = np.prod(env.observation_space.shape).item()
        self.action_dims = env.action_space[0].n
        self.model = PPO_Model(self.obs_shape, self.action_dims)
        # Use the learning_rate passed from the command line
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        print(f"Functional Multi-Agent Brain Initialized. Learning Rate: {learning_rate}")

    def learn(self, total_timesteps):
        print(f"Starting REAL training for {total_timesteps} timesteps...")
        obs, info = self.env.reset()
        for step in range(total_timesteps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                value, logits1, logits2 = self.model(obs_tensor)
            
            probs1 = Categorical(logits=logits1)
            action1 = probs1.sample()
            probs2 = Categorical(logits=logits2)
            action2 = probs2.sample()
            action = (action1.item(), action2.item())

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            new_value, new_logits1, new_logits2 = self.model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            advantage = reward - new_value.item()
            log_prob1 = Categorical(logits=new_logits1).log_prob(action1)
            log_prob2 = Categorical(logits=new_logits2).log_prob(action2)
            actor_loss = -(log_prob1 + log_prob2) * advantage
            critic_loss = nn.functional.mse_loss(new_value, torch.tensor([[reward]], dtype=torch.float32))
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            obs = next_obs
            if terminated or truncated:
                print(f"Episode finished at step {step+1}. Reward: {reward}")
                obs, info = self.env.reset()
        print("Training complete.")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Multi-agent model saved to {path}")

# --- MAIN EXECUTION BLOCK (MODIFIED FOR ARGUMENTS) ---
if __name__ == "__main__":
    # --- ADDED: Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="Run Ares Prime PPO training.")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--run_name", type=str, default="default_run", help="A name for this specific run, used for saving the model.")
    args = parser.parse_args()

    print(f"Execution Started: Launching Run '{args.run_name}' with LR={args.learning_rate}")
    
    config = {"name": "The Strategist - Emergent Intelligence"}
    env = AresEnv(config=config)
    # Pass the learning rate from the arguments to the agent's brain
    agent_brain = MultiAgentBrain(env, learning_rate=args.learning_rate)
    
    agent_brain.learn(total_timesteps=100000)
    # Save the model with a unique name for this run
    agent_brain.save(f"models/{args.run_name}.pth")
    
    print(f"\nExecution of run '{args.run_name}' finished.")