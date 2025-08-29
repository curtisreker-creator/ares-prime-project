# main_executor.py (Upgraded for Phase 3)

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import argparse

from environments.ares_environment_bridge import AresEnv

# --- NEURAL NETWORK ARCHITECTURE (UPGRADED FOR DICT OBSERVATIONS) ---
class PPO_Model(nn.Module):
    def __init__(self, env):
        super(PPO_Model, self).__init__()
        # Extract shapes and sizes from the environment's spaces
        self.agent_pos_shape = np.prod(env.observation_space["agent_pos"].shape).item()
        self.agent_energy_shape = np.prod(env.observation_space["agent_energy"].shape).item()
        self.agent_inv_shape = np.prod(env.observation_space["agent_inventory"].shape).item()
        
        # A simple network to process the concatenated "agent" features
        combined_agent_features = self.agent_pos_shape + self.agent_energy_shape + self.agent_inv_shape
        self.agent_feature_extractor = nn.Sequential(nn.Linear(combined_agent_features, 64), nn.Tanh())
        
        # The main body of the network now takes the processed agent features
        # plus the global state features (refinery stock, construction progress)
        self.main_body_input_size = 64 + 1 + 1 
        
        self.critic = nn.Sequential(
            nn.Linear(self.main_body_input_size, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(self.main_body_input_size, 64), nn.Tanh(),
            nn.Linear(64, env.action_space[0].n * 2) # Actions for 2 agents
        )
        self.action_dims = env.action_space[0].n

    def forward(self, x):
        # x is now a dictionary of observations
        agent_pos = x["agent_pos"].reshape(-1, self.agent_pos_shape).float()
        agent_energy = x["agent_energy"].reshape(-1, self.agent_energy_shape).float()
        agent_inv = x["agent_inventory"].reshape(-1, self.agent_inv_shape).float()
        
        # Process and combine agent-specific features
        agent_features = torch.cat([agent_pos, agent_energy, agent_inv], dim=1)
        processed_agent_features = self.agent_feature_extractor(agent_features)
        
        # Combine with global features
        refinery_stock = x["refinery_stock"].reshape(-1, 1).float()
        construction_prog = x["construction_progress"].reshape(-1, 1).float()
        
        combined_state = torch.cat([processed_agent_features, refinery_stock, construction_prog], dim=1)
        
        # Actor-Critic outputs
        value = self.critic(combined_state)
        logits = self.actor(combined_state)
        logits_agent1 = logits[:, :self.action_dims]
        logits_agent2 = logits[:, self.action_dims:]
        return value, logits_agent1, logits_agent2

# --- AGENT CLASS (LOGIC REMAINS SIMILAR, BUT HANDLES DICT OBS) ---
class MultiAgentBrain:
    def __init__(self, env, learning_rate=0.0005):
        self.env = env
        self.model = PPO_Model(env) # Pass the whole env to the model now
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        print("Functional Multi-Agent Brain Initialized for Phase 3.")

    def learn(self, total_timesteps):
        print(f"Starting Phase 3 training for {total_timesteps} timesteps...")
        obs, info = self.env.reset()
        for step in range(total_timesteps):
            # Convert dictionary of numpy arrays to dictionary of tensors
            obs_tensor = {key: torch.tensor(value).unsqueeze(0) for key, value in obs.items()}
            
            with torch.no_grad():
                value, logits1, logits2 = self.model(obs_tensor)
            
            # Action selection remains the same
            probs1 = Categorical(logits=logits1)
            action1 = probs1.sample()
            probs2 = Categorical(logits=logits2)
            action2 = probs2.sample()
            action = (action1.item(), action2.item())

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Learning step also needs to handle the dictionary observation
            next_obs_tensor = {key: torch.tensor(value).unsqueeze(0) for key, value in next_obs.items()}
            new_value, new_logits1, new_logits2 = self.model(obs_tensor)
            
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
                print(f"Episode finished at step {step+1}. Final Reward: {reward}")
                obs, info = self.env.reset()
        print("Training complete.")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Phase 3 model saved to {path}")

# --- MAIN EXECUTION BLOCK (UNCHANGED) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ares Prime PPO training.")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--run_name", type=str, default="phase3_run", help="A name for this specific run.")
    args = parser.parse_args()

    print(f"Execution Started: Launching Run '{args.run_name}' with LR={args.learning_rate}")
    
    config = {"name": "Phase 3 Training"}
    env = AresEnv(config=config)
    agent_brain = MultiAgentBrain(env, learning_rate=args.learning_rate)
    
    agent_brain.learn(total_timesteps=200000) # Increased timesteps for more complex task
    agent_brain.save(f"models/{args.run_name}.pth")
    
    print(f"\nExecution of run '{args.run_name}' finished.")