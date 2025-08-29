# train_worker.py (Version 2.1 - Complete and Corrected)

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import argparse
from enum import Enum
import os
import csv
from torch.utils.tensorboard import SummaryWriter

from environments.ares_environment_bridge import AresEnv

# --- High-Level Goals ---
class HighLevelGoal(Enum):
    DEPOSIT_ORE_AT_REFINERY = 0
    GET_ORE_TO_REFINERY_DEPOSIT = 1

# --- PPO Model (The "Actor-Critic" Brain) ---
class PPO_Model(nn.Module):
    def __init__(self, env):
        super(PPO_Model, self).__init__()
        self.agent_pos_shape = np.prod(env.observation_space["agent_pos"].shape).item()
        self.agent_energy_shape = np.prod(env.observation_space["agent_energy"].shape).item()
        self.agent_inv_shape = np.prod(env.observation_space["agent_inventory"].shape).item()
        combined_agent_features = self.agent_pos_shape + self.agent_energy_shape + self.agent_inv_shape
        self.agent_feature_extractor = nn.Sequential(nn.Linear(combined_agent_features, 64), nn.Tanh())
        self.main_body_input_size = 64 + 1 + 1 
        self.critic = nn.Sequential(nn.Linear(self.main_body_input_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.actor = nn.Sequential(nn.Linear(self.main_body_input_size, 64), nn.Tanh(), nn.Linear(64, env.action_space[0].n * 2))
        self.action_dims = env.action_space[0].n
    def forward(self, x):
        agent_pos = x["agent_pos"].reshape(-1, self.agent_pos_shape).float()
        agent_energy = x["agent_energy"].reshape(-1, self.agent_energy_shape).float()
        agent_inv = x["agent_inventory"].reshape(-1, self.agent_inv_shape).float()
        agent_features = torch.cat([agent_pos, agent_energy, agent_inv], dim=1)
        processed_agent_features = self.agent_feature_extractor(agent_features)
        refinery_stock = x["refinery_stock"].reshape(-1, 1).float()
        construction_prog = x["construction_progress"].reshape(-1, 1).float()
        combined_state = torch.cat([processed_agent_features, refinery_stock, construction_prog], dim=1)
        value = self.critic(combined_state)
        logits = self.actor(combined_state)
        logits_agent1 = logits[:, :self.action_dims]
        logits_agent2 = logits[:, self.action_dims:]
        return value, logits_agent1, logits_agent2

# --- Intrinsic Curiosity Module ---
class ICM(nn.Module):
    def __init__(self, env):
        super(ICM, self).__init__()
        obs_shape = int(sum(np.prod(space.shape) if space.shape else 1 for space in env.observation_space.values()))
        num_actions_per_agent = env.action_space[0].n
        self.joint_action_count = num_actions_per_agent ** env.sim.num_agents
        self.feature_encoder = nn.Sequential(nn.Linear(obs_shape, 128), nn.ReLU())
        self.inverse_model = nn.Sequential(nn.Linear(128 * 2, 128), nn.ReLU(), nn.Linear(128, self.joint_action_count))
        self.forward_model = nn.Sequential(nn.Linear(128 + self.joint_action_count, 128), nn.ReLU(), nn.Linear(128, 128))
    def forward(self, state, next_state, joint_action_one_hot):
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)
        predicted_action_logits = self.inverse_model(torch.cat((state_feat, next_state_feat), dim=1))
        predicted_next_state_feat = self.forward_model(torch.cat((state_feat, joint_action_one_hot), dim=1))
        return predicted_action_logits, predicted_next_state_feat, next_state_feat

# --- Worker Brain, with ICM ---
class WorkerBrain:
    def __init__(self, env, learning_rate=0.0005):
        self.env = env
        self.model = PPO_Model(env)
        self.icm = ICM(env)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.icm.parameters()), lr=learning_rate)
        self.num_actions_per_agent = env.action_space[0].n
        print("Worker Brain Initialized with ICM.")
    def _obs_to_flat_tensor(self, obs):
        obs_list = [v.flatten() if isinstance(v, np.ndarray) else [v] for v in obs.values()]
        return torch.tensor(np.concatenate(obs_list), dtype=torch.float32).unsqueeze(0)
    def learn(self, total_timesteps, writer, csv_writer):
        print(f"Starting worker training for {total_timesteps} timesteps...")
        obs, info = self.env.reset()
        current_episode, current_episode_reward, current_episode_length = 0, 0, 0
        for step in range(total_timesteps):
            obs_tensor_dict = {key: torch.tensor(value).unsqueeze(0) for key, value in obs.items()}
            with torch.no_grad():
                value, logits1, logits2 = self.model(obs_tensor_dict)
            probs1 = Categorical(logits=logits1); action1 = probs1.sample()
            probs2 = Categorical(logits=logits2); action2 = probs2.sample()
            action_tuple = (action1.item(), action2.item())
            next_obs, extrinsic_reward, terminated, truncated, info = self.env.step(action_tuple)
            current_episode_reward += extrinsic_reward; current_episode_length += 1
            state_flat = self._obs_to_flat_tensor(obs); next_state_flat = self._obs_to_flat_tensor(next_obs)
            joint_action_id = action_tuple[0] * self.num_actions_per_agent + action_tuple[1]; joint_action_tensor = torch.tensor([joint_action_id])
            joint_action_one_hot = torch.nn.functional.one_hot(joint_action_tensor, num_classes=self.icm.joint_action_count).float()
            pred_action_logits, pred_next_state_feat, true_next_state_feat = self.icm(state_flat, next_state_flat, joint_action_one_hot)
            intrinsic_reward = nn.functional.mse_loss(pred_next_state_feat, true_next_state_feat).item()
            icm_loss = nn.functional.cross_entropy(pred_action_logits, joint_action_tensor) + nn.functional.mse_loss(pred_next_state_feat, true_next_state_feat)
            total_reward = extrinsic_reward + (0.1 * intrinsic_reward)
            new_value, new_logits1, new_logits2 = self.model(obs_tensor_dict)
            advantage = total_reward - new_value.item()
            log_prob1 = Categorical(logits=new_logits1).log_prob(action1); log_prob2 = Categorical(logits=new_logits2).log_prob(action2)
            actor_loss = -(log_prob1 + log_prob2) * advantage
            critic_loss = nn.functional.mse_loss(new_value, torch.tensor([[total_reward]], dtype=torch.float32))
            loss = actor_loss + critic_loss + icm_loss
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            obs = next_obs
            if terminated or truncated:
                print(f"Ep: {current_episode}, Timestep: {step}, Reward: {current_episode_reward:.2f}, Length: {current_episode_length}")
                writer.add_scalar("Reward/Episode Reward", current_episode_reward, current_episode)
                writer.add_scalar("Metrics/Episode Length", current_episode_length, current_episode)
                csv_writer.writerow([current_episode, step, current_episode_reward, current_episode_length])
                current_episode += 1; current_episode_reward = 0; current_episode_length = 0
                obs, info = self.env.reset()
        print("Worker training complete.")
    def save(self, path): torch.save(self.model.state_dict(), path)
    def load_model(self, path): self.model.load_state_dict(torch.load(path))

# --- Specialized Environment for Drills ---
class WorkerDrillEnv(AresEnv):
    def __init__(self, config, goal: HighLevelGoal):
        super().__init__(config)
        self.goal = goal
        print(f"Worker Drill Environment created. Current Goal: {self.goal.name}")
    def reset(self, seed=None, options=None):
        if self.goal == HighLevelGoal.DEPOSIT_ORE_AT_REFINERY:
            obs, info = super().reset(seed=seed, options=options)
            self.sim._agent_positions[0] = self.sim._refinery_pos.copy()
            self.sim._agent_inventories[0] = self.sim.ITEM_IRON_ORE
            self.sim._agent_positions[1] = np.array([-1, -1]) # Move agent 2 out of bounds
            return self.sim._get_obs(), info
        else:
            return super().reset(seed=seed, options=options)
    def step(self, action):
        old_obs = self.sim._get_obs()
        next_obs, _, terminated, truncated, info = super().step(action)
        drill_reward = 0
        if self.goal == HighLevelGoal.DEPOSIT_ORE_AT_REFINERY:
             if next_obs["refinery_stock"] > old_obs["refinery_stock"]:
                drill_reward += 50
        elif self.goal == HighLevelGoal.GET_ORE_TO_REFINERY_DEPOSIT:
            agent_has_ore = [inv == self.sim.ITEM_IRON_ORE for inv in old_obs["agent_inventory"]]
            for i in range(self.sim.num_agents):
                if agent_has_ore[i]:
                    old_dist = np.linalg.norm(old_obs["agent_pos"][i] - self.sim._refinery_pos)
                    new_dist = np.linalg.norm(next_obs["agent_pos"][i] - self.sim._refinery_pos)
                    if new_dist < old_dist:
                        drill_reward += 1
            if next_obs["refinery_stock"] > old_obs["refinery_stock"]:
                drill_reward += 50
        if drill_reward >= 50:
             terminated = True
        return next_obs, drill_reward, terminated, truncated, info

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Worker Competency Trials.")
    parser.add_argument("--goal", type=str, required=True, choices=[g.name for g in HighLevelGoal])
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--model_input", type=str, default=None)
    args = parser.parse_args()
    
    run_name = f"drill_{args.goal}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    tensorboard_log_dir = os.path.join("runs", run_name)
    csv_log_dir = os.path.join("logs", run_name)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(csv_log_dir, exist_ok=True)
    
    writer = SummaryWriter(tensorboard_log_dir)
    csv_file_path = os.path.join(csv_log_dir, "training_log.csv")
    csv_file = open(csv_file_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode_num", "timestep", "episode_reward", "episode_length"])

    goal_to_train = HighLevelGoal[args.goal]
    print(f"--- LAUNCHING WORKER DRILL: {goal_to_train.name} ---")

    config = {"name": "Worker Drill"}
    env = WorkerDrillEnv(config=config, goal=goal_to_train)
    worker_brain = WorkerBrain(env)
    
    if args.model_input:
        worker_brain.load_model(args.model_input)
        
    worker_brain.learn(total_timesteps=20000, writer=writer, csv_writer=csv_writer)
    
    writer.close()
    csv_file.close()
    
    output_path = os.path.join("models", os.path.basename(args.model_output))
    worker_brain.save(output_path)
    
    print(f"\n--- WORKER DRILL {goal_to_train.name} COMPLETE ---")
    print(f"Log file saved to: {csv_file_path}")