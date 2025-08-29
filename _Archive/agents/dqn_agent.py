import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

# --- 1. DQN Model (Now Multi-Headed) ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size, num_agents=2):
        super(DQN, self).__init__()
        self.num_agents = num_agents
        self.action_size = action_size
        
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        # Output layer is now larger to accommodate actions for all agents
        self.layer3 = nn.Linear(128, action_size * num_agents)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        # Reshape the output to (batch_size, num_agents, action_size)
        q_values = self.layer3(x)
        return q_values.view(q_values.size(0), self.num_agents, self.action_size)

# --- 2. Replay Memory (Now stores joint actions) ---
Transition = namedtuple('Transition',
                        ('state', 'actions', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# --- 3. DQN Agent (Now for a Team) ---
class DQNAgent:
    def __init__(self, state_size, action_size, num_agents=2, learning_rate=1e-4, gamma=0.99, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.gamma = gamma

        # A single brain for the whole team
        self.policy_net = DQN(state_size, action_size, num_agents)
        self.target_net = DQN(state_size, action_size, num_agents)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayMemory(memory_size)
        
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 2000
        self.steps_done = 0

    def select_actions(self, state): # Note: plural actions
        sample = random.random()
        epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        if sample > epsilon_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.max(2)[1]
        else:
            actions = [random.randrange(self.action_size) for _ in range(self.num_agents)]
            return torch.tensor([actions], dtype=torch.long)

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.actions)
        reward_batch = torch.cat(batch.reward)
        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        
        action_batch = action_batch.unsqueeze(-1)
        state_action_values = self.policy_net(state_batch).gather(2, action_batch)

        next_state_values = torch.zeros(batch_size, self.num_agents)
        with torch.no_grad():
            best_next_actions = self.policy_net(non_final_next_states).max(2)[1].unsqueeze(-1)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(2, best_next_actions).squeeze(-1)
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.unsqueeze(1)

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(-1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())