# filename: src/agents/gen_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .base_agent import BaseAgent
from tetris.pieces import PIECES
from tetris.utils import extract_features

class GenNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

class GenAgent(BaseAgent):
    """基于 DQN 的生成器 Agent，将生成下一块拼图类型以最小化玩家得分。"""
    def __init__(self, width=10, height=20,
                 lr=1e-4, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, memory_size=50000,
                 hidden_dim=32, batch_size=128,
                 target_update=250):
        # 特征维度
        dummy = np.zeros((height, width), dtype=int)
        feats = extract_features(dummy)
        self.feature_keys = sorted(feats.keys())
        self.state_dim = len(self.feature_keys)
        # 动作大小（7 种拼图）
        self.n_actions = len(PIECES)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = deque(maxlen=memory_size)
        self.hidden_dim = hidden_dim
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 策略和目标网络
        self.policy_net = GenNet(self.state_dim, hidden_dim, self.n_actions).to(self.device)
        self.target_net = GenNet(self.state_dim, hidden_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.step_count = 0
    
    def _encode_state(self, board):
        feats = extract_features(board)
        vec = np.array([feats[k] for k in self.feature_keys], dtype=np.float32)
        return torch.from_numpy(vec).unsqueeze(0).to(self.device)
    
    def act(self, state, env=None):
        """基于 epsilon-greedy 选择下一块拼图类型序号"""
        s = self._encode_state(state)
        if random.random() < self.epsilon:
            idx = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                q = self.policy_net(s)
                idx = int(q.argmax(dim=1).cpu().item())
        return idx  # 返回 0-6
    
    def learn(self, state, action, reward, next_state, done, env=None):
        """存储经验并更新网络"""
        s = self._encode_state(state)
        ns = self._encode_state(next_state)
        self.memory.append((s, action, reward, ns, done))
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        state_batch = torch.cat(states)
        next_batch = torch.cat(next_states)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)
        # 当前 Q
        q_values = self.policy_net(state_batch)
        q_val = q_values.gather(1, torch.tensor(actions, device=self.device).unsqueeze(1)).squeeze(1)
        # 目标 Q
        with torch.no_grad():
            q_next = self.target_net(next_batch).max(1)[0]
        target = reward_batch + (1 - done_batch) * self.gamma * q_next
        loss = self.loss_fn(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self, decay, min_eps):
        self.epsilon = max(self.epsilon * decay, min_eps)
    
    def save_weights(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)
    
    def load_weights(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
