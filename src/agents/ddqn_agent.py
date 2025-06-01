import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .base_agent import BaseAgent
from tetris.pieces import PIECES
from tetris.game import DoubleTetrisGame
from tetris.utils import extract_features
import copy


class DDQNNet(nn.Module):
    """估计 Q 值的全连接网络"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DDQNAgent(BaseAgent):
    """基于 Double DQN 的代理，适用于 DoubleTetrisGame"""
    def __init__(self, width=10, height=20,
                 lr=1e-4, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, memory_size=50000,
                 hidden_dim=64, batch_size=128,
                 target_update=250):
        self.width = width * 2  # 双区宽度
        self.height = height
        # 特征维度
        dummy = np.zeros((self.height, self.width), dtype=int)
        feats = extract_features(dummy)
        self.feature_keys = sorted(feats.keys())
        self.state_dim = len(self.feature_keys)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = deque(maxlen=memory_size)
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 网络
        self.policy_net = DDQNNet(self.state_dim, hidden_dim, 1).to(self.device)
        self.target_net = DDQNNet(self.state_dim, hidden_dim, 1).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.step_count = 0

    def _encode_state(self, board):
        feats = extract_features(board)
        # print(feats)
        vec = np.array([feats[k] for k in self.feature_keys], dtype=np.float32)
        return torch.from_numpy(vec).unsqueeze(0).to(self.device)

    def _actions(self, env: DoubleTetrisGame):
        # 首先按边界宽度生成候选宏动作
        p1, p2 = env.game1.current_piece, env.game2.current_piece
        len1, len2 = len(PIECES[p1]), len(PIECES[p2])
        max_rot = max(len1, len2)
        potential = []
        for rot in range(max_rot):
            shape1 = PIECES[p1][rot % len1]
            shape2 = PIECES[p2][rot % len2]
            w1, w2 = shape1.shape[1], shape2.shape[1]
            # 取两区均可放置的 x 上限
            x_max = min(env.game1.width - w1, env.game2.width - w2)
            for x in range(x_max + 1):
                potential.append((rot, x))
        # 过滤会导致任一区立即结束的动作
        valid = []
        for act in potential:
            # 深拷贝环境并应用动作
            env_copy = DoubleTetrisGame(env.game1.width, env.game1.height)
            env_copy.game1 = copy.deepcopy(env.game1)
            env_copy.game2 = copy.deepcopy(env.game2)
            _, _, done, _ = env_copy.step(act)
            if not done:
                valid.append(act)
        # 若过滤后无合法动作，则回退至所有候选
        if not valid:
            valid = potential
        return valid

    def act(self, state, env: DoubleTetrisGame):
        valid_actions = self._actions(env)
        feats = []
        for action in valid_actions:
            # 拷贝环境状态
            env_copy = DoubleTetrisGame(env.game1.width, env.game1.height)
            # 复制子游戏状态
            for i, g in enumerate([env.game1, env.game2]):
                setattr(env_copy, f'game{i+1}', copy.deepcopy(g))
            next_obs, _, done, _ = env_copy.step(action)
            vec = self._encode_state(next_obs)
            feats.append(vec)
        next_states = torch.cat(feats, dim=0)
        if random.random() < self.epsilon:
            idx = random.randint(0, len(valid_actions) - 1)
        else:
            with torch.no_grad():
                q_vals = self.policy_net(next_states).squeeze(1)
                idx = int(torch.argmax(q_vals).cpu().item())
        return valid_actions[idx]

    def learn(self, state, action, reward, next_state, done, env=None):
        state_feat = self._encode_state(state)
        next_feat = self._encode_state(next_state)
        # print(reward)
        # time.sleep(1)
        self.memory.append((state_feat, reward, next_feat, done))
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s_batch, r_batch, n_batch, d_batch = zip(*batch)
        s_batch = torch.cat(s_batch)
        n_batch = torch.cat(n_batch)
        r_batch = torch.tensor(r_batch, dtype=torch.float32, device=self.device)
        d_batch = torch.tensor(d_batch, dtype=torch.float32, device=self.device)
        # 当前 Q(s) 估计
        q_s = self.policy_net(s_batch).squeeze(1)
        with torch.no_grad():
            # 直接使用 target_net 估计下一状态的最大 Q 值
            q_next = self.target_net(n_batch).squeeze(1)
        # 计算目标：r + gamma * Q(next_state)
        target_q = r_batch + (1 - d_batch) * self.gamma * q_next
        loss = self.loss_fn(q_s, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_weights(self, file_path: str):
        torch.save(self.policy_net.state_dict(), file_path)

    def load_weights(self, file_path: str):
        self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
