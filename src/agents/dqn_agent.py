import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .base_agent import BaseAgent
from tetris.pieces import PIECES
from tetris.game import TetrisGame, DoubleTetrisGame
import copy
from tetris.utils import extract_features


class DQNNet(nn.Module):
    """简单的全连接网络，用于估计 Q 值"""
    # def __init__(self, input_dim, hidden_dim, output_dim):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(input_dim, hidden_dim),
    #         nn.ReLU(),
    #         nn.Linear(hidden_dim, hidden_dim),
    #         nn.ReLU(),
    #         nn.Linear(hidden_dim, output_dim)
    #     )

    # def forward(self, x):
    #     return self.net(x)
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DQNAgent(BaseAgent):
    """基于 PyTorch 的 Deep Q-Network 代理，支持 Double Tetris"""
    def __init__(self, width=10, height=20,
                 lr=1e-4, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, memory_size=50000,
                 hidden_dim=32, batch_size=128,
                 target_update=250):
        self.width = width
        self.height = height
        # 假设双棋盘
        self.num_boards = 2
        # 特征维度（双棋盘特征拼接）
        dummy = np.zeros((height, width), dtype=int)
        feats = extract_features(dummy)
        self.feature_keys = sorted(feats.keys())
        self.state_dim = len(self.feature_keys) * self.num_boards
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = deque(maxlen=memory_size)
        self.hidden_dim = hidden_dim
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 策略网络和目标网络，输出单个 Q 值
        self.policy_net = DQNNet(self.state_dim, self.hidden_dim, 1).to(self.device)
        self.target_net = DQNNet(self.state_dim, self.hidden_dim, 1).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.step_count = 0

    def _encode_state(self, state):
        """拼接单/双棋盘特征：双棋盘为左右拼接，拆分后分别提特征再合并"""
        # state: np.ndarray, shape (h, w) 或 (h, 2*w)
        if isinstance(state, np.ndarray) and state.ndim == 2 and state.shape[1] == self.width * 2:
            left, right = state[:, :self.width], state[:, self.width:]
            boards = [left, right]
        else:
            boards = [state]
        vec_list = []
        for b in boards:
            feats = extract_features(b)
            vec_list += [feats[k] for k in self.feature_keys]
        arr = np.array(vec_list, dtype=np.float32)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def _actions(self, env: TetrisGame):
        """返回当前合法宏观动作列表，兼容单/双棋盘：
           双棋盘时取两侧动作集合的交集"""
        def single_actions(game):
            piece = game.current_piece
            width = game.width
            acts = []
            for rot in range(len(PIECES[piece])):
                shape = PIECES[piece][rot]
                for x in range(width - shape.shape[1] + 1):
                    acts.append((rot, x))
            return acts

        if isinstance(env, DoubleTetrisGame):
            a1 = set(single_actions(env.game1))
            a2 = set(single_actions(env.game2))
            return list(a1 & a2)
        else:
            return single_actions(env)

    def act(self, state, env: TetrisGame):
        """支持 DoubleTetrisGame 的 epsilon-greedy 策略"""
        valid_actions = self._actions(env)
        feats = []
        for action in valid_actions:
            env_copy = copy.deepcopy(env)
            next_state, _, done, _ = env_copy.step(action)
            feats.append(self._encode_state(next_state))
        next_states = torch.cat(feats, dim=0).to(self.device)

        if random.random() < self.epsilon:
            idx = random.randrange(len(valid_actions))
        else:
            with torch.no_grad():
                q_values = self.policy_net(next_states).squeeze(1)
                idx = int(q_values.argmax().item())
        return valid_actions[idx]

    def learn(self, state, action, reward, next_state, done, env=None):
        """存储经验并训练 DQN 网络"""
        # 提取特征
        state_feat = self._encode_state(state)
        next_feat = self._encode_state(next_state)
        self.memory.append((state_feat, reward, next_feat, done))
        if len(self.memory) < self.batch_size:
            return
        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        state_batch, reward_batch, next_batch, done_batch = zip(*batch)
        state_batch = torch.cat(state_batch)
        next_batch = torch.cat(next_batch)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)
        # 当前 Q(s)
        q_values = self.policy_net(state_batch).squeeze(1)
        # 目标 Q
        with torch.no_grad():
            q_next = self.target_net(next_batch).squeeze(1)
        target_q = reward_batch + (1 - done_batch) * self.gamma * q_next
        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 同步目标网络
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_weights(self, file_path: str):
        """保存主网络参数"""
        torch.save(self.policy_net.state_dict(), file_path)

    def load_weights(self, file_path: str):
        """加载参数到主/目标网络"""
        # 加载到策略网络
        self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
        # 同步目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
