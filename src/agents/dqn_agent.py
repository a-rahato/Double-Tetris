import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .base_agent import BaseAgent
from tetris.pieces import PIECES
from tetris.game import TetrisGame
from tetris.utils import extract_features
import copy


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
    """基于 PyTorch 的 Deep Q-Network 代理"""
    def __init__(self, width=10, height=20,
                 lr=1e-4, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, memory_size=50000,
                 hidden_dim=32, batch_size=128,
                 target_update=250):
        self.width = width
        self.height = height
        # 特征维度
        dummy = np.zeros((height, width), dtype=int)
        feats = extract_features(dummy)
        self.feature_keys = sorted(feats.keys())
        self.state_dim = len(self.feature_keys)
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

    def _encode_state(self, board):
        """基于 extract_features 提取特征向量"""
        feats = extract_features(board)
        vec = np.array([feats[k] for k in self.feature_keys], dtype=np.float32)
        return torch.from_numpy(vec).unsqueeze(0).to(self.device)

    def _actions(self, env: TetrisGame):
        # 返回当前合法宏观动作列表
        rotations = list(range(len(PIECES[env.current_piece])))
        actions = []
        for rot in rotations:
            shape = PIECES[env.current_piece][rot]
            for x in range(env.width - shape.shape[1] + 1):
                actions.append((rot, x))
        return actions

    def act(self, state, env: TetrisGame):
        """基于枚举宏观动作的 epsilon-greedy 策略"""
        # 枚举所有合法宏观动作
        valid_actions = self._actions(env)
        # 计算每个动作对应的下一个状态特征
        feats = []
        for action in valid_actions:
            # 手动拷贝核心游戏状态，避免 deepcopy Pygame 对象
            env_copy = TetrisGame(env.width, env.height)
            env_copy.board = env.board.copy()
            env_copy.current_piece = env.current_piece
            env_copy.current_rot = env.current_rot
            env_copy.current_x = env.current_x
            env_copy.current_y = env.current_y
            env_copy.done = env.done
            env_copy.lines_cleared = env.lines_cleared
            env_copy.score = env.score
            next_obs, _, done, _ = env_copy.step(action)
            feat = extract_features(next_obs)
            vec = np.array([feat[k] for k in self.feature_keys], dtype=np.float32)
            feats.append(torch.from_numpy(vec).unsqueeze(0))
        next_states = torch.cat(feats, dim=0).to(self.device)
        # epsilon-greedy 选择
        if random.random() < self.epsilon:
            idx = random.randint(0, len(valid_actions) - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(next_states).squeeze(1)
                idx = int(torch.argmax(q_values).cpu().item())
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
