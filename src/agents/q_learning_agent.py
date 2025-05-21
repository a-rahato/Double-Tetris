"""
q_learning_agent.py: 近似 Q-learning 策略实现
"""
import random
import pickle
import numpy as np
from .base_agent import BaseAgent
from tetris.utils import extract_features
from tetris.pieces import PIECES
from tetris.game import TetrisGame

class QLearningAgent(BaseAgent):
    """使用线性特征近似的 Q-learning 实现"""
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, w_clip=100, decay=0.999, min_epsilon=0.05):
        self.alpha = alpha  # 学习率
        self.gamma = gamma
        self.epsilon = epsilon  # 初始探索率
        self.w_clip = w_clip  # 权重剪裁阈值
        self.decay = decay  # 每轮 epsilon 衰减系数
        self.min_epsilon = min_epsilon  # 最低 epsilon
        # 启发式初始权重（来源：经典 Tetris 特征线性评估）
        self.weights = {
            'aggregate_height': -0.510066,
            'holes': -0.35663,
            'bumpiness': -0.184483,
            'complete_lines': 0.760666,
            # 其他高级特征初始为 0
            'max_height': 0.0,
            'mean_height': 0.0,
            'var_height': 0.0,
            'row_transitions': 0.0,
            'col_transitions': 0.0,
            'well_sums': 0.0,
        }

    def get_q_value(self, state, action):
        feats = extract_features(state, action)
        q = 0.0
        for f, v in feats.items():
            q += self.weights.get(f, 0.0) * v
        return q

    def act(self, state, env: TetrisGame):
        """宏观 epsilon-greedy: 旋转+列位置+硬落，选择最佳放置"""
        # 生成候选宏观动作
        rotations = list(range(len(PIECES[env.current_piece])))
        macro_actions = []
        for rot in rotations:
            shape = PIECES[env.current_piece][rot]
            for x in range(env.width - shape.shape[1] + 1):
                macro_actions.append((rot, x))
        # 探索
        if random.random() < self.epsilon:
            return random.choice(macro_actions)
        # 评估每个宏观动作
        best_q, best_action = -float('inf'), None
        for rot, x in macro_actions:
            # 复制环境核心状态
            env_copy = TetrisGame(env.width, env.height)
            env_copy.board = env.board.copy()
            env_copy.current_piece = env.current_piece
            env_copy.current_rot = rot
            env_copy.current_x = x
            env_copy.current_y = 0
            # 一次放置
            next_state, _, done, _ = env_copy.step((rot, x))
            feats = extract_features(next_state)
            q_val = sum(self.weights.get(f, 0.0) * v for f, v in feats.items())
            if q_val > best_q:
                best_q, best_action = q_val, (rot, x)
        return best_action

    def learn(self, state, action, reward, next_state, done=False, env=None):
        """更新权重：基于宏观动作的 Q-learning"""
        # 计算当前 Q(s,a)
        feats = extract_features(state, action)
        q_sa = sum(self.weights.get(f, 0.0) * v for f, v in feats.items())
        # 计算 target
        if done:
            target = reward
        else:
            # 使用宏观动作列表估计下步 Q
            if env is None:
                raise ValueError('QLearningAgent.learn: env 必需用于 macro actions')
            # 构建候选宏观动作
            rotations = list(range(len(PIECES[env.current_piece])))
            macro_actions = [(rot, x)
                             for rot in rotations
                             for x in range(env.width - PIECES[env.current_piece][rot].shape[1] + 1)]
            next_q = -float('inf')
            for act_macro in macro_actions:
                # 复制环境核心状态
                env_copy = TetrisGame(env.width, env.height)
                env_copy.board = env.board.copy()
                env_copy.current_piece = env.current_piece
                env_copy.current_rot = act_macro[0]
                env_copy.current_x = act_macro[1]
                env_copy.current_y = 0
                # 执行宏观动作
                s2, _, done2, _ = env_copy.step(act_macro)
                feats2 = extract_features(s2, act_macro)
                q2 = sum(self.weights.get(f, 0.0) * v for f, v in feats2.items())
                if q2 > next_q:
                    next_q = q2
            if next_q == -float('inf'):
                next_q = 0.0
            target = reward + self.gamma * next_q
        # 更新权重
        for f, v in feats.items():
            old_w = self.weights.get(f, 0.0)
            delta = self.alpha * (target - q_sa) * v
            new_w = old_w + delta
            # 权重剪裁
            self.weights[f] = max(min(new_w, self.w_clip), -self.w_clip)

    def save_weights(self, file_path: str):
        """将当前权重字典保存到文件"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_weights(self, file_path: str):
        """从文件加载权重字典"""
        with open(file_path, 'rb') as f:
            self.weights = pickle.load(f)
