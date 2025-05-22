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
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, w_clip=1000, decay=0.9995, min_epsilon=0.01):
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

    def _value_from_state(self, board):
        """基于 board 特征计算价值"""
        feats = extract_features(board)
        return sum(self.weights.get(f, 0.0) * v for f, v in feats.items())

    def act(self, state, env: TetrisGame):
        """宏观 epsilon-greedy: 模拟放置后基于 next_state 特征选择最佳放置"""
        # 生成候选宏观动作
        rotations = list(range(len(PIECES[env.current_piece])))
        macro_actions = [(rot, x)
                         for rot in rotations
                         for x in range(env.width - PIECES[env.current_piece][rot].shape[1] + 1)]
        # 探索
        if random.random() < self.epsilon:
            return random.choice(macro_actions)
        # 评估所有动作：模拟执行并计算 next_state 价值
        best_q, best_action = -float('inf'), None
        for action in macro_actions:
            # 复制环境核心状态
            env_copy = TetrisGame(env.width, env.height)
            env_copy.board = env.board.copy()
            env_copy.current_piece = env.current_piece
            # 执行动作并获取 next_state_board
            next_board, _, done, _ = env_copy.step(action)
            q_val = self._value_from_state(next_board)
            if q_val > best_q:
                best_q, best_action = q_val, action
        return best_action

    def learn(self, state, action, reward, next_state, done=False, env=None):
        """更新权重：基于 macro 动作后状态的 TD 学习"""
        # 提取 next_state 特征，并计算当前估计价值
        feats_next = extract_features(next_state)
        q_sa = sum(self.weights.get(f, 0.0) * v for f, v in feats_next.items())
        # 计算 target
        if done:
            target = reward
        else:
            # 在 next_state 上评估所有后续动作的价值
            rotations = list(range(len(PIECES[env.current_piece])))
            macro_actions = [(rot, x)
                             for rot in rotations
                             for x in range(env.width - PIECES[env.current_piece][rot].shape[1] + 1)]
            next_q = -float('inf')
            for a in macro_actions:
                # 模拟环境
                env_copy = TetrisGame(env.width, env.height)
                env_copy.board = next_state.copy()
                env_copy.current_piece = env.current_piece
                # 执行动作
                ns2, _, done2, _ = env_copy.step(a)
                v2 = sum(self.weights.get(f, 0.0) * v for f, v in extract_features(ns2).items())
                next_q = max(next_q, v2)
            if next_q == -float('inf'):
                next_q = 0.0
            target = reward + self.gamma * next_q
        # 更新权重：TD(0)
        delta = target - q_sa
        for f, v in feats_next.items():
            new_w = self.weights.get(f, 0.0) + self.alpha * delta * v
            self.weights[f] = max(min(new_w, self.w_clip), -self.w_clip)

    def save_weights(self, file_path: str):
        """将当前权重字典保存到文件"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_weights(self, file_path: str):
        """从文件加载权重字典"""
        with open(file_path, 'rb') as f:
            self.weights = pickle.load(f)
