# filename: src/train_gen.py
# Replace with updated training logic for generator
import argparse
import os
from datetime import datetime
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt

from tetris.game import TetrisGame
from tetris.pieces import PIECES
from agents.dqn_agent import DQNAgent
from agents.gen_agent import GenAgent

class GenTetrisGame(TetrisGame):
    """Tetris environment where next piece is chosen by generator"""
    def __init__(self, generator, width=10, height=20):
        # set generator before calling super to ensure availability in reset
        self.generator = generator
        self.current_piece_idx = None
        super().__init__(width, height)

    def _spawn_piece(self):
        # choose next piece by generator
        # state before spawn: current board
        board_copy = self.board.copy()
        idx = self.generator.act(board_copy, self)
        self.current_piece_idx = idx
        # set current_piece
        keys = list(PIECES.keys())
        self.current_piece = keys[idx]
        self.current_rot = 0
        shape = PIECES[self.current_piece][self.current_rot]
        self.current_y = 0
        self.current_x = (self.width - shape.shape[1]) // 2
        if self._collision():
            self.done = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_min', type=float, default=0.01)
    parser.add_argument('--memory_size', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--target_update', type=int, default=250)
    parser.add_argument('--render', action='store_true', help='是否渲染')
    # DQN agent training parameters
    parser.add_argument('--dqn_epsilon', type=float, default=1.0, help='DQN 初始探索率')
    parser.add_argument('--dqn_epsilon_min', type=float, default=0.01, help='DQN 最低探索率')
    parser.add_argument('--dqn_decay', type=float, default=0.995, help='DQN epsilon 衰减')
    args = parser.parse_args()
    # initialize agents
    # DQN agent 支持在线训练
    dqn = DQNAgent(lr=args.lr, gamma=args.gamma, epsilon=args.dqn_epsilon, epsilon_min=args.dqn_epsilon_min,
                   memory_size=args.memory_size, batch_size=args.batch_size, target_update=args.target_update)
    if os.path.exists('dqn_weights.pkl'):
        dqn.load_weights('dqn_weights.pkl')
    gen = GenAgent(lr=args.lr, gamma=args.gamma, epsilon=args.epsilon,
                   epsilon_min=args.epsilon_min, memory_size=args.memory_size,
                   batch_size=args.batch_size, target_update=args.target_update)
    # train generator with stepwise reward: penalize player's gain and longer survival
    scores = []  # cumulative generator rewards per episode
    dqn_scores = []  # player scores per episode
    pbar = trange(args.episodes, desc='TrainGen')
    for ep in pbar:
        env = GenTetrisGame(generator=gen)
        state = env.reset()
        done = False
        episode_reward = 0.0
        step_penalty = 0.5
        while not done:
            state_old = state
            piece_idx = env.current_piece_idx
            # DQN agent 决策并执行
            action = dqn.act(state_old, env)
            next_state, reward, done, _ = env.step(action)
            # DQN agent 在线学习
            dqn.learn(state_old, action, reward, next_state, done, env)
            # 渲染
            if args.render:
                env.render(mode='gui')
            # 生成器奖励：负玩家 reward 和步惩罚
            r_gen = -reward - step_penalty
            episode_reward += r_gen
            gen.learn(state_old, piece_idx, r_gen, next_state, done)
            state = next_state
        # epsilon 衰减
        gen.update_epsilon(decay=0.995, min_eps=args.epsilon_min)
        dqn.epsilon = max(dqn.epsilon * args.dqn_decay, args.dqn_epsilon_min)
        scores.append(episode_reward)
        dqn_scores.append(env.score)
        pbar.set_postfix({'GenReward': episode_reward, 'DQN_epsilon': dqn.epsilon})
    # save generator weights
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('weights_history', exist_ok=True)
    gen.save_weights(f'weights_history/gen_weights_{ts}.pkl')
    # 同步保存最新生成器权重到根目录，便于后续加载和继续训练
    gen.save_weights('gen_weights.pkl')
    print('已保存生成器最新权重到 gen_weights.pkl')
    # 同步保存最新玩家 DQN 权重到根目录
    dqn.save_weights('dqn_weights.pkl')
    print('已保存DQN最新权重到 dqn_weights.pkl')
    print('Generator training completed.')
    # 绘制并保存训练曲线（Generator Reward 与 DQN Player Score）于同一图像的两个子图
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    # 上半部分：生成器奖励曲线
    plt.subplot(2, 1, 1)
    plt.plot(scores, color='tab:blue', label='Gen Reward')
    plt.title('Generator Training')
    plt.xlabel('Episode')
    plt.ylabel('Gen Reward')
    # 下半部分：DQN 玩家得分曲线
    plt.subplot(2, 1, 2)
    plt.plot(dqn_scores, color='tab:orange', label='DQN Score')
    plt.title('DQN Player Score During GAN-style')
    plt.xlabel('Episode')
    plt.ylabel('DQN Score')
    plt.tight_layout()
    plot_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_path = os.path.join(plots_dir, f'combined_train_{plot_ts}.png')
    plt.savefig(combined_path)
    print(f'已保存组合训练曲线到 {combined_path}')
