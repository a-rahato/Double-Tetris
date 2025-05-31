"""
train.py: 训练脚本，仅支持 DQN 代理
"""
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import trange
from datetime import datetime

# 修改：引入 DoubleTetrisGame
from tetris.game import TetrisGame, DoubleTetrisGame  
from agents.dqn_agent import DQNAgent


def train(agent, episodes=1000, render=False):
    # 分别存储两个棋盘的得分
    scores1, scores2 = [], []
    init_eps = agent.epsilon
    final_eps = getattr(agent, 'epsilon_min', 0.01)
    decay_epochs = episodes
    pbar = trange(episodes, desc='Train')

    for episode in pbar:
        # 修改：使用双棋盘环境
        env = DoubleTetrisGame()  
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, env)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done, env)
            state = next_state
            if render:
                env.render(mode='gui')
        # 分别获取左右棋盘得分
        s1, s2 = env.game1.score, env.game2.score
        total_score = s1 + s2
        scores1.append(s1)
        scores2.append(s2)
        agent.epsilon = final_eps + max(decay_epochs - (episode + 1), 0) * (init_eps - final_eps) / decay_epochs
        if hasattr(agent, 'save_weights') and (episode + 1) % 2500 == 0:
            history_dir = 'weights_history'
            os.makedirs(history_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = os.path.join(history_dir, f"{agent.__class__.__name__}_weights_{ts}_ep{episode+1}.pkl")
            agent.save_weights(fname)
            print(f"Episode {episode+1}: 保存权重到 {fname}")
        # 进度条显示总得分(sum_point)和总消行(sum_lines)
        sum_lines = env.game1.lines_cleared + env.game2.lines_cleared
        pbar.set_postfix({
            'sum_point': total_score,
            'sum_lines': sum_lines
        })
    return scores1, scores2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000, help='训练回合数')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    parser.add_argument('--hidden', type=int, default=32, help='隐藏层维度')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--epsilon', type=float, default=1.0, help='初始探索率')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='最低探索率')
    parser.add_argument('--memory_size', type=int, default=50000, help='经验回放容量')
    parser.add_argument('--batch_size', type=int, default=128, help='训练批量大小')
    parser.add_argument('--target_update', type=int, default=250, help='目标网络更新频率（步数）')
    args = parser.parse_args()

    agent = DQNAgent(lr=args.lr,
                     gamma=args.gamma,
                     epsilon=args.epsilon,
                     epsilon_min=args.epsilon_min,
                     memory_size=args.memory_size,
                     hidden_dim=args.hidden,
                     batch_size=args.batch_size,
                     target_update=args.target_update)
    # 修改：更改权重文件名
    weight_file = 'double_dqn_weights.pkl'  
    if os.path.exists(weight_file):
        agent.load_weights(weight_file)
        print(f"已加载 DQN 权重：{weight_file}")
    scores1, scores2 = train(agent, episodes=args.episodes, render=args.render)
    agent.save_weights(weight_file)
    print(f"已保存 DQN 权重到 {weight_file}")

    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 修改：绘图文件名与标题
    plot_path = os.path.join(plots_dir, f'double_dqn_train_{timestamp}.png')
    plt.figure()
    plt.plot(scores1, label='Game1')
    plt.plot(scores2, label='Game2')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Double Tetris DQN Training')
    plt.legend()
    plt.savefig(plot_path)
    print(f"已保存训练曲线到 {plot_path}")


if __name__ == '__main__':
    main()
