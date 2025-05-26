"""
train.py: 训练脚本，仅支持 DQN 代理
"""
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import trange
from datetime import datetime  # 用于生成时间戳

from tetris.game import TetrisGame
from agents.dqn_agent import DQNAgent  # DQN 代理


def train(agent, episodes=1000, render=False):
    scores = []
    # epsilon 线性衰减参数
    init_eps = agent.epsilon
    final_eps = getattr(agent, 'epsilon_min', 0.01)
    decay_epochs = episodes
    pbar = trange(episodes, desc='Train')  # 拿到进度条对象

    # 训练循环，每一轮使用 episode 计数
    for episode in pbar:
        env = TetrisGame()
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            # 使用环境调用宏观动作接口
            action = agent.act(state, env)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done, env)
            state = next_state
            if render:
                env.render(mode='gui')
        scores.append(env.score)
        # epsilon 线性衰减（基于当前 episode）
        agent.epsilon = final_eps + max(decay_epochs - (episode + 1), 0) * (init_eps - final_eps) / decay_epochs
        # 每 2500 轮保存一次权重到历史文件夹
        if hasattr(agent, 'save_weights') and (episode + 1) % 2500 == 0:
            history_dir = 'weights_history'
            os.makedirs(history_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = os.path.join(history_dir, f"{agent.__class__.__name__}_weights_{ts}_ep{episode+1}.pkl")
            agent.save_weights(fname)
            print(f"Episode {episode+1}: 保存权重到 {fname}")
        pbar.set_postfix({
            'Score': env.score,
            'Lines': env.lines_cleared
        })
    return scores


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
    # 初始化 DQN 代理
    agent = DQNAgent(lr=args.lr,
                     gamma=args.gamma,
                     epsilon=args.epsilon,
                     epsilon_min=args.epsilon_min,
                     memory_size=args.memory_size,
                     hidden_dim=args.hidden,
                     batch_size=args.batch_size,
                     target_update=args.target_update)
    weight_file = 'dqn_weights.pkl'
    if os.path.exists(weight_file):
        agent.load_weights(weight_file)
        print(f"已加载 DQN 权重：{weight_file}")
    # 开始训练
    scores = train(agent, episodes=args.episodes, render=args.render)
    # 保存最新权重
    agent.save_weights(weight_file)
    print(f"已保存 DQN 权重到 {weight_file}")
    # 绘制并保存训练曲线
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(plots_dir, f'dqn_train_{timestamp}.png')
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training')
    plt.savefig(plot_path)
    print(f"已保存训练曲线到 {plot_path}")


if __name__ == '__main__':
    main()
