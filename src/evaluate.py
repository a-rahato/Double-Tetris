"""
evaluate.py: 验证脚本，对比不同 Agent 表现并渲染示例游戏。
"""
import argparse
import os  # 用于加载权重
import matplotlib.pyplot as plt  # 用于绘制直方图
from datetime import datetime  # 用于生成时间戳
from tetris.game import TetrisGame
from agents.dqn_agent import DQNAgent  # 仅保留 DQN 验证


def evaluate(agent, episodes=10, render=False):
    scores = []
    line_counts = []
    for _ in range(episodes):
        env = TetrisGame()
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            # 使用环境和状态调用宏观动作接口
            action = agent.act(state, env)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if render:
                # 使用 GUI 模式渲染
                env.render(mode='gui')
        scores.append(total_reward)
        # 记录本轮消除行数
        line_counts.append(env.lines_cleared)
    return scores, line_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5, help='评估回合数')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    parser.add_argument('--hidden', type=int, default=32, help='隐藏层维度')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--epsilon', type=float, default=0.0, help='评估时探索率')
    args = parser.parse_args()
    # 初始化 DQN 代理并加载权重
    agent = DQNAgent(lr=args.lr, gamma=args.gamma, epsilon=args.epsilon, hidden_dim=args.hidden)
    weight_file = 'dqn_weights.pkl'
    if os.path.exists(weight_file):
        agent.load_weights(weight_file)
        print(f"已加载 DQN 权重：{weight_file}")
    # 评估
    scores, line_counts = evaluate(agent, episodes=args.episodes, render=args.render)
    # 输出平均分
    print(f"DQN 平均分: {sum(scores)/len(scores):.2f}")
    # 输出消除行数最多和最少的5次
    sorted_lines = sorted(line_counts)
    min5 = sorted_lines[:5]
    max5 = sorted_lines[-5:][::-1]
    print(f"消除行数最少的5次: {min5}")
    print(f"消除行数最多的5次: {max5}")
    # 绘制消行次数分布直方图
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure()
    bins = range(min(line_counts), max(line_counts) + 2)
    plt.hist(line_counts, bins=bins, align='left', color='tab:green', edgecolor='black')
    plt.xlabel('Lines Cleared')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lines Cleared')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    hist_path = os.path.join(plots_dir, f'clear_lines_hist_{ts}.png')
    plt.savefig(hist_path)
    print(f'已保存消行次数分布直方图到 {hist_path}')


if __name__ == '__main__':
    main()
