"""
evaluate.py: 验证脚本，对比不同 Agent 表现并渲染示例游戏。
"""
import argparse
import os  # 用于加载权重
from tetris.game import TetrisGame, DoubleTetrisGame
from agents.dqn_agent import DQNAgent  # 仅保留 DQN 验证


def evaluate(agent, episodes=10, render=False):
    scores1, scores2 = [], []
    lines1, lines2 = [], []
    for _ in range(episodes):
        env = DoubleTetrisGame()
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, env)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if render:
                env.render(mode='gui')
        # 分别记录两侧分数和消行
        s1, s2 = env.game1.score, env.game2.score
        scores1.append(s1)
        scores2.append(s2)
        lines1.append(env.game1.lines_cleared)
        lines2.append(env.game2.lines_cleared)
    return scores1, scores2, lines1, lines2


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
    weight_file = 'double_dqn_weights.pkl'
    if os.path.exists(weight_file):
        agent.load_weights(weight_file)
        print(f"已加载 DQN 权重：{weight_file}")
    # 对 Double Tetris 两棋盘进行评估
    scores1, scores2, lines1, lines2 = evaluate(agent, episodes=args.episodes, render=args.render)
    print(f"DQN 双Tetris 评估 ({args.episodes} 回合):")
    print(f"  Game1 平均分: {sum(scores1)/len(scores1):.2f}, 消行最少: {min(lines1)}, 消行最多: {max(lines1)}")
    print(f"  Game2 平均分: {sum(scores2)/len(scores2):.2f}, 消行最少: {min(lines2)}, 消行最多: {max(lines2)}")


if __name__ == '__main__':
    main()
