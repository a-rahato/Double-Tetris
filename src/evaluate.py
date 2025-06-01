"""
evaluate.py: 验证脚本，对比不同 Agent 表现并渲染示例游戏。
"""
import argparse
import os  # 用于加载权重
from tetris.game import DoubleTetrisGame
from agents.ddqn_agent import DDQNAgent  # DDQN 验证


def evaluate(agent, episodes=10, render=False):
    scores = []
    line_counts = []
    # 若渲染，初始化环境与窗口一次
    if render:
        env = DoubleTetrisGame()
        env.reset()
        env._init_gui()
    # 循环评估回合
    for _ in range(episodes):
        # 非渲染模式下，每轮创建新环境；渲染模式复用同一窗口
        if not render:
            env = DoubleTetrisGame()
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
        # 记录本轮消除行数（两区之和）
        lines_total = env.game1.lines_cleared + env.game2.lines_cleared
        line_counts.append(lines_total)
    return scores, line_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5, help='评估回合数')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    parser.add_argument('--hidden', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--epsilon', type=float, default=0.0, help='评估时探索率')
    args = parser.parse_args()
    # 初始化 DDQN 代理并加载权重
    agent = DDQNAgent(width=10, height=20,
                      lr=args.lr, gamma=args.gamma,
                      epsilon=args.epsilon, hidden_dim=args.hidden)
    weight_file = 'ddqn_weights.pkl'
    if os.path.exists(weight_file):
        agent.load_weights(weight_file)
        print(f"已加载 DDQN 权重：{weight_file}")
    # 评估
    scores, line_counts = evaluate(agent, episodes=args.episodes, render=args.render)
    # 输出平均分
    print(f"DDQN 平均分: {sum(scores)/len(scores):.2f}")
    # 输出消除行数最多和最少的5次
    sorted_lines = sorted(line_counts)
    min5 = sorted_lines[:5]
    max5 = sorted_lines[-5:][::-1]
    print(f"消除行数最少的5次: {min5}")
    print(f"消除行数最多的5次: {max5}")


if __name__ == '__main__':
    main()
