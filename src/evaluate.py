"""
evaluate.py: 验证脚本，对比不同 Agent 表现并渲染示例游戏。
"""
import argparse
from tetris.game import TetrisGame
from agents.random_agent import RandomAgent
from agents.q_learning_agent import QLearningAgent


def evaluate(agent, episodes=10, render=False):
    scores = []
    for _ in range(episodes):
        env = TetrisGame()
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            # 使用环境和状态调用宏观动作接口
            action = agent.act(state, env)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done, env)
            total_reward += reward
            if render:
                # 使用 GUI 模式渲染
                env.render(mode='gui')
        scores.append(total_reward)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['random','qlearn'], default='random')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    agent = RandomAgent() if args.agent=='random' else QLearningAgent()
    if args.agent == 'qlearn':
        agent.load_weights('qlearn_weights.pkl')
        scores = evaluate(agent, episodes=10, render=True)
    else:
        scores = evaluate(agent, episodes=args.episodes, render=args.render)
    print(f"Agent={args.agent} 平均分: {sum(scores)/len(scores)}")

if __name__ == '__main__':
    main()
