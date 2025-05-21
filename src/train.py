"""
train.py: 训练脚本，可选随机验证或 Q-learning
"""
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import trange

from tetris.game import TetrisGame
from agents.random_agent import RandomAgent
from agents.q_learning_agent import QLearningAgent


def train(agent, episodes=1000):
    scores = []
    for _ in trange(episodes):
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
            total_reward += reward
        scores.append(total_reward)
        # 探索率衰减（针对 QLearningAgent）
        if hasattr(agent, 'decay') and hasattr(agent, 'min_epsilon'):
            agent.epsilon = max(agent.epsilon * agent.decay, agent.min_epsilon)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['random','qlearn'], default='random')
    parser.add_argument('--episodes', type=int, default=500)
    args = parser.parse_args()

    if args.agent == 'random':
        agent = RandomAgent()
    else:
        agent = QLearningAgent()
        # 若已有保存的权重文件，则加载并降低 epsilon
        weight_file = f"{args.agent}_weights.pkl"
        if os.path.exists(weight_file):
            agent.load_weights(weight_file)
            # 将 epsilon 降低至 min_epsilon，减少随机性
            agent.epsilon = agent.min_epsilon
            print(f"已加载权重并将初始探索率降低至 epsilon={agent.epsilon}")

    scores = train(agent, episodes=args.episodes)
    # 绘制分数曲线
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f"Agent: {args.agent}")
    plt.savefig(f"train_{args.agent}.png")
    # 保存 Q-learning 模型权重
    if hasattr(agent, 'save_weights'):
        weight_file = f"{args.agent}_weights.pkl"
        agent.save_weights(weight_file)
        print(f"已保存权重到 {weight_file}")

if __name__ == '__main__':
    main()
