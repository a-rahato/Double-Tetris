"""
train.py: 训练脚本，可选随机验证或 Q-learning
"""
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import trange
from datetime import datetime  # 用于生成时间戳

from tetris.game import TetrisGame
from agents.random_agent import RandomAgent
from agents.q_learning_agent import QLearningAgent


def train(agent, episodes=1000):
    scores = []
    # 训练循环，每一轮使用 episode 计数
    for episode in trange(episodes):
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
        # 每 10000 轮保存一次权重到历史文件夹
        if hasattr(agent, 'save_weights') and (episode + 1) % 2500 == 0:
            history_dir = 'weights_history'
            os.makedirs(history_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = os.path.join(history_dir, f"{agent.__class__.__name__}_weights_{ts}_ep{episode+1}.pkl")
            agent.save_weights(fname)
            print(f"Episode {episode+1}: 保存权重到 {fname}")
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
            agent.epsilon /= 2
            print(f"已加载权重并将初始探索率降低至 epsilon={agent.epsilon}")

    scores = train(agent, episodes=args.episodes)
    # 绘制分数曲线
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f"Agent: {args.agent}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"train_{args.agent}_{timestamp}.png")
    # 保存 Q-learning 模型权重
    if hasattr(agent, 'save_weights'):
        weight_file = f"{args.agent}_weights.pkl"
        agent.save_weights(weight_file)
        print(f"已保存权重到 {weight_file}")

if __name__ == '__main__':
    main()
