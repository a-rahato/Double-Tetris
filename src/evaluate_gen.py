# filename: src/evaluate_gen.py
"""
evaluate_gen.py: 评估生成器 Agent 性能，对固定 DQN 玩家代理的得分进行评估。
"""
import argparse
import os
from tetris.game import TetrisGame
from tetris.pieces import PIECES
from agents.dqn_agent import DQNAgent
from agents.gen_agent import GenAgent

class GenTetrisGame(TetrisGame):
    """Tetris 环境，下一块由生成器 Agent 决定"""
    def __init__(self, generator, width=10, height=20):
        # set generator before super to allow _spawn_piece in reset
        self.generator = generator
        self.current_piece_idx = None
        super().__init__(width, height)

    def _spawn_piece(self):
        # 在方块生成时调用生成器
        board_copy = self.board.copy()
        idx = self.generator.act(board_copy, self)
        self.current_piece_idx = idx
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
    parser.add_argument('--episodes', type=int, default=100, help='评估回合数')
    parser.add_argument('--gen_weights', type=str, default='gen_weights.pkl', help='生成器权重文件')
    parser.add_argument('--dqn_weights', type=str, default='dqn_weights.pkl', help='玩家 DQN 权重文件')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    args = parser.parse_args()
    # 加载玩家代理
    dqn = DQNAgent(epsilon=0.0)
    if os.path.exists(args.dqn_weights):
        dqn.load_weights(args.dqn_weights)
        print(f"已加载玩家 DQN 权重: {args.dqn_weights}")
    # 加载生成器代理
    gen = GenAgent(epsilon=0.0)
    if os.path.exists(args.gen_weights):
        gen.load_weights(args.gen_weights)
        print(f"已加载生成器权重: {args.gen_weights}")
    # 评估
    total_scores = []
    for _ in range(args.episodes):
        env = GenTetrisGame(generator=gen)
        state = env.reset()
        done = False
        while not done:
            action = dqn.act(state, env)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render(mode='gui')
        total_scores.append(env.score)
    avg_score = sum(total_scores) / len(total_scores)
    print(f"固定 DQN 玩家代理平均得分: {avg_score:.2f}")
    # 生成器奖励为负玩家得分
    gen_rewards = [-s for s in total_scores]
    avg_gen = sum(gen_rewards) / len(gen_rewards)
    print(f"生成器平均奖励: {avg_gen:.2f}")
