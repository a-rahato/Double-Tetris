"""
game.py: 定义俄罗斯方块核心游戏逻辑。
"""

import numpy as np
import random
from .pieces import PIECES
# 增加颜色映射
PIECE_COLORS = {
    'I': (0, 255, 255),  # 青
    'O': (255, 255, 0),  # 黄
    'T': (128, 0, 128),  # 紫
    'L': (255, 165, 0),  # 橙
    'J': (0, 0, 255),    # 蓝
    'S': (0, 255, 0),    # 绿
    'Z': (255, 0, 0),    # 红
}
PIECE_IDS = {p: i+1 for i, p in enumerate(PIECES.keys())}
ID_TO_COLOR = {PIECE_IDS[p]: PIECE_COLORS[p] for p in PIECE_IDS}
from .utils import clear_lines

# 可选导入 pygame，用于 GUI 渲染
try:
    import pygame
except ImportError:
    pygame = None

BLOCK_SIZE = 30  # 方块像素大小
CLEAR_REWARD = 1  # 每行清除奖励
GAME_OVER_PENALTY = 2  # 游戏结束小惩罚
SURVIVAL_REWARD = 0  # 无生存奖励

class TetrisGame:
    """代表一个俄罗斯方块游戏环境。"""

    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.score = 0
        self.done = False
        self.current_piece = None
        self.current_rot = 0
        self.current_x = 0
        self.current_y = 0
        self.lines_cleared = 0
        self.screen = None
        # 初始化后立即重置
        self.reset()

    def reset(self):
        """重置游戏状态并返回初始观测。"""
        self.board.fill(0)
        self.score = 0
        self.done = False
        self._spawn_piece()
        return self._get_observation()

    def step(self, action):
        """执行一步动作：0-left,1-right,2-rotate,3-drop，或宏观动作 (rot, x) 完整放置，返回 (next_state, reward, done, info)。"""
        if self.done:
            return self._get_observation(), -GAME_OVER_PENALTY, True, {}

        # 自然下落：action None 表示软下落一步
        if action is None:
            self.current_y += 1
            if self._collision():
                self.current_y -= 1
                self._lock_piece()
                self.board, lines = clear_lines(self.board)
                self.lines_cleared += lines
                reward = 1 + (lines ** 2) * self.width
                if self.done:
                    reward -= GAME_OVER_PENALTY
                self.score += reward
                next_obs = self.board.copy()
                self._spawn_piece()
                return next_obs, reward, self.done, {}
            return self._get_observation(), SURVIVAL_REWARD, self.done, {}

        # 简单动作：左右移动、旋转、一键硬下落
        if isinstance(action, int):
            if action == 0:
                self._move(-1)
            elif action == 1:
                self._move(1)
            elif action == 2:
                self._rotate()
            elif action == 3:
                self._hard_drop()
                self._lock_piece()
                self.board, lines = clear_lines(self.board)
                self.lines_cleared += lines
                reward = 1 + (lines ** 2) * self.width
                if self.done:
                    reward -= GAME_OVER_PENALTY
                self.score += reward
                next_obs = self.board.copy()
                self._spawn_piece()
                return next_obs, reward, self.done, {}
            return self._get_observation(), SURVIVAL_REWARD, self.done, {}

        # 宏观动作：指定旋转和落点
        rot, x = action
        self.current_rot = rot % len(PIECES[self.current_piece])
        self.current_x = x
        self._hard_drop()
        self._lock_piece()
        self.board, lines = clear_lines(self.board)
        self.lines_cleared += lines
        reward = 1 + (lines ** 2) * self.width
        if self.done:
            reward -= GAME_OVER_PENALTY
        self.score += reward
        next_obs = self.board.copy()
        self._spawn_piece()
        return next_obs, reward, self.done, {}

    def render(self, mode='text', delay=20):
        """渲染当前游戏状态。mode 支持 'text' 或 'gui'. delay 为 GUI 延迟(ms)。"""
        if mode == 'text':
            obs = self._get_observation()
            for row in obs:
                print(''.join(['█' if x else '·' for x in row]))
            print(f"Score: {self.score}")
        elif mode == 'gui':
            if pygame is None:
                raise RuntimeError('pygame 未安装，无法使用 GUI 模式')
            self._draw_gui()
            pygame.time.delay(delay)

    def _init_gui(self):
        """初始化 Pygame 窗口"""
        if pygame is None:
            raise RuntimeError('pygame 未安装，无法使用 GUI 模式')
        # 初始化 pygame
        if not pygame.get_init():
            pygame.init()
        # 尝试复用已有窗口
        w, h = self.width * BLOCK_SIZE, self.height * BLOCK_SIZE
        surface = pygame.display.get_surface()
        if surface:
            self.screen = surface
        else:
            self.screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption('Tetris')
        # 初始化字体
        if not pygame.font.get_init():
            pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)

    def _draw_gui(self):
        """使用 Pygame 绘制当前棋盘"""
        if pygame is None:
            raise RuntimeError('pygame 未安装，无法使用 GUI 模式')
        if self.screen is None:
            self._init_gui()
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        # 背景填充
        self.screen.fill((0, 0, 0))
        # 绘制已锁定方块
        for y in range(self.height):
            for x in range(self.width):
                code = self.board[y, x]
                if code:
                    color = ID_TO_COLOR.get(code, (0, 255, 255))
                else:
                    color = (50, 50, 50)
                rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (20, 20, 20), rect, 1)
        # 绘制当前移动方块
        shape = PIECES[self.current_piece][self.current_rot]
        cp_color = PIECE_COLORS[self.current_piece]
        for i in range(shape.shape[0]):
            for j in range(shape.shape[1]):
                if shape[i, j]:
                    y = self.current_y + i
                    x = self.current_x + j
                    if 0 <= y < self.height and 0 <= x < self.width:
                        rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                        pygame.draw.rect(self.screen, cp_color, rect)
                        pygame.draw.rect(self.screen, (20, 20, 20), rect, 1)
        # 渲染消行数文本
        text_surf = self.font.render(f"Lines: {self.lines_cleared}", True, (255, 255, 255))
        self.screen.blit(text_surf, (5, 5))
        text_surf = self.font.render(f"Scores: {self.score}", True, (255, 255, 255))
        self.screen.blit(text_surf, (200, 5))
        pygame.display.flip()

    # 内部方法
    def _get_observation(self):
        grid = self.board.copy()
        shape = PIECES[self.current_piece][self.current_rot]
        h, w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i, j]:
                    y = self.current_y + i
                    x = self.current_x + j
                    if 0 <= y < self.height and 0 <= x < self.width:
                        grid[y, x] = shape[i, j]
        return grid

    def _spawn_piece(self):
        self.current_piece = random.choice(list(PIECES.keys()))
        self.current_rot = 0
        shape = PIECES[self.current_piece][self.current_rot]
        self.current_y = 0
        self.current_x = (self.width - shape.shape[1]) // 2
        if self._collision():
            self.done = True

    def _move(self, dx):
        old_x = self.current_x
        self.current_x += dx
        if self._collision():
            self.current_x = old_x

    def _rotate(self):
        old_rot = self.current_rot
        self.current_rot = (self.current_rot + 1) % len(PIECES[self.current_piece])
        if self._collision():
            self.current_rot = old_rot

    def _hard_drop(self):
        while True:
            self.current_y += 1
            if self._collision():
                self.current_y -= 1
                break

    def _collision(self):
        shape = PIECES[self.current_piece][self.current_rot]
        h, w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i, j]:
                    y = self.current_y + i
                    x = self.current_x + j
                    if x < 0 or x >= self.width or y >= self.height or (y >= 0 and self.board[y, x]):
                        return True
        return False

    def _lock_piece(self):
        """将当前方块写入棋盘，并检测重叠或越界以结束游戏"""
        shape = PIECES[self.current_piece][self.current_rot]
        h, w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i, j]:
                    y = self.current_y + i
                    x = self.current_x + j
                    if 0 <= y < self.height and 0 <= x < self.width:
                        # 检测重叠：已有方块时结束游戏
                        if self.board[y, x] != 0:
                            self.done = True
                        # 写入方块ID
                        self.board[y, x] = PIECE_IDS[self.current_piece]
