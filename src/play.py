"""
play.py: 人工操控 Tetris 游戏，使用 pygame GUI 接受键盘输入。
"""
import sys
# 从相对包导入
from tetris.game import TetrisGame


def main():
    # 初始化游戏环境
    env = TetrisGame()
    state = env.reset()

    # 初始化 Pygame 时钟和定时事件
    import pygame
    # 初始化 Pygame 模块并创建窗口
    pygame.init()
    # 调用环境的 GUI 初始化，确保 display 模块可用
    env._init_gui()
    clock = pygame.time.Clock()
    DROP_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(DROP_EVENT, 500)  # 每 500ms 自动下落

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == DROP_EVENT:
                # 自动自然下落
                _, _, done, _ = env.step(None)
                if done:
                    running = False
            elif event.type == pygame.KEYDOWN:
                # 键盘控制动作
                if event.key == pygame.K_LEFT:
                    env.step(0)
                elif event.key == pygame.K_RIGHT:
                    env.step(1)
                elif event.key == pygame.K_UP:
                    env.step(2)
                elif event.key == pygame.K_SPACE:
                    env.step(3)  # 硬下落
                elif event.key == pygame.K_ESCAPE:
                    running = False
        # 渲染 GUI
        env.render(mode='gui', delay=0)
        # 控制帧率
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
