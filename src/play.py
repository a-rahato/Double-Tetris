"""
play.py: 人工操控 Tetris 游戏，使用 pygame GUI 接受键盘输入。
"""
import sys
# 从相对包导入
from tetris.game import DoubleTetrisGame
import pygame


def main():
    """通过 Pygame 键盘控制双区图形化俄罗斯方块"""
    env = DoubleTetrisGame()
    env.reset()
    # 初始化 Pygame 和环境 GUI
    pygame.init()
    env._init_gui()
    clock = pygame.time.Clock()
    DROP_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(DROP_EVENT, 500)  # 每500ms自动下落
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == DROP_EVENT:
                _, _, done, _ = env.step(None)
                if done:
                    # 游戏结束后保持窗口，自动重置游戏
                    env.reset()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    env.step(0)
                elif event.key == pygame.K_RIGHT:
                    env.step(1)
                elif event.key == pygame.K_UP:
                    env.step(2)
                elif event.key == pygame.K_SPACE:
                    env.step(3)
                elif event.key == pygame.K_ESCAPE:
                    running = False
        # 渲染 GUI
        env.render(mode='gui', delay=0)
        clock.tick(60)
    pygame.quit()
    # sys.exit()  # 不要自动退出解释器，窗口关闭后程序结束


if __name__ == '__main__':
    main()
