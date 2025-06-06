"""
utils.py: 辅助函数，包括消行函数和特征提取，用于近似 Q-learning。
"""
import numpy as np
from typing import Tuple

def clear_lines(board: np.ndarray) -> tuple[np.ndarray, int]:
    """
    清除已满的行，向上移动，返回 (new_board, lines_cleared)，不修改输入。
    """
    full_rows = [i for i, row in enumerate(board) if all(row)]
    n = len(full_rows)
    if n > 0:
        mask = [i for i in range(board.shape[0]) if i not in full_rows]
        new_board = np.zeros_like(board)
        if mask:
            new_board[-len(mask):] = board[mask]
    else:
        new_board = board.copy()
    return new_board, n


def extract_features(board: np.ndarray) -> dict:
    """
    特征：本次消行数、孔洞总数、颠簸度、总高度（不修改原棋盘）
    """
    # 复制棋盘，用于行清除和后续统计
    board_cp = board.copy()
    # 1. 清除满行，得到新棋盘与消行数
    board_cp, lines_cleared = clear_lines(board_cp)
    h, w = board.shape
    # 2. 计算各列高度（第一个填充格子之上算高度）
    heights = []
    for j in range(w):
        col = board_cp[:, j]
        filled = np.where(col)[0]
        height = h - filled[0] if filled.size > 0 else 0
        heights.append(height)
    total_height = float(sum(heights))
    # 3. 孔洞数：每列第一个填充格子之后的空格
    holes = 0
    for j in range(w):
        col = board_cp[:, j]
        filled = np.where(col)[0]
        if filled.size == 0:
            continue
        idx = filled[0]
        holes += int(np.sum(col[idx+1:] == 0))
    holes = float(holes)
    # 4. 颠簸度：相邻列高度差绝对值之和
    bumpiness = float(sum(abs(heights[i] - heights[i+1]) for i in range(w-1)))
    return {
        'complete_lines': float(lines_cleared),
        'holes': holes,
        'bumpiness': bumpiness,
        'aggregate_height': total_height
    }
