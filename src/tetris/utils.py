"""
utils.py: 辅助函数，包括消行函数和特征提取，用于近似 Q-learning。
"""
import numpy as np

def clear_lines(board: np.ndarray) -> int:
    """
    清除已满的行，向上移动，返回清除的行数。
    board 会被原地修改。
    """
    full_rows = [i for i, row in enumerate(board) if all(row)]
    n = len(full_rows)
    if n > 0:
        # 保留未满的行索引
        mask = [i for i in range(board.shape[0]) if i not in full_rows]
        # 构建新棋盘，在底部放置保留下来的行
        new_board = np.zeros_like(board)
        if mask:
            new_board[-len(mask):] = board[mask]
        # 原地更新
        board[:] = new_board
    return n


def extract_features(board: np.ndarray, action=None) -> dict:
    """
    从状态和动作中提取特征，包含宏观动作 one-hot。
    action 可为 None、int 或 (rot,x) 宏观动作。
    返回字典特征。
    """
    # 获取尺寸
    h, w = board.shape
    # 列高度
    heights = []
    for col in range(w):
        column = board[:, col]
        # 从上到下找到第一个方块
        idx = np.where(column)[0]
        height = h - idx[0] if idx.size > 0 else 0
        heights.append(height)
    aggregate_height = sum(heights)
    # 孔洞数
    holes = 0
    for col in range(board.shape[1]):
        column = board[:, col]
        found_block = False
        for cell in column:
            if cell:
                found_block = True
            elif found_block and not cell:
                holes += 1
    # 凹凸不平 (bumpiness)
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
    # 完整消行数
    complete_lines = sum(1 for row in board if all(row))

    # 高级特征
    max_height = max(heights)
    mean_height = aggregate_height / w
    var_height = np.var(heights)
    # 行转换数 (row transitions)
    row_transitions = 0
    for r in range(h):
        for c in range(w-1):
            if (board[r, c] > 0) != (board[r, c+1] > 0):
                row_transitions += 1
    # 列转换数 (column transitions)
    col_transitions = 0
    for c in range(w):
        for r in range(h-1):
            if (board[r, c] > 0) != (board[r+1, c] > 0):
                col_transitions += 1
    # 井深度 (wells)
    well_sums = 0
    for c in range(w):
        depth = 0
        for r in range(h):
            left = True if c == 0 else board[r, c-1] > 0
            right = True if c == w-1 else board[r, c+1] > 0
            if board[r, c] == 0 and left and right:
                depth += 1
                well_sums += depth
            else:
                depth = 0
    # 原有棋面特征
    feats = {
        'aggregate_height': aggregate_height / (h * w),
        'max_height': max_height / h,
        'mean_height': mean_height / h,
        'var_height': var_height / (h*h),
        'holes': holes / (h * w),
        'bumpiness': bumpiness / (h * w),
        'row_transitions': row_transitions / (h * w),
        'col_transitions': col_transitions / (h * w),
        'well_sums': well_sums / (h * h),
        'complete_lines': complete_lines / h,
    }
    # 动作特征
    if action is not None:
        # 旋转索引 one-hot
        if isinstance(action, tuple):
            rot, x = action
        else:
            rot, x = action, None
        # 假设最多 4 种旋转
        for r in range(4):
            feats[f'rot_{r}'] = 1 if r == rot else 0
        # 水平位置 one-hot
        if x is not None:
            for col in range(w):
                feats[f'pos_{col}'] = 1 if col == x else 0
    return feats
