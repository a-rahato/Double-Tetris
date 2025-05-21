"""
defines all Tetris piece shapes and their rotations as numpy arrays.
"""
import numpy as np

# 每个 piece 的所有旋转形状 (1 表示方块)
PIECES = {
    'I': [
        np.array([[1, 1, 1, 1]]),
        np.array([[1], [1], [1], [1]]),
    ],
    'O': [
        np.array([[1, 1], [1, 1]]),
    ],
    'T': [
        np.array([[0, 1, 0], [1, 1, 1]]),
        np.array([[1, 0], [1, 1], [1, 0]]),
        np.array([[1, 1, 1], [0, 1, 0]]),
        np.array([[0, 1], [1, 1], [0, 1]]),
    ],
    'L': [
        np.array([[1, 0], [1, 0], [1, 1]]),
        np.array([[0, 0, 1], [1, 1, 1]]),
        np.array([[1, 1], [0, 1], [0, 1]]),
        np.array([[1, 1, 1], [1, 0, 0]]),
    ],
    'J': [
        np.array([[0, 1], [0, 1], [1, 1]]),
        np.array([[1, 1, 1], [0, 0, 1]]),
        np.array([[1, 1], [1, 0], [1, 0]]),
        np.array([[1, 0, 0], [1, 1, 1]]),
    ],
    'S': [
        np.array([[0, 1, 1], [1, 1, 0]]),
        np.array([[1, 0], [1, 1], [0, 1]]),
    ],
    'Z': [
        np.array([[1, 1, 0], [0, 1, 1]]),
        np.array([[0, 1], [1, 1], [1, 0]]),
    ],
}
