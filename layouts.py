"""Maze layouts. 0 = free, 1 = wall, 2 = start, 3 = goal. Each cell is 1 m x 1 m."""
import numpy as np
from maze_generator import MAZE as _TEAMMATE_MAZE   # teammate's original maze

LAYOUTS = {
    "layout1": _TEAMMATE_MAZE,
    "layout2": np.array([
        [2, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 3],
    ], dtype=np.int8),
    "layout3": np.array([
        [2, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 3],
    ], dtype=np.int8),
}