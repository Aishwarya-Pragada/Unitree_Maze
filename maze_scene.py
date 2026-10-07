"""
Grid maze -> A* path -> MuJoCo scene XML (robot + walls).

Uses the teammate's astar.py and maze_generator.py unchanged.
Grid convention (same as teammate's): world x = col * CELL, world y = (rows-1-row) * CELL.
"""
import os
import numpy as np

from astar import astar
from maze_generator import get_start_and_goal

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = 1.0
WALL_H = 0.5


def grid_to_world(row, col, rows):
    return col * CELL, (rows - 1 - row) * CELL


def solve(maze):
    """Returns (path_cells, waypoints_xy for every cell on the path)."""
    start, goal = get_start_and_goal(maze)
    path = astar(maze, start, goal)
    if path is None:
        raise RuntimeError("A* found no path")
    rows = maze.shape[0]
    pts = [grid_to_world(int(r), int(c), rows) for r, c in path]
    return path, pts


def prune_collinear(pts):
    """Keep only start, corners and goal (straight runs become one segment)."""
    pts = [np.array(p, dtype=float) for p in pts]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        d1 = pts[i] - pts[i - 1]
        d2 = pts[i + 1] - pts[i]
        if abs(d1[0] * d2[1] - d1[1] * d2[0]) > 1e-9:   # direction changed
            out.append(pts[i])
    out.append(pts[-1])
    return np.array(out)


def build_scene(maze, name, path_pts=None, out_dir=HERE, base_scene="scene.xml"):
    """Writes maze_<name>.xml that includes your scene.xml (robot + floor) and adds walls."""
    rows, cols = maze.shape
    start, goal = get_start_and_goal(maze)
    lines = []
    wid = 0
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1:
                x, y = grid_to_world(r, c, rows)
                lines.append(
                    f'    <geom name="wall_{wid}" type="box" size="{CELL/2} {CELL/2} {WALL_H/2}" '
                    f'pos="{x} {y} {WALL_H/2}" rgba="0.55 0.55 0.6 1"/>'
                )
                wid += 1
    sx, sy = grid_to_world(start[0], start[1], rows)
    gx, gy = grid_to_world(goal[0], goal[1], rows)
    lines.append(f'    <geom name="start_marker" type="cylinder" size="0.3 0.005" pos="{sx} {sy} 0.006" '
                 f'rgba="0 0.8 0 1" contype="0" conaffinity="0"/>')
    lines.append(f'    <geom name="goal_marker" type="cylinder" size="0.3 0.005" pos="{gx} {gy} 0.006" '
                 f'rgba="0.9 0.8 0 1" contype="0" conaffinity="0"/>')
    if path_pts is not None:
        for i, (x, y) in enumerate(path_pts):
            lines.append(f'    <geom name="path_{i}" type="cylinder" size="0.05 0.003" pos="{x} {y} 0.005" '
                         f'rgba="0.2 0.5 1 1" contype="0" conaffinity="0"/>')
    xml = (f'<mujoco model="a1 maze {name}">\n'
           f'  <include file="{base_scene}"/>\n'
           f'  <worldbody>\n' + "\n".join(lines) + '\n  </worldbody>\n</mujoco>\n')
    out = os.path.join(out_dir, f"maze_{name}.xml")
    with open(out, "w") as f:
        f.write(xml)
    return out