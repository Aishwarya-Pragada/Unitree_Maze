import numpy as np


# 0 = free space
# 1 = wall
# 2 = start
# 3 = goal

MAZE = np.array([
    [2, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3],
], dtype=np.int8)


def get_start_and_goal(maze):
    start = tuple(np.argwhere(maze == 2)[0])
    goal = tuple(np.argwhere(maze == 3)[0])

    return start, goal


def print_maze(maze, path=None):
    path = set(path or [])

    for r in range(maze.shape[0]):
        row = ""

        for c in range(maze.shape[1]):
            position = (r, c)

            if position in path and maze[r, c] not in (2, 3):
                row += "*"
            elif maze[r, c] == 0:
                row += "."
            elif maze[r, c] == 1:
                row += "#"
            elif maze[r, c] == 2:
                row += "S"
            elif maze[r, c] == 3:
                row += "G"

        print(row)


if __name__ == "__main__":
    start, goal = get_start_and_goal(MAZE)

    print("Maze:")
    print_maze(MAZE)

    print()
    print("Start:", start)
    print("Goal :", goal)