import heapq


def heuristic(a, b):
    """Manhattan distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(maze, node):
    """Return valid 4-connected neighboring cells."""

    rows, cols = maze.shape
    r, c = node

    directions = [
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
    ]

    neighbors = []

    for dr, dc in directions:
        nr = r + dr
        nc = c + dc

        if 0 <= nr < rows and 0 <= nc < cols:
            if maze[nr, nc] != 1:
                neighbors.append((nr, nc))

    return neighbors


def astar(maze, start, goal):
    """
    Find the shortest path from start to goal using A*.

    Returns:
        list of (row, column) positions
    """

    open_set = []

    heapq.heappush(
        open_set,
        (heuristic(start, goal), 0, start)
    )

    came_from = {}

    g_cost = {
        start: 0
    }

    while open_set:

        _, current_cost, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(maze, current):

            new_cost = current_cost + 1

            if neighbor not in g_cost or new_cost < g_cost[neighbor]:

                g_cost[neighbor] = new_cost
                came_from[neighbor] = current

                f_cost = new_cost + heuristic(
                    neighbor,
                    goal
                )

                heapq.heappush(
                    open_set,
                    (f_cost, new_cost, neighbor)
                )

    return None


def reconstruct_path(came_from, current):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()

    return path