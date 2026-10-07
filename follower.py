"""
Pure-pursuit waypoint follower: path polyline -> (vx, vy, yaw_rate) command for the trot policy.
Pure numpy, no MuJoCo.
"""
import numpy as np


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def dist_to_polyline(p, pts):
    best = 1e9
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        t = np.clip(np.dot(p - a, d) / np.dot(d, d), 0.0, 1.0)
        best = min(best, np.linalg.norm(p - (a + t * d)))
    return best


class WaypointFollower:
    def __init__(self, pts, v_max=0.5, yaw_max=0.8, lookahead=0.6,
                 k_yaw=1.5, corner_tol=0.35, goal_tol=0.3):
        self.pts = np.asarray(pts, dtype=float)
        self.v_max, self.yaw_max = v_max, yaw_max
        self.lookahead, self.k_yaw = lookahead, k_yaw
        self.corner_tol, self.goal_tol = corner_tol, goal_tol
        self.seg = 0
        self.done = False

    def command(self, x, y, yaw):
        p = np.array([x, y])
        goal = self.pts[-1]
        if np.linalg.norm(p - goal) < self.goal_tol:
            self.done = True
        if self.done:
            return 0.0, 0.0, 0.0

        # move on to the next segment when close to the end of the current one
        while self.seg < len(self.pts) - 2 and np.linalg.norm(p - self.pts[self.seg + 1]) < self.corner_tol:
            self.seg += 1

        a, b = self.pts[self.seg], self.pts[self.seg + 1]
        d = b - a
        L = np.linalg.norm(d)
        u = d / L
        s = np.clip(np.dot(p - a, u), 0.0, L)
        carrot = a + u * min(s + self.lookahead, L)

        err = wrap(np.arctan2(carrot[1] - p[1], carrot[0] - p[0]) - yaw)
        wz = float(np.clip(self.k_yaw * err, -self.yaw_max, self.yaw_max))

        # slow down when pointing the wrong way; slow down towards the goal
        vx = self.v_max * float(np.clip(1.0 - abs(err) / 1.2, 0.0, 1.0))
        if self.seg == len(self.pts) - 2:
            vx *= float(np.clip(np.linalg.norm(p - goal) / 0.8, 0.3, 1.0))
        return vx, 0.0, wz