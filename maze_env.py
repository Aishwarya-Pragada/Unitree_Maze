"""
MazeEnv: A1Env placed inside a maze scene.
Differences from the training env (policy and observations are unchanged):
  - robot starts at a given (x, y, yaw) instead of the origin / random heading
  - touching a wall is counted (info['wall_hit']) and does NOT terminate the episode
  - episode length is long enough to cross the maze
"""
import numpy as np
import mujoco

from rl_env import A1Env


class MazeEnv(A1Env):
    def __init__(self, xml_path, start_pose=None, **kw):
        self.start_pose = start_pose                      # (x, y, yaw)
        self._settle_xy = start_pose[:2] if start_pose is not None else (0.0, 0.0)
        kw.setdefault("episode_len_s", 120.0)
        super().__init__(xml_path=xml_path, **kw)
        self.wall_geoms = {
            g for g in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("wall_")
        }
        if not self.wall_geoms:
            raise ValueError("No geoms named 'wall_*' found in the scene")
        self.wall_hit = False

    def _find_default_pose(self):
        super()._find_default_pose()
        self._start_qpos[0:2] = self._settle_xy           # settle on free floor, not inside a wall

    def _contacts(self):
        """Like A1Env._contacts but walls are separated from the ground."""
        m, d = self.model, self.data
        foot = np.zeros(4, dtype=bool)
        trunk = False
        wall = False
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.wall_geoms or g2 in self.wall_geoms:
                wall = True
                continue
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if b1 == 0:
                og, ob = g2, b2
            elif b2 == 0:
                og, ob = g1, b1
            else:
                continue
            if ob == self.trunk_id:
                trunk = True
            elif og in self.foot_of_geom:
                mujoco.mj_contactForce(m, d, i, f6)
                if f6[0] > 1.0:
                    foot[self.foot_of_geom[og]] = True
        self.wall_hit = wall
        return foot, trunk

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.start_pose is not None:
            x, y, yaw = self.start_pose
            self.data.qpos[0], self.data.qpos[1] = x, y
            self.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            mujoco.mj_forward(self.model, self.data)
            obs = self._obs()
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        info["wall_hit"] = self.wall_hit
        return obs, r, term, trunc, info

    def pose(self):
        """(x, y, yaw) of the trunk in the world frame."""
        R = self.data.xmat[self.trunk_id].reshape(3, 3)
        return self.data.qpos[0], self.data.qpos[1], float(np.arctan2(R[1, 0], R[0, 0]))