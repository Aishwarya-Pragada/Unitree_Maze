"""
Full pipeline: A* planner -> waypoint follower -> trot policy -> maze goal.

  python run_maze.py --layout layout1 --run runs/trot_v1 --video maze1.mp4
  python run_maze.py --layout layout2 --run runs/trot_v1            # live viewer
  python run_maze.py --layout layout3 --run runs/trot_v1 --no-render
"""
import argparse
import os
import sys
import time
import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from layouts import LAYOUTS
from maze_scene import solve, prune_collinear, build_scene, HERE
from maze_env import MazeEnv
from follower import WaypointFollower, dist_to_polyline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="layout1", choices=list(LAYOUTS))
    p.add_argument("--run", default=os.path.join(HERE, "runs", "trot_v1"))
    p.add_argument("--model", default=None)
    p.add_argument("--stats", default=None)
    p.add_argument("--speed", type=float, default=0.5, help="max forward command (m/s)")
    p.add_argument("--max-time", type=float, default=90.0)
    p.add_argument("--video", default=None)
    p.add_argument("--cam", default="top", choices=["top", "track"])
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    # ---- plan
    maze = LAYOUTS[args.layout]
    rows, cols = maze.shape
    cells, pts = solve(maze)
    corners = prune_collinear(pts)
    xml = build_scene(maze, args.layout, path_pts=pts)
    d0 = corners[1] - corners[0]
    start_pose = (corners[0][0], corners[0][1], float(np.arctan2(d0[1], d0[0])))
    astar_len = float(np.sum(np.linalg.norm(np.diff(corners, axis=0), axis=1)))
    print(f"{args.layout}: {rows}x{cols}, A* path {len(cells)} cells, {len(corners)-2} corners, {astar_len:.1f} m")

    # ---- policy
    venv = DummyVecEnv([lambda: MazeEnv(xml, start_pose=start_pose)])
    venv = VecNormalize.load(args.stats or os.path.join(args.run, "vecnormalize.pkl"), venv)
    venv.training, venv.norm_reward = False, False
    policy = PPO.load(args.model or os.path.join(args.run, "model.zip"), device="cpu")
    base = venv.envs[0]
    dt = base.dt
    follower = WaypointFollower(corners, v_max=args.speed)

    # ---- render setup
    viewer = renderer = cam = None
    frames = []
    if args.video:
        renderer = mujoco.Renderer(base.model, 480, 640)
    elif not args.no_render:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(base.model, base.data)
    if renderer or viewer:
        cam = viewer.cam if viewer else mujoco.MjvCamera()
        if args.cam == "top":
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [(cols - 1) / 2, (rows - 1) / 2, 0.0]
            cam.distance, cam.azimuth, cam.elevation = 1.25 * max(rows, cols) + 3, 90, -80
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = base.trunk_id
            cam.distance, cam.azimuth, cam.elevation = 3.0, 135, -30

    # ---- run
    obs = venv.reset()
    trail, dev = [], []
    wall_steps = wall_events = 0
    prev_wall = False
    fell = False
    t_end_hold = 0
    for i in range(int(args.max_time / dt)):
        x, y, yaw = base.pose()
        vx, vy, wz = follower.command(x, y, yaw)
        base.set_command(vx, vy, wz)
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, done, infos = venv.step(action)
        info = infos[0]

        trail.append((x, y))
        dev.append(dist_to_polyline(np.array([x, y]), corners))
        wall_steps += int(info["wall_hit"])
        wall_events += int(info["wall_hit"] and not prev_wall)
        prev_wall = info["wall_hit"]

        if viewer:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(dt)
        if renderer:
            renderer.update_scene(base.data, camera=cam)
            frames.append(renderer.render())

        if info["fell"]:
            fell = True
            break
        if follower.done:                       # hold still ~1.5 s so the stop is visible
            t_end_hold += 1
            if t_end_hold > int(1.5 / dt):
                break

    if viewer:
        viewer.close()
    if renderer and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=int(round(1 / dt)))
        print("saved video:", args.video)

    # ---- metrics
    trail = np.array(trail)
    travelled = float(np.sum(np.linalg.norm(np.diff(trail, axis=0), axis=1))) if len(trail) > 1 else 0.0
    gx, gy = corners[-1]
    final_dist = float(np.hypot(trail[-1][0] - gx, trail[-1][1] - gy))
    print("\n===== maze result =====")
    print(f"success               : {follower.done and not fell}")
    print(f"fell                  : {fell}")
    print(f"time                  : {len(trail) * dt:.1f} s")
    print(f"distance to goal      : {final_dist:.2f} m")
    print(f"path length (A* / actual): {astar_len:.1f} m / {travelled:.1f} m")
    print(f"path deviation        : mean {np.mean(dev):.2f} m, max {np.max(dev):.2f} m")
    print(f"wall contacts         : {wall_events} events ({wall_steps} steps)")
    sys.stdout.flush()
    os._exit(0)      # avoids the MuJoCo viewer segfault on exit


if __name__ == "__main__":
    main()
    