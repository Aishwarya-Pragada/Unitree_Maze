"""
Test a trained policy.

  python test.py --run runs/trot_v1                       # live viewer, demo command sequence
  python test.py --run runs/trot_v1 --vx 0.6              # fixed command in viewer
  python test.py --run runs/trot_v1 --video out.mp4       # record video (needs imageio + imageio-ffmpeg)
  python test.py --run runs/trot_v1 --no-render           # metrics only, fast
  (macOS: run the viewer with `mjpython test.py ...`)
"""
import argparse
import os
import time
import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env import A1Env, HERE

# (duration_s, vx, vy, yaw)
DEMO = [
    (2.0, 0.0, 0.0, 0.0),    # stand
    (4.0, 0.5, 0.0, 0.0),    # forward slow
    (4.0, 0.8, 0.0, 0.0),    # forward fast
    (3.0, 0.4, 0.0, 0.7),    # arc left
    (3.0, 0.4, 0.0, -0.7),   # arc right
    (3.0, 0.0, 0.3, 0.0),    # sidestep left
    (3.0, 0.0, -0.3, 0.0),   # sidestep right
    (3.0, -0.3, 0.0, 0.0),   # backward
    (2.0, 0.0, 0.0, 0.0),    # stop
]


def demo_cmd(t):
    for dur, vx, vy, yaw in DEMO:
        if t < dur:
            return vx, vy, yaw
        t -= dur
    return 0.0, 0.0, 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=os.path.join(HERE, "scene.xml"))
    p.add_argument("--run", default=os.path.join(HERE, "runs", "trot_v1"))
    p.add_argument("--model", default=None, help="model .zip (default: <run>/model.zip)")
    p.add_argument("--stats", default=None, help="vecnormalize .pkl (default: <run>/vecnormalize.pkl)")
    p.add_argument("--vx", type=float, default=None)
    p.add_argument("--vy", type=float, default=None)
    p.add_argument("--yaw", type=float, default=None)
    p.add_argument("--duration", type=float, default=None, help="seconds (default: demo length or 20)")
    p.add_argument("--video", default=None)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    model_path = args.model or os.path.join(args.run, "model.zip")
    stats_path = args.stats or os.path.join(args.run, "vecnormalize.pkl")

    venv = DummyVecEnv([lambda: A1Env(xml_path=args.xml)])
    venv = VecNormalize.load(stats_path, venv)
    venv.training = False
    venv.norm_reward = False
    policy = PPO.load(model_path, device="cpu")
    base = venv.envs[0]
    dt = base.dt

    fixed = any(v is not None for v in (args.vx, args.vy, args.yaw))
    fixed_cmd = (args.vx or 0.0, args.vy or 0.0, args.yaw or 0.0)
    total_t = args.duration or (20.0 if fixed else sum(d[0] for d in DEMO))
    n_steps = int(total_t / dt)

    # ---- render setup
    viewer = renderer = cam = None
    frames = []
    if args.video:
        renderer = mujoco.Renderer(base.model, 480, 640)
    elif not args.no_render:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(base.model, base.data)
    if args.video or viewer:
        cam = viewer.cam if viewer else mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = base.trunk_id
        cam.distance, cam.azimuth, cam.elevation = 2.2, 135, -20

    # ---- rollout
    obs = venv.reset()
    log = dict(cmd=[], vel=[], wz=[], feet=[], falls=0)
    for i in range(n_steps):
        t = i * dt
        base.set_command(*(fixed_cmd if fixed else demo_cmd(t)))
        action, _ = policy.predict(obs, deterministic=True)
        obs, rew, done, infos = venv.step(action)
        info = infos[0]
        log["cmd"].append(info["cmd"])
        log["vel"].append(info["v_local"][:2])
        log["wz"].append(info["w_local"][2])
        log["feet"].append(info["foot_contacts"])
        if info["fell"]:
            log["falls"] += 1

        if viewer:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(dt)
        if renderer:
            renderer.update_scene(base.data, camera=cam)
            frames.append(renderer.render())

    if viewer:
        viewer.close()
    if args.video and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=int(round(1 / dt)))
        print("saved video:", args.video)

    # ---- metrics
    cmd = np.array(log["cmd"])
    vel = np.array(log["vel"])
    wz = np.array(log["wz"])
    feet = np.array(log["feet"])
    moving = np.linalg.norm(cmd[:, :2], axis=1) > 0.05
    print("\n===== metrics =====")
    print(f"simulated time        : {len(cmd) * dt:.1f} s")
    print(f"falls                 : {log['falls']}")
    print(f"mean lin vel error    : {np.mean(np.linalg.norm(cmd[:, :2] - vel, axis=1)):.3f} m/s")
    print(f"mean yaw rate error   : {np.mean(np.abs(cmd[:, 2] - wz)):.3f} rad/s")
    print(f"contact fraction FR/FL/RR/RL: {np.round(feet.mean(axis=0), 2)}")
    if moving.any():
        f = feet[moving]
        diag = np.mean((f[:, 0] == f[:, 3]) & (f[:, 1] == f[:, 2]))
        anti = np.mean(f[:, 0] != f[:, 1])
        print(f"trot check (moving)   : diagonal pairs in sync {diag:.0%}, pairs in anti-phase {anti:.0%}")

    

if __name__ == "__main__":
    main()