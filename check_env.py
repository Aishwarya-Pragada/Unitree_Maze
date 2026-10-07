"""Run this first: prints model info and checks the robot can stand with zero action."""
import argparse
import os
import numpy as np
import mujoco
from rl_env import A1Env, HERE

p = argparse.ArgumentParser()
p.add_argument("--xml", default=os.path.join(HERE, "scene.xml"))
args = p.parse_args()

m = mujoco.MjModel.from_xml_path(args.xml)
print(f"timestep: {m.opt.timestep}  nq={m.nq} nv={m.nv} nu={m.nu}")
print("joints   :", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)])
print("actuators:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])
print("bodies   :", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)])
print("keyframes:", m.nkey)

env = A1Env(xml_path=args.xml)
print("\nactuator type      :", "position (PD in actuator)" if env.pos_act else "torque motor (PD in code)")
print("control dt         :", env.dt, " frame_skip:", env.frame_skip)
print("foot geom names    :", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in env.foot_geoms])
print("default joint pose :", np.round(env.default_q, 3))
print("settled base height:", round(env.target_height, 3))
print("obs dim            :", env.observation_space.shape)

obs, _ = env.reset(seed=0)
env.set_command(0.0, 0.0, 0.0)
total = 0.0
for i in range(150):  # 3 seconds of "do nothing" (stay in standing pose)
    obs, r, term, trunc, info = env.step(np.zeros(12))
    total += r
    if i % 30 == 0:
        print(f"step {i:3d}  z={env.data.qpos[2]:.3f}  feet={info['foot_contacts'].astype(int)}  reward={r:.2f}")
    if term:
        print("TERMINATED at step", i, "-> robot fell with zero action, check XML / pose")
        break
else:
    print("OK: robot stands with zero action. total reward over 3s:", round(total, 1))
