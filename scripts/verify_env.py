"""M0 验收脚本：BimanualDexEnv 端到端冒烟测试。

检查项：
  1. gym 注册与空间形状；
  2. reset 笔随机化（多次合法）+ 三路相机出图（拼图存 verify_out/env_reset.png）；
  3. 动作限幅：任意大动作单帧目标变化 ≤ π/18；
  4. home 保持 5 s：无终局、笔留在桌面；
  5. 手指闭合：20 个手驱动器到最大指令，手指确实运动，出图 env_grasp.png；
  6. 20 Hz 一步（含三路渲染）耗时。

运行（项目根目录）：MUJOCO_GL=egl python scripts/verify_env.py
"""
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import envs  # noqa: E402  (register BimanualDex-v0)
from envs import BimanualDexConfig, BimanualDexEnv  # noqa: E402

OUT = os.path.join(ROOT, "verify_out")
os.makedirs(OUT, exist_ok=True)
RATE = np.pi / 18

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def montage(obs, tag):
    imgs = obs["images"]
    canvas = np.concatenate([imgs["head"], imgs["left_wrist"], imgs["right_wrist"]], axis=1)
    path = os.path.join(OUT, f"env_{tag}.png")
    Image.fromarray(canvas).save(path)
    return path


def main():
    # 1) 注册与空间
    env = gym.make("BimanualDex-v0", render_mode="rgb_array").unwrapped
    check("register/spaces",
          env.action_space.shape == (54,) and
          env.observation_space["state"].shape == (54,) and
          env.observation_space["object"].shape == (13,) and
          len(env.observation_space["images"].spaces) == 3,
          f"frame_skip={env.frame_skip}, dt={env.model.opt.timestep}")

    # 2) reset 随机化
    poses = []
    obs, info = env.reset(seed=0)
    for _ in range(10):
        obs, _ = env.reset()
        poses.append(obs["object"][:3].copy())
    poses = np.array(poses)
    check("reset shapes/dtypes",
          obs["state"].shape == (54,) and obs["images"]["head"].shape == (224, 224, 3)
          and obs["images"]["head"].dtype == np.uint8)
    check("reset randomization legal",
          np.all(np.abs(poses[:, 2]) - 0.508 < 0.03) and poses[:, :2].std() > 0.02,
          f"xy std={poses[:, :2].std(axis=0).round(3)}, z range={poses[:,2].min():.3f}..{poses[:,2].max():.3f}")
    p = montage(obs, "reset")
    print("     saved", p)

    # 3) 限幅
    obs, _ = env.reset(seed=1)
    before = env._target.copy()
    env.step(np.ones(54, dtype=np.float32))
    delta = np.abs(env._target - before).max()
    check("action rate limit", delta <= RATE + 1e-6, f"max|Δtarget|={delta:.4f} (limit {RATE:.4f})")

    # 4) home 保持 5s
    obs, _ = env.reset(seed=2)
    home = env.home_action_rad
    pen_z0 = obs["object"][2]
    ended = False
    for _ in range(100):
        obs, r, term, trunc, info = env.step_rad(home)
        if term or trunc:
            ended = True
            break
    check("hold home 5s", not ended and not info["grasped"] is None
          and abs(obs["object"][2] - pen_z0) < 0.05,
          f"pen_z {pen_z0:.3f}->{obs['object'][2]:.3f}")
    p = montage(obs, "hold")
    print("     saved", p)

    # 5) 手指闭合：手驱动器到上限（手臂弧度制锁定 home），保持 2.5 s
    obs, _ = env.reset(seed=3)
    close = home.copy()
    close[7:27] = env.ctrl_hi[7:27]
    close[34:54] = env.ctrl_hi[34:54]
    hand_before = obs["state"][7:27].copy()
    arm_before = obs["state"][0:7].copy()
    for _ in range(50):
        obs, r, term, trunc, info = env.step_rad(close)
    hand_after = obs["state"][7:27].copy()
    arm_after = obs["state"][0:7].copy()
    moved = np.abs(hand_after - hand_before).max()
    arm_drift = np.abs(arm_after - arm_before).max()
    check("hand close actuation", moved > 0.2 and arm_drift < 0.15 and not (term or trunc),
          f"hand motion={moved:.3f} rad, arm drift={arm_drift:.3f}")
    p = montage(obs, "grasp")
    print("     saved", p)

    # 6) 耗时
    env.reset(seed=4)
    n = 50
    t0 = time.time()
    for _ in range(n):
        env.step(np.zeros(54, dtype=np.float32))
    dt_step = (time.time() - t0) / n
    check("step timing", dt_step < 0.05, f"{dt_step*1000:.1f} ms/step (20Hz budget 50 ms)")

    env.close()
    failed = [n for n, ok, _ in results if not ok]
    print("\n== %d/%d checks passed ==" % (len(results) - len(failed), len(results)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
