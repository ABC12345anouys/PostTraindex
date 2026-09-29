"""M1 调参：固定笔位跑一遍脚本专家，输出各阶段头/腕相机关键帧与诊断信息。

用法: MUJOCO_GL=egl python3 scripts/tune_expert.py [--seed 0] [--cfg configs/pick_up_marker.yaml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from envs import BimanualDexConfig  # noqa: E402
from envs.bimanual_dex_env import BimanualDexEnv  # noqa: E402
from teleop.ik_solver import quat_to_R  # noqa: E402
from teleop.scripted_expert import ScriptedExpert, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/pick_up_marker.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x", type=float, default=0.0)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--out", default="verify_out/expert_tune")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / args.cfg)

    env_cfg = BimanualDexConfig(
        max_episode_seconds=12,
        pen_xy=((args.x, args.x), (args.y, args.y)))
    env = BimanualDexEnv(config=env_cfg, render_mode="rgb_array")
    unwrapped = env
    orig_randomize = unwrapped._randomize_pen

    def fixed_pen(d):
        orig_randomize(d)
        yaw = args.yaw
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        qflat = np.array([0.7071068, 0.0, 0.7071068, 0.0])
        qyaw = np.array([c, 0.0, 0.0, s])
        d.qpos[3:7] = unwrapped._quat_mul(qyaw, qflat)

    unwrapped._randomize_pen = fixed_pen

    obs, info = env.reset(seed=args.seed)
    expert = ScriptedExpert(unwrapped, cfg)
    expert.reset(obs)
    print(f"selected hand={expert.side} mode={expert.mode}")
    for nm, ep, eo, _, _ in expert.plan_errors:
        print(f"plan {nm:8s} pos={ep*100:5.1f}cm ori={np.degrees(eo):5.1f}deg")

    shots = {}
    frame = 0
    while True:
        action = expert.act(obs)
        obs, r, term, trunc, info = env.step_rad(action)
        frame += 1
        st = cfg["stages"][expert.si]
        st_name = st["name"]
        if st_name not in shots or expert.k == expert.n - 1:
            _, ep_plan, eo_plan, tpos, tR = expert.plan_errors[expert.si]
            ep, eo = expert.ik.errors(tpos, tR)
            pp = obs["object"][:3]
            print(f"[{frame:3d}] {st_name:8s} IK pos={ep*100:5.1f}cm "
                  f"ori={np.degrees(eo):5.1f}deg pen=({pp[0]:+.3f},{pp[1]:+.3f},{pp[2]:.3f}) "
                  f"touch={int(info['grasped'])} lifted={int(info['lifted'])} hold={info['hold']}")
            imgs = obs["images"]
            montage = np.concatenate(
                [imgs["head"], imgs["left_wrist"], imgs["right_wrist"]], axis=1)
            tag = "end" if expert.k == expert.n - 1 else "mid"
            Image.fromarray(montage).save(out / f"f{frame:03d}_{st_name}_{tag}.png")
            if st_name not in shots:
                shots[st_name] = montage
        if term or trunc:
            print(f"\nDONE success={info['success']} term={term} trunc={trunc} "
                  f"pen_z={info['pen_z']:.3f} frames={frame}")
            break
    env.close()

    rows = list(shots.values())
    Image.fromarray(np.concatenate(rows, axis=0)).save(out / "montage.png")
    print(f"saved {out / 'montage.png'}")


if __name__ == "__main__":
    main()
