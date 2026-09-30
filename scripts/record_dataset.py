"""g5 录数：LeRobot v3 数据集（白名单好点采样，成功条采集，8:2 train/val）。

跑法（必须 lerobot-env，物理 mujoco 3.7.0 与 M2/M3 评测一致）:
  MUJOCO_GL=egl <lerobot-env>/bin/python scripts/record_dataset.py \
      --cfg configs/record_pick_up_marker.yaml --total 12   # 分批前台跑，凑满 cfg.total

机制：每条随机抽白名单好点 → env 固定笔位 reset → 专家换 plan_seed 重试≤tries 次
→ 成功且 max_z≤max_z_sane（过滤甩飞脏成功）才落盘；断点续录（按 meta 已有条数）。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import BimanualDexConfig  # noqa: E402
from envs.bimanual_dex_env import BimanualDexEnv  # noqa: E402
from teleop.scripted_expert import ScriptedExpert, load_config  # noqa: E402


def build_features(size, act_dim=54, obj_dim=13):
    vid = {"dtype": "video", "shape": (size, size, 3), "names": None}
    feats = {f"observation.images.{c}": dict(vid) for c in ("head", "left_wrist", "right_wrist")}
    feats["observation.state"] = {"dtype": "float32", "shape": (act_dim,), "names": None}
    feats["observation.object"] = {"dtype": "float32", "shape": (obj_dim,), "names": None}
    feats["action"] = {"dtype": "float32", "shape": (act_dim,), "names": None}
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/record_pick_up_marker.yaml")
    ap.add_argument("--total", type=int, default=None,
                    help="覆盖本次累计目标条数（分批前台跑：每次比已有多 N 条）")
    ap.add_argument("--smoke", action="store_true", help="写入 *_smoke 数据集做链路验证")
    args = ap.parse_args()

    rc = yaml.safe_load((ROOT / args.cfg).read_text())
    if args.total is not None:
        rc["total"] = args.total
    ecfg = load_config(ROOT / rc["expert_cfg"])

    env_cfg = BimanualDexConfig(max_episode_seconds=float(rc["episode_seconds"]))
    env = BimanualDexEnv(config=env_cfg, render_mode="rgb_array")
    unwrapped = env
    cams = list(unwrapped.cfg.cameras)
    size = int(unwrapped.cfg.image_size)
    orig_randomize = unwrapped._randomize_pen

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = {"train": rc["train_name"], "val": rc["val_name"]}
    if args.smoke:
        names = {k: v + "_smoke" for k, v in names.items()}
    out_root = ROOT / rc["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    ds, n_have = {}, {}
    for split, nm in names.items():
        root = out_root / nm
        d = LeRobotDataset.create(f"local/{nm}", fps=int(rc["fps"]),
                                  features=build_features(size), root=root,
                                  robot_type="bimanual_dex")
        ds[split] = d
        n_have[split] = int(d.meta.total_episodes)
    n0 = sum(n_have.values())
    print(f"resume: train={n_have['train']} val={n_have['val']} target={rc['total']}")
    if n0 >= rc["total"]:
        print("already complete")
        env.close()
        return

    yaw = float(rc["yaw"])
    points = [tuple(map(float, p)) for p in rc["points"]]
    t0 = time.time()
    fail_streak = 0
    g = n0
    while g < rc["total"]:
        rng_i = np.random.default_rng(int(rc["seed"]) * 100003 + g)  # 每条独立可复现
        split = "val" if rng_i.random() < float(rc["val_ratio"]) else "train"
        px, py = points[int(rng_i.integers(len(points)))]
        env_cfg.pen_xy = ((px, px), (py, py))

        def fixed_pen(d, _px=px, _py=py):
            orig_randomize(d)
            c, s = np.cos(yaw / 2), np.sin(yaw / 2)
            qyaw = np.array([c, 0.0, 0.0, s])
            qflat = np.array([0.7071068, 0.0, 0.7071068, 0.0])
            d.qpos[3:7] = unwrapped._quat_mul(qyaw, qflat)

        unwrapped._randomize_pen = fixed_pen
        env_seed = int(rng_i.integers(1 << 30))
        obs, _ = env.reset(seed=env_seed)

        frames, max_z, tries_used = None, 0.0, 0
        for attempt in range(int(rc["tries"])):
            ecfg_a = dict(ecfg)
            ecfg_a["plan_seed"] = attempt
            expert = ScriptedExpert(unwrapped, ecfg_a)
            expert.reset(obs)
            buf, max_z = [], 0.0
            while True:
                action = expert.act(obs)
                buf.append({
                    "images": {c: obs["images"][c] for c in cams},
                    "state": np.asarray(obs["state"], np.float32),
                    "object": np.asarray(obs["object"], np.float32),
                    "action": np.asarray(action, np.float32),
                })
                obs, _, term, trunc, info = env.step_rad(action)
                max_z = max(max_z, float(info["pen_z"]))
                if term or trunc:
                    break
            tries_used = attempt + 1
            if bool(info["success"]) and max_z <= float(rc["max_z_sane"]):
                frames = buf
                break
            if attempt + 1 < int(rc["tries"]):
                obs, _ = env.reset(seed=env_seed)  # 同笔位复位重试

        if frames is None:
            fail_streak += 1
            print(f"[{g}] ({px:+.3f},{py:+.3f}) FAIL x{tries_used} "
                  f"streak={fail_streak} elapsed={time.time()-t0:.0f}s", flush=True)
            if fail_streak >= int(rc["fail_streak_stop"]):
                print("ABORT: 连续全败过多，检查专家/包络")
                break
            continue
        fail_streak = 0

        d = ds[split]
        for fr in frames:
            frame = {"task": rc["task"]}
            for c in cams:
                frame[f"observation.images.{c}"] = fr["images"][c]
            frame["observation.state"] = fr["state"]
            frame["observation.object"] = fr["object"]
            frame["action"] = fr["action"]
            d.add_frame(frame)
        d.save_episode()
        n_have[split] += 1
        g += 1
        print(f"[{g}] ({px:+.3f},{py:+.3f}) -> {split} tries={tries_used} "
              f"frames={len(frames)} max_z={max_z:.3f} "
              f"train/val={n_have['train']}/{n_have['val']} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    env.close()
    print(f"DONE total={sum(n_have.values())} train={n_have['train']} val={n_have['val']}")


if __name__ == "__main__":
    main()
