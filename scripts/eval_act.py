"""M2 闭环评测：加载 ACT checkpoint 在固定笔位网格上跑成功率（lerobot-env / mujoco 3.7.0）。

用法:
  MUJOCO_GL=egl <lerobot-env>/bin/python scripts/eval_act.py \
      --ckpt checkpoints/act_m2/checkpoints/020000/pretrained_model \
      --grid whitelist9 --tag act20k_wl9
  --grid: whitelist9（录数 9 点）| narrow9 | rand20（白名单点 ±1.5cm 抖动 20 次）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import BimanualDexConfig  # noqa: E402
from envs.bimanual_dex_env import BimanualDexEnv  # noqa: E402

WL9 = [(-0.025, 0.0), (0.0, -0.05), (0.0, 0.0), (0.05, -0.04), (0.05, -0.01),
       (0.05, 0.0), (0.075, -0.05), (0.075, 0.0), (0.075, 0.025)]
NARROW9 = [(x, y) for x in (-0.05, 0.0, 0.05) for y in (-0.05, 0.0, 0.05)]
CAMS = ("head", "left_wrist", "right_wrist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model 目录")
    ap.add_argument("--grid", default="whitelist9",
                    choices=["whitelist9", "narrow9", "rand20"])
    ap.add_argument("--tag", default="act_run")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ensemble", type=float, default=0.01,
                    help="temporal ensemble 系数（ACT 原版 0.01，默认开；传 -1 关闭用 queue）")
    ap.add_argument("--n_actions", type=int, default=None,
                    help="覆盖 n_action_steps（queue 模式每 chunk 执行步数 C）")
    args = ap.parse_args()

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(args.ckpt)
    if args.ensemble >= 0:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.config.temporal_ensemble_coeff = args.ensemble
        policy.temporal_ensembler = ACTTemporalEnsembler(
            args.ensemble, policy.config.chunk_size)
    if args.n_actions is not None:
        policy.config.n_action_steps = args.n_actions
    policy.config.device = device
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": device}})

    if args.grid == "whitelist9":
        pts = WL9
    elif args.grid == "narrow9":
        pts = NARROW9
    else:
        rng = np.random.default_rng(args.seed)
        pts = []
        for _ in range(20):
            bx, by = WL9[int(rng.integers(len(WL9)))]
            pts.append((float(np.clip(bx + rng.uniform(-0.015, 0.015), -0.1, 0.1)),
                        float(np.clip(by + rng.uniform(-0.015, 0.015), -0.1, 0.1))))

    env_cfg = BimanualDexConfig(max_episode_seconds=12)
    env = BimanualDexEnv(config=env_cfg, render_mode="rgb_array")
    unwrapped = env
    orig_randomize = unwrapped._randomize_pen
    fixed_qflat = np.array([0.7071068, 0.0, 0.7071068, 0.0])

    results = []
    for i, (px, py) in enumerate(pts):
        def fixed_pen(d):
            orig_randomize(d)
            d.qpos[3:7] = unwrapped._quat_mul(np.array([1.0, 0.0, 0.0, 0.0]), fixed_qflat)

        unwrapped._randomize_pen = fixed_pen
        env_cfg.pen_xy = ((px, px), (py, py))
        obs, _ = env.reset(seed=3000 + i)
        policy.reset()
        max_z, frames = 0.0, 0
        while True:
            # 与 train.py:438 一致：图像 uint8 → float32/255；AddBatch 由 preprocessor 完成
            ob = {}
            for c in CAMS:
                ob[f"observation.images.{c}"] = torch.from_numpy(
                    obs["images"][c]).permute(2, 0, 1).to(torch.float32) / 255.0  # HWC→CHW
            ob["observation.state"] = torch.from_numpy(
                np.asarray(obs["state"], np.float32))
            ob["observation.object"] = torch.from_numpy(
                np.asarray(obs["object"], np.float32))
            with torch.inference_mode():
                out = post(policy.select_action(pre(ob)))
            act = out["action"] if isinstance(out, dict) else out
            a = act.squeeze(0).cpu().numpy().astype(np.float64)
            obs, _, term, trunc, info = env.step_rad(a)
            max_z = max(max_z, float(info["pen_z"]))
            frames += 1
            if term or trunc:
                break
        ok = bool(info["success"])
        results.append({"x": px, "y": py, "success": ok, "frames": frames,
                        "max_z": round(max_z, 4)})
        print(f"[{i+1:2d}/{len(pts)}] ({px:+.3f},{py:+.3f}) "
              f"{'SUC' if ok else 'FAIL'} frames={frames} max_z={max_z:.3f}", flush=True)

    env.close()
    n = len(results)
    ns = sum(r["success"] for r in results)
    print(f"\n== {args.tag}: {ns}/{n} = {ns/n:.0%} ==")
    out = ROOT / "verify_out"
    out.mkdir(exist_ok=True)
    (out / f"act_{args.tag}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"saved {out / f'act_{args.tag}.json'}")


if __name__ == "__main__":
    main()
