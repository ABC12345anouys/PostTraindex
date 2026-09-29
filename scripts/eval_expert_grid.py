"""M1 鲁棒性网格评测：固定笔位网格逐点跑脚本专家（前台运行，结果才可信）。
用法: MUJOCO_GL=egl python3 scripts/eval_expert_grid.py \
        [--grid narrow|mid12|mid36] [--yaw DEG] [--tag 名字] [--cfg ...]
网格定义（笔在桌面上的 xy 偏移，单位 m）：
  narrow : 3×3，x/y ∈ {-0.05, 0, 0.05}                     （9 点）
  mid12  : 4×3，x ∈ {-0.06,-0.02,0.02,0.06}，y ∈ {-0.09,0,0.09}（12 点）
  mid36  : yaw ∈ {-30°, 0°, 30°} × mid12                    （36 点）
--yaw 指定后只跑该 yaw 档（便于分段前台执行，避免单命令超时）。
每点输出：成功/失败、终止阶段、笔最高 z；汇总写 verify_out/grid_<tag>.json。
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import BimanualDexConfig  # noqa: E402
from envs.bimanual_dex_env import BimanualDexEnv  # noqa: E402
from teleop.scripted_expert import ScriptedExpert, load_config  # noqa: E402

GRIDS = {
    "narrow": [(x, y) for x in (-0.05, 0.0, 0.05) for y in (-0.05, 0.0, 0.05)],
    "mid12": [(x, y) for x in (-0.06, -0.02, 0.02, 0.06)
              for y in (-0.09, 0.0, 0.09)],
}
GRIDS["mid36"] = GRIDS["mid12"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/pick_up_marker.yaml")
    ap.add_argument("--grid", default="mid12", choices=list(GRIDS))
    ap.add_argument("--yaw", type=float, default=None,
                    help="只跑该 yaw（度）；mid36 不指定则跑全 3 档")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--hand", default=None, choices=["left", "right"],
                    help="覆盖 allowed_hands 只用该手（诊断左手不稳定分支用）")
    ap.add_argument("--diag", action="store_true",
                    help="只跑 expert.reset，打印各候选 descend 标定残差，不跑 episode")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.cfg)
    if args.hand:
        cfg["allowed_hands"] = [args.hand]
    yaws = (-30.0, 0.0, 30.0) if args.grid == "mid36" else (0.0,)
    if args.yaw is not None:
        yaws = (args.yaw,)
    pts = GRIDS["mid12" if args.grid == "mid36" else args.grid]

    env_cfg = BimanualDexConfig(max_episode_seconds=12)
    env = BimanualDexEnv(config=env_cfg, render_mode="rgb_array")
    unwrapped = env
    orig_randomize = unwrapped._randomize_pen
    fixed_qflat = np.array([0.7071068, 0.0, 0.7071068, 0.0])
    results = []
    for yaw_deg in yaws:
        yaw = np.radians(yaw_deg)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        qyaw = np.array([c, 0.0, 0.0, s])

        def fixed_pen(d, _qyaw=qyaw):
            orig_randomize(d)
            d.qpos[3:7] = unwrapped._quat_mul(_qyaw, fixed_qflat)

        for i, (px, py) in enumerate(pts):
            unwrapped._randomize_pen = fixed_pen
            env_cfg.pen_xy = ((px, px), (py, py))
            obs, _ = env.reset(seed=1000 + i)
            expert = ScriptedExpert(unwrapped, cfg)
            expert.reset(obs)
            if args.diag:
                cs = {f"{h}/m{m}": round(float(v), 4)
                      for (h, m), v in expert.cand_scores.items()}
                print(f"yaw={yaw_deg:+.0f} x={px:+.2f} y={py:+.2f} "
                      f"pick={expert.side}/m{expert.mode} scores={cs}", flush=True)
                continue
            max_z = 0.0
            stage = cfg["stages"][0]["name"]
            info = {}
            while True:
                action = expert.act(obs)
                obs, _, term, trunc, info = env.step_rad(action)
                max_z = max(max_z, float(info["pen_z"]))
                stage = cfg["stages"][min(expert.si, len(cfg["stages"]) - 1)]["name"]
                if term or trunc:
                    break
            ok = bool(info["success"])
            results.append({"yaw": yaw_deg, "x": px, "y": py, "success": ok,
                            "hand": expert.side, "mode": expert.mode,
                            "stage": stage, "max_z": round(max_z, 4)})
            print(f"yaw={yaw_deg:+.0f} x={px:+.2f} y={py:+.2f} "
                  f"{'SUC' if ok else 'FAIL'} hand={expert.side} m={expert.mode} "
                  f"stage={stage:8s} max_z={max_z:.3f}",
                  flush=True)

    env.close()
    n = len(results)
    ns = sum(r["success"] for r in results)
    print(f"\n== {args.tag}: {ns}/{n}"
          + (f" = {ns / n:.0%} ==" if n else " =="))
    out = ROOT / "verify_out"
    out.mkdir(exist_ok=True)
    (out / f"grid_{args.tag}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"saved {out / f'grid_{args.tag}.json'}")


if __name__ == "__main__":
    main()
