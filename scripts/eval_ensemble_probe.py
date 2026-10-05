"""M4 n8/n9 ensemble 复评探针：在 RL harness 内对零残差（纯 ACT）策略做 K 策略 × R 轮
12 变体贪心评测，量化 EGL 1px 渲染噪声造成的成功率抖动（PROGRESS 问题 44）。

用法：
  MUJOCO_GL=egl <py> scripts/eval_ensemble_probe.py \
      --name k3 --ckpts checkpoints/act_m4/checkpoints/030000/pretrained_model \
                          checkpoints/act_m4/checkpoints/035000/pretrained_model \
                          checkpoints/act_m4/checkpoints/040000/pretrained_model \
      --reps 3
结果追加写入 verify_out/n8_ensemble_probe.json（按 name 去重覆盖同名组）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402
import yaml  # noqa: E402


def load_rt():
    spec = importlib.util.spec_from_file_location(
        "rt", ROOT / "policy" / "residual_td3_transfer.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    return rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    rt = load_rt()
    rc = yaml.safe_load((ROOT / "configs" / "residual_td3_transfer.yaml").read_text())
    rt.A_DIM = sum(b - a for a, b in [(7, 27), (34, 54)])
    rc["_slices"] = [(7, 27), (34, 54)]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(args.ckpts) == 1:
        act = rt.ACTBase(ROOT / args.ckpts[0], 0.01, device)
    else:
        from policy.act_ensemble import ACTEnsemble
        act = ACTEnsemble([ROOT / c for c in args.ckpts], 0.01, device)
    print(f"[{args.name}] K={len(args.ckpts)} ensemble 加载完成", flush=True)

    env, unw = rt.make_env(rc["episode_seconds"])
    pfx = rt.Prefix(env, unw, ROOT / "configs" / "transfer_marker.yaml")
    pool = json.loads((ROOT / rc["pool"]).read_text())
    variants, seen = [], set()
    for r in pool:
        k = (r["point"][0], r["point"][1], r["plan_seed"])
        if k not in seen:
            seen.add(k)
            variants.append({"point": r["point"], "plan_seed": r["plan_seed"]})

    class Zero:
        def __call__(self, x):
            return torch.zeros(x.shape[0], rt.A_DIM, device=device)

    rep_rates, rep_detail = [], []
    for rep in range(args.reps):
        rate, res = rt.evaluate(env, unw, pfx, act, Zero(), rc, device,
                                variants, f"{args.name}-r{rep}")
        rep_rates.append(round(rate, 4))
        rep_detail.append([int(x["success"]) for x in res])
        n_s = sum(rep_detail[-1])
        print(f"[{args.name} rep{rep}] {n_s}/{len(variants)} = {rate:.0%}", flush=True)
    env.close()

    total = int(sum(np.mean(rep_detail, axis=0) >= 0.5) if False else 0)
    succ_all = sum(sum(d) for d in rep_detail)
    entry = {
        "name": args.name,
        "k": len(args.ckpts),
        "ckpts": args.ckpts,
        "reps": rep_rates,
        "per_rep_success": rep_detail,
        "merged_trials": f"{succ_all}/{args.reps * len(variants)}",
        "merged_rate": round(succ_all / (args.reps * len(variants)), 4),
        "spread_pct_points": round((max(rep_rates) - min(rep_rates)) * 100, 1),
    }
    out = ROOT / "verify_out" / "n8_ensemble_probe.json"
    data = {}
    if out.exists():
        data = json.loads(out.read_text())
    data[args.name] = entry
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(json.dumps(entry, ensure_ascii=False))


if __name__ == "__main__":
    main()
