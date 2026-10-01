"""M4 n4：浅轴候选 × 参数池全程验证（构建 transfer 录数成功池）。

对 scan_candidates.json 每个候选跑 proto_transfer 全程 × 3 参数组合
（默认 / left_goal_dz=0.05 / catch_close_start=0.5），并行子进程，
收集互异成功 (point, plan_seed, params) 组合 → transfer_success_pool.json。
确定性：同参数重跑 = 逐比特相同（录数阶段直接复现，无需重验）。

用法:
  MUJOCO_GL=egl <lerobot-env>/bin/python scripts/run_shallow.py [--workers 8]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/lifd/anaconda3/envs/lerobot-env/bin/python"

PARAM_SETS = [("def", []),
              ("dz05", ["--set", "left_goal_dz=0.05"]),
              ("cs05", ["--set", "catch_close_start=0.5"]),
              ("dz03", ["--set", "left_goal_dz=0.03"]),
              ("dz06", ["--set", "left_goal_dz=0.06"]),
              ("cs04", ["--set", "catch_close_start=0.4"]),
              ("cs07", ["--set", "catch_close_start=0.7"]),
              ("am01", ["--set", "along=-0.01"]),
              ("ap01", ["--set", "along=0.01"])]


def run_one(cand, pname, pargs, out_root):
    x, y = cand["point"]
    ps = cand["plan_seed"]
    tag = f"x{x:+.3f}_y{y:+.3f}_s{ps}_{pname}".replace("+", "p").replace("-", "m")
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)
    res_file = out / "result.json"
    if res_file.exists():                       # 断点续跑
        res = json.loads(res_file.read_text())
        return {**cand, "param_set": pname, "success": bool(res.get("success")),
                "out": str(out.relative_to(ROOT)), "cached": True}
    cmd = [PY, str(ROOT / "scripts/proto_transfer.py"),
           "--x", str(x), "--y", str(y), "--seed", "1002",
           "--plan-seed", str(ps), "--out", str(out.relative_to(ROOT))] + pargs
    log = out / "run.log"
    try:
        with log.open("w") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           timeout=290)
    except subprocess.TimeoutExpired:
        return {**cand, "param_set": pname, "success": False,
                "out": str(out.relative_to(ROOT)), "timeout": True}
    res = json.loads(res_file.read_text()) if res_file.exists() else {}
    return {**cand, "param_set": pname, "success": bool(res.get("success")),
            "out": str(out.relative_to(ROOT))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="verify_out/scan_pick/scan_candidates.json")
    ap.add_argument("--out", default="verify_out/transfer_pool")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cands = json.loads((ROOT / args.candidates).read_text())
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = [(c, pn, pa) for c in cands for pn, pa in PARAM_SETS]
    print(f"[pool] {len(cands)} 候选 × {len(PARAM_SETS)} 参数 = {len(jobs)} 全程运行",
          flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, c, pn, pa, out_root): (c, pn)
                for c, pn, pa in jobs}
        for i, fut in enumerate(futs):
            r = fut.result()
            rows.append(r)
            mark = "OK " if r["success"] else "xxx"
            print(f"[pool] {i + 1}/{len(jobs)} {mark} ({r['point'][0]:+.3f},"
                  f"{r['point'][1]:+.3f}) ps={r['plan_seed']} {r['param_set']}",
                  flush=True)
            (out_root / "pool_runs.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=1))

    pool = [r for r in rows if r["success"]]
    (out_root / "transfer_success_pool.json").write_text(
        json.dumps(pool, ensure_ascii=False, indent=1))
    n_pts = len({(r["point"][0], r["point"][1]) for r in pool})
    print(f"[pool] 成功组合 {len(pool)}/{len(jobs)}，覆盖桌面点 {n_pts} 个")

    # ---- 成功矩阵图（候选 × 参数集）----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    labels = [f"({c['point'][0]*1000:+.0f},{c['point'][1]*1000:+.0f})s{c['plan_seed']}"
              f"\n{c['angle_deg']:.1f}°" for c in cands]
    M = np.zeros((len(cands), len(PARAM_SETS)))
    for r in rows:
        ci = next(i for i, c in enumerate(cands)
                  if c["point"] == r["point"] and c["plan_seed"] == r["plan_seed"])
        pi = [p[0] for p in PARAM_SETS].index(r["param_set"])
        M[ci, pi] = 1.0 if r["success"] else 0.0
    fig, ax = plt.subplots(figsize=(max(8, len(cands) * 0.8), 3.5))
    ax.imshow(M.T, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_yticks(range(len(PARAM_SETS)))
    ax.set_yticklabels([p[0] for p in PARAM_SETS])
    for ci in range(len(cands)):
        for pi in range(len(PARAM_SETS)):
            ax.text(ci, pi, "✓" if M[ci, pi] else "✗", ha="center", va="center",
                    color="#222", fontsize=11)
    ax.set_title(f"M4 n4 浅轴候选 × 参数池：transfer 全程成功 {len(pool)}/{len(jobs)}")
    fig.tight_layout()
    fig.savefig(out_root / "pool_matrix.png", dpi=150)
    print(f"[pool] 图: {out_root / 'pool_matrix.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
