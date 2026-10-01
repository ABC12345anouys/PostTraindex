"""M4 Transfer 跨姿态鲁棒扫测：WL9 九点 × plan_seed，跑 proto_transfer 并汇总成图。

每个组合串行跑（单组合 ~2min，<5min 上限），聚合 normalize 前后笔轴夹角与
transfer 成功率，输出 summary.json + normalize_sweep.png。

用法:
  MUJOCO_GL=egl python3 scripts/sweep_transfer.py [--plan-seeds 4] [--points 0-8]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/lifd/anaconda3/envs/lerobot-env/bin/python"

WL9 = [(-0.025, 0.0), (0.0, -0.05), (0.0, 0.0), (0.05, -0.04), (0.05, -0.01),
       (0.05, 0.0), (0.075, -0.05), (0.075, 0.0), (0.075, 0.025)]


def run_one(i, ps, out_root):
    x, y = WL9[i]
    out = out_root / f"p{i}_s{ps}"
    log = out / "run.log"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(ROOT / "scripts/proto_transfer.py"),
           "--x", str(x), "--y", str(y), "--seed", str(1000 + i),
           "--plan-seed", str(ps), "--out", str(out.relative_to(ROOT))]
    with log.open("w") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.DEVNULL,
                           timeout=290)
    res_file = out / "result.json"
    res = json.loads(res_file.read_text()) if res_file.exists() else {}
    nm = res.get("normalize", {})
    pick_ok = bool(res.get("pick_grasped", False))
    ok = bool(res.get("success", False))
    cat = "success" if ok else ("pick_fail" if not pick_ok else "transfer_fail")
    return {"point": [x, y], "plan_seed": ps, "exit": r.returncode,
            "success": ok, "category": cat, "pick_grasped": pick_ok,
            "settle_axis": res.get("settle_axis"),
            "angle_before": nm.get("angle_before_deg"),
            "angle_after": nm.get("angle_after_deg"),
            "norm_iters": nm.get("iters", 0),
            "pen_z_end": res.get("phases", [{}])[-1].get("pen_z_end")}


def make_figure(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    seeds = sorted({r["plan_seed"] for r in rows})
    fig, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 5), sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    for ax, ps in zip(axes, seeds):
        rs = [r for r in rows if r["plan_seed"] == ps]
        labels = [f"p{i}\n({r['point'][0]:.2f},{r['point'][1]:.2f})"
                  for i, r in enumerate(rs)]
        xs = np.arange(len(rs))
        before = [r["angle_before"] if r["angle_before"] is not None else 0 for r in rs]
        after = [r["angle_after"] if r["angle_after"] is not None else 0 for r in rs]
        ax.bar(xs - 0.2, before, width=0.4, label="normalize 前", color="#d62728", alpha=0.8)
        ax.bar(xs + 0.2, after, width=0.4, label="normalize 后", color="#1f77b4", alpha=0.8)
        for x, r in zip(xs, rs):
            mark = "✓" if r["success"] else "✗"
            ax.text(x, max(before[int(x)], after[int(x)]) + 2, mark,
                    ha="center", fontsize=14,
                    color="#2ca02c" if r["success"] else "#888888")
        n_ok = sum(r["success"] for r in rs)
        ax.set_title(f"plan_seed={ps}  transfer 成功 {n_ok}/{len(rs)}")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("笔轴与标准轴夹角 (°)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("M4 Transfer 跨 pick 姿态鲁棒：笔轴规范化效果 + 逐点成功率（WL9）")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[sweep] 图: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-seeds", type=int, nargs="+", default=[4])
    ap.add_argument("--points", default="0-8", help="如 0-8 或 2,5")
    ap.add_argument("--out", default="verify_out/transfer_grid")
    args = ap.parse_args()

    if "-" in args.points:
        a, b = args.points.split("-")
        idxs = list(range(int(a), int(b) + 1))
    else:
        idxs = [int(v) for v in args.points.split(",")]

    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for ps in args.plan_seeds:
        for i in idxs:
            print(f"[sweep] point{i} {WL9[i]} plan_seed={ps} ...", flush=True)
            try:
                row = run_one(i, ps, out_root)
            except subprocess.TimeoutExpired:
                row = {"point": list(WL9[i]), "plan_seed": ps, "exit": -1,
                       "success": False, "category": "timeout",
                       "angle_before": None, "angle_after": None,
                       "norm_iters": 0, "pen_z_end": None}
            row["index"] = i
            rows.append(row)
            print(f"    -> {row.get('category','?')} 轴偏 "
                  f"{row['angle_before']}→{row['angle_after']} iters={row['norm_iters']}",
                  flush=True)
            (out_root / "summary.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=1))

    n_ok = sum(r["success"] for r in rows)
    print(f"[sweep] 总计 {n_ok}/{len(rows)} 成功")
    make_figure(rows, out_root / "normalize_sweep.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
