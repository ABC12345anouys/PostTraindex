"""M4 n5 评测扫描 + 曲线：逐 ckpt 跑 eval_act_transfer（新口径），
汇总 成功率-vs-步数 + loss 曲线 一张图 -> verify_out/act_m4_curves.png。

用法:
  MUJOCO_GL=egl <py> scripts/eval_sweep_m4.py --steps 10000 20000 30000 40000
  MUJOCO_GL=egl <py> scripts/eval_sweep_m4.py --plot-only   # 只重画图
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_eval(step, args):
    tag = f"{args.tag_prefix}_{step // 1000}k_v3"
    out = ROOT / "verify_out" / f"eval_transfer_{tag}" / "results.json"
    if out.exists():
        print(f"[sweep] {tag} 已有结果，跳过", flush=True)
    else:
        ckpt = Path(args.ckpt_root) / f"{step:06d}" / "pretrained_model"
        if not ckpt.exists():
            print(f"[sweep] {ckpt} 不存在，跳过", flush=True)
            return None
        cmd = [sys.executable, str(ROOT / "scripts" / "eval_act_transfer.py"),
               "--ckpt", str(ckpt), "--tag", tag]
        print(f"[sweep] eval {tag} ...", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    r = json.loads(out.read_text())
    ns = sum(x["success"] for x in r)
    return {"step": step, "ns": ns, "n": len(r), "rate": ns / len(r), "tag": tag}


def plot(rows, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]

    steps, losses = [], []
    for m in re.finditer(r"step:(\d+).*?loss:([0-9.]+)", Path(args.log).read_text()):
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(steps, losses, lw=0.8, color="#4472a8")
    ax1.set_xlabel("训练步数")
    ax1.set_ylabel("loss")
    ax1.set_title(f"{args.title_prefix} ACT 训练 loss")
    ax1.grid(alpha=0.3)

    xs = [r["step"] for r in rows]
    ys = [r["rate"] for r in rows]
    ax2.plot(xs, ys, "o-", color="#c0392b")
    for r in rows:
        ax2.annotate(f"{r['ns']}/{r['n']}", (r["step"], r["rate"]),
                     textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
    ax2.set_xlabel("训练步数")
    ax2.set_ylabel("闭环成功率（12 变体）")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(f"{args.title_prefix} 闭环成功率 vs 训练步数（lh&!rh 口径）")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=150)
    print(f"[plot] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="*",
                    default=[10000, 20000, 30000, 40000])
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--ckpt-root", default=str(ROOT / "checkpoints" / "act_m4" / "checkpoints"))
    ap.add_argument("--log", default=str(ROOT / "verify_out" / "train_act_m4_40k.log"))
    ap.add_argument("--tag-prefix", default="act_m4")
    ap.add_argument("--title-prefix", default="M4 n5（46 条多模态）")
    ap.add_argument("--agg", default=str(ROOT / "verify_out" / "act_m4_sweep.json"))
    ap.add_argument("--out", default=str(ROOT / "verify_out" / "act_m4_curves.png"))
    args = ap.parse_args()

    agg_p = Path(args.agg)
    rows = json.loads(agg_p.read_text()) if agg_p.exists() else []
    have = {r["step"] for r in rows}
    if not args.plot_only:
        for s in args.steps:
            if s in have:
                continue
            r = run_eval(s, args)
            if r:
                rows.append(r)
                rows.sort(key=lambda x: x["step"])
                agg_p.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for r in rows:
        print(f"  {r['step']:>6}: {r['ns']}/{r['n']} = {r['rate']:.0%}")
    plot(rows, args)


if __name__ == "__main__":
    main()
