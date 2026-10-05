"""残差 TD3 曲线：success-vs-transitions。
默认（n6，rl_m4）：左=49 eval 点+滑动均值+噪声带；右=三策略 3 轮复评误差棒+Fisher。
n10（rl_m4_k3）：K=3 ensemble 基线下评测已确定性（n8 复现 0pp 极差），单面板即可。

用法：
  <py> scripts/plot_rl_m4.py                                          # n6
  <py> scripts/plot_rl_m4.py --run rl_m4_k3 --baseline-pct 58.3 \
       --baseline-label "K=3 ensemble 冻结基线 58%（确定性复评）"
"""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fisher_p(a_succ, a_n, b_succ, b_n):
    n, k = a_n + b_n, a_succ + b_succ
    lo = max(0, k - b_n)
    hi = min(k, a_n)
    denom = math.comb(n, k)

    def prob(x):
        return math.comb(a_n, x) * math.comb(b_n, k - x) / denom
    p_obs = prob(a_succ)
    return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs + 1e-12)


def load_points(run):
    log_p = ROOT / "checkpoints" / run / "log.jsonl"
    rows = [json.loads(line) for line in log_p.read_text().splitlines() if line.strip()]
    pts = [(r["transitions"], r["pool_rate"]) for r in rows]
    final_p = ROOT / "verify_out" / f"{run}_final.json"
    if final_p.exists():
        f = json.loads(final_p.read_text())
        pts.append((f["transitions"], f["pool"]["rate"]))
    pts.sort()
    return rows, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rl_m4")
    ap.add_argument("--baseline-pct", type=float, default=47.0)
    ap.add_argument("--baseline-label", default="纯 ACT 基线（36 trial 合并 47%）")
    ap.add_argument("--noise-json", default=None,
                    help="三策略复评 json；不给则单面板（确定性评测口径）")
    args = ap.parse_args()

    rows, pts = load_points(args.run)
    with open(ROOT / "verify_out" / f"{args.run}_curve.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["transitions", "pool_rate"])
        w.writerows([(t, round(v * 100, 1)) for t, v in pts])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]

    xs = [t for t, _ in pts]
    ys = [v * 100 for _, v in pts]
    n_var = len(rows[-1]["results"]) if rows else 12

    if args.noise_json:
        probe = json.loads(Path(args.noise_json).read_text())
        z, b, fn = probe["zero"], probe["best"], probe["final50k"]
        z_s, b_s = round(sum(z) * 12), round(sum(b) * 12)
        p_zb = fisher_p(b_s, 36, z_s, 36)
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                      gridspec_kw={"width_ratios": [2.1, 1]})
    else:
        fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(xs, ys, "-", color="#c0392b", lw=1.2, alpha=0.5,
            label="逐次评测（12 变体贪心）")
    ax.scatter(xs, ys, s=22, color="#c0392b", zorder=3)
    win = 5
    ma = [sum(ys[max(0, i - win + 1):i + 1]) / min(i + 1, win) for i in range(len(ys))]
    ax.plot(xs, ma, color="#7b241c", lw=2.2, label="5 点滑动均值")
    ax.axhline(args.baseline_pct, ls="--", color="#4472a8", label=args.baseline_label)
    if args.noise_json:
        ax.axhspan(min(z) * 100, max(z) * 100, color="#4472a8", alpha=0.12)
    ax.axhline(90, ls=":", color="#2e7d32", label="目标 90%")
    ax.set_xlabel("交互 transitions")
    ax.set_ylabel(f"闭环成功率（{n_var} 变体）")
    ax.set_ylim(-5, 100)
    ax.set_title(f"{args.run} raw 残差 TD3：success-vs-transitions（50k）")
    ax.grid(alpha=0.3)

    if args.noise_json:
        ax.legend(loc="upper right", fontsize=8.5)
        labels = ["纯 ACT\n残差=0", "best actor\n(训中峰值存档)", "末段 actor\n50k"]
        means = [sum(v) / 3 * 100 for v in (z, b, fn)]
        rngs = [[min(v) * 100, max(v) * 100] for v in (z, b, fn)]
        colors = ["#4472a8", "#e67e22", "#7f8c8d"]
        for i, (m, rng, c) in enumerate(zip(means, rngs, colors)):
            ax2.bar(i, m, color=c, width=0.55, alpha=0.85)
            ax2.plot([i, i], rng, color="k", lw=1.2)
            ax2.plot([i - 0.08, i + 0.08], [rng[0]] * 2, color="k", lw=1.2)
            ax2.plot([i - 0.08, i + 0.08], [rng[1]] * 2, color="k", lw=1.2)
            ax2.annotate(f"{m:.0f}%", (i, m), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=10)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.set_ylim(-5, 100)
        ax2.set_ylabel("3 轮复评平均成功率（误差棒=单轮极差）")
        ax2.set_title(f"评测噪声校准：best vs 基线\nFisher p={p_zb:.2f}", fontsize=10)
        ax2.grid(alpha=0.3, axis="y")
    else:
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = ROOT / "verify_out" / f"{args.run}_curve.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
