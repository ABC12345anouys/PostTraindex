"""M4 n6 残差 TD3 曲线：success-vs-transitions（pool 12 变体贪心评测）。
数据源：
  checkpoints/rl_m4/log.jsonl        —— 训练中 49 个 eval 点（每 10 episode）
  verify_out/rl_m4_eval_noise.json   —— 三策略 ×3 轮复评（量化 ACT latent 采样噪声）
  verify_out/rl_m4_final.json        —— --finalize best actor 单轮终评（留档）

结论口径：单点成功率含 ±2~4/12 的评测随机性，判读以 36 trial 合并成功率为准。
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "checkpoints" / "rl_m4" / "log.jsonl"
NOISE = ROOT / "verify_out" / "rl_m4_eval_noise.json"
FINAL = ROOT / "verify_out" / "rl_m4_final.json"


def fisher_p(a_succ, a_n, b_succ, b_n):
    """双侧 Fisher 精确检验（固定边际，超几何枚举）。"""
    n, k = a_n + b_n, a_succ + b_succ
    lo = max(0, k - b_n)
    hi = min(k, a_n)
    denom = math.comb(n, k)

    def prob(x):
        return math.comb(a_n, x) * math.comb(b_n, k - x) / denom
    p_obs = prob(a_succ)
    return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs + 1e-12)


def main():
    rows = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    pts = [(r["transitions"], r["pool_rate"]) for r in rows]
    if FINAL.exists():
        f = json.loads(FINAL.read_text())
        pts.append((f["transitions"], f["pool"]["rate"]))
    pts.sort()

    probe = json.loads(NOISE.read_text())
    z, b, fn = probe["zero"], probe["best"], probe["final50k"]
    z_s, b_s, f_s = round(sum(z) * 12), round(sum(b) * 12), round(sum(fn) * 12)
    p_zb = fisher_p(b_s, 36, z_s, 36)
    summary = {
        "replicates": 3, "variants": 12,
        "zero_mean": sum(z) / len(z), "zero_trials": f"{z_s}/36",
        "best_mean": sum(b) / len(b), "best_trials": f"{b_s}/36",
        "final_mean": sum(fn) / len(fn), "final_trials": f"{f_s}/36",
        "single_eval_range": {"best": [min(b), max(b)], "final": [min(fn), max(fn)],
                              "zero": [min(z), max(z)]},
        "fisher_p_best_vs_zero": round(p_zb, 3),
    }
    (ROOT / "verify_out" / "rl_m4_eval_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))

    with open(ROOT / "verify_out" / "rl_m4_curve.csv", "w", newline="") as fp:
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

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                  gridspec_kw={"width_ratios": [2.1, 1]})
    ax.plot(xs, ys, "-", color="#c0392b", lw=1.2, alpha=0.5,
            label="逐次评测（单轮，含采样噪声）")
    ax.scatter(xs, ys, s=22, color="#c0392b", zorder=3)
    win = 5
    ma = [sum(ys[max(0, i - win + 1):i + 1]) / min(i + 1, win) for i in range(len(ys))]
    ax.plot(xs, ma, color="#7b241c", lw=2.2, label="5 点滑动均值")
    zm = sum(z) / len(z) * 100
    ax.axhline(zm, ls="--", color="#4472a8",
               label=f"纯 ACT 基线（36 trial 合并 {zm:.0f}%）")
    ax.axhspan(min(z) * 100, max(z) * 100, color="#4472a8", alpha=0.12)
    ax.axhline(90, ls=":", color="#2e7d32", label="n6 目标 90%")
    ax.set_xlabel("交互 transitions")
    ax.set_ylabel(f"闭环成功率（{n_var} 变体）")
    ax.set_ylim(-5, 100)
    ax.set_title("n6 raw 残差 TD3：success-vs-transitions（50k）")
    ax.grid(alpha=0.3)
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
    ax2.set_title(f"评测噪声校准：best vs 基线\nFisher p={p_zb:.2f}（不显著）",
                  fontsize=10)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = ROOT / "verify_out" / "rl_m4_curve.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] {out}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
