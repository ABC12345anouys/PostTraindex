"""M4 n4：自然 pick 变体扫描器（浅轴候选筛选，问题 35-39 后的录数路线）。

单进程循环 桌面点 × plan_seed：reset（yaw0 同协议）→ 右手 pick 专家 → settle 25 帧，
记录 settle 笔心/笔轴/pen_z/grasped。筛 |Δaxis|<10°（对 a_star=[0.903,-0.224,-0.366]）
且 pen_z 达标的浅轴候选，供 proto_transfer 全程 × 参数池收集互异成功组合。

物理与 proto_transfer pick 段完全一致（同 env 构造、fixed_pen yaw0、plan_seed 注入、
pick 后才开臂阻尼）；render_mode=None 不渲染（渲染不影响物理，probe 提速）。

用法:
  # 分片跑（每片独立进程，结果增量写盘，可断点续跑）
  /home/lifd/anaconda3/envs/lerobot-env/bin/python scripts/scan_pick_poses.py \
      --shard 0 --shards 8
  # 合并 + 出图 + 候选清单
  /home/lifd/anaconda3/envs/lerobot-env/bin/python scripts/scan_pick_poses.py --merge
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import proto_transfer as PT  # noqa: E402
from envs import BimanualDexConfig  # noqa: E402
from envs.bimanual_dex_env import BimanualDexEnv  # noqa: E402
from teleop.scripted_expert import ScriptedExpert  # noqa: E402

SEED = 1002          # (0,0) yaw0 同协议
YAW = 0.0
A_STAR = np.array([0.903, -0.224, -0.366])
A_STAR = A_STAR / np.linalg.norm(A_STAR)

# 25 点：中心 + 半径 2/5/8/12mm × 6 方向环
POINTS = [(0.0, 0.0)]
for _r in (0.002, 0.005, 0.008, 0.012):
    for _k in range(6):
        _th = np.radians(60 * _k)
        POINTS.append((round(_r * np.cos(_th), 6), round(_r * np.sin(_th), 6)))
PLAN_SEEDS = list(range(10))


def combo_rows(shard, shards, out_dir):
    """跑本片所有组合，增量写 scan_poses_s{shard}.json（可续跑）。"""
    cfg = PT.load_cfg(ROOT / "configs/transfer_marker.yaml")
    tc = cfg["transfer"]
    poses = {k: np.asarray(v, dtype=np.float64) for k, v in cfg["hand_poses"].items()}
    n_settle = int(tc.get("settle_frames", 25))
    arm_damp = float(tc.get("arm_damping", 10.0))

    env_cfg = BimanualDexConfig(max_episode_seconds=20,
                                pen_xy=((0.0, 0.0), (0.0, 0.0)))
    env = BimanualDexEnv(config=env_cfg, render_mode=None)   # 不渲染提速
    unwrapped = env
    PT.build_geom_sides(unwrapped)
    damp0 = unwrapped.model.dof_damping.copy()  # 模型默认阻尼（XML joint damping=1）

    def set_damping(c):
        for _side in ("l", "r"):
            for _ji in range(1, 8):
                _jid = mujoco.mj_name2id(unwrapped.model, mujoco.mjtObj.mjOBJ_JOINT,
                                         f"{_side}_joint{_ji}")
                unwrapped.model.dof_damping[unwrapped.model.jnt_dofadr[_jid]] = c

    orig_randomize = unwrapped._randomize_pen

    def fixed_pen(d):                       # 与 proto_transfer 同协议：yaw 固定
        orig_randomize(d)
        c, s = np.cos(YAW / 2), np.sin(YAW / 2)
        qflat = np.array([0.7071068, 0.0, 0.7071068, 0.0])
        qyaw = np.array([c, 0.0, 0.0, s])
        d.qpos[3:7] = unwrapped._quat_mul(qyaw, qflat)

    unwrapped._randomize_pen = fixed_pen

    # 左手 home/张开 静态保持位（借用一次左 expert 的元数据）
    cfg_l = copy.deepcopy(cfg)
    cfg_l["allowed_hands"] = ["left"]
    expert_meta = ScriptedExpert(unwrapped, cfg_l)
    l_q9 = unwrapped.model.key_qpos[expert_meta._key_home][expert_meta.iks["left"].qadr].copy()
    l_f18 = poses["open"][2:20].copy()
    r_f18 = poses["pinch3"][2:20].copy()    # grip_tighten=0（问题 38 已弃用）

    out_file = out_dir / f"scan_poses_s{shard}.json"
    done = {}
    if out_file.exists():
        for r in json.loads(out_file.read_text()):
            done[(round(r["point"][0], 6), round(r["point"][1], 6), r["plan_seed"])] = r

    combos = [(pi, ps) for pi in range(len(POINTS)) for ps in PLAN_SEEDS]
    todo = [(pi, ps) for i, (pi, ps) in enumerate(combos)
            if i % shards == shard
            and (round(POINTS[pi][0], 6), round(POINTS[pi][1], 6), ps) not in done]
    print(f"[scan] shard {shard}/{shards}: {len(todo)} 个组合待跑（已完成 {len(done)}）",
          flush=True)

    rows = list(done.values())
    for n, (pi, ps) in enumerate(todo):
        x, y = POINTS[pi]
        unwrapped.cfg.pen_xy = ((x, x), (y, y))   # _randomize_pen 每次 reset 读 cfg
        unwrapped.model.dof_damping[:] = damp0    # pick 用模型默认阻尼（同 proto，勿清零）
        obs, _ = env.reset(seed=SEED)

        cfg_r = copy.deepcopy(cfg)
        cfg_r["allowed_hands"] = ["right"]
        cfg_r["plan_seed"] = int(ps)
        expert_r = ScriptedExpert(unwrapped, cfg_r)
        expert_r.reset(obs)
        info = {}
        pick_frames = 0
        while expert_r.si < 4:                    # approach/descend/close/lift
            obs, _, term, trunc, info = env.step_rad(expert_r.act(obs))
            pick_frames += 1
        pick_grasped = bool(info.get("grasped", False))

        set_damping(arm_damp)                     # pick 后才开臂阻尼（同 proto）
        r_q9 = unwrapped.data.qpos[expert_r.ik.qadr].copy()
        for _ in range(n_settle):
            obs, _, _, _, info = env.step_rad(PT.blend_action(l_q9, l_f18, r_q9, r_f18))

        pen = np.asarray(obs["object"][:3], dtype=np.float64)
        axis = PT.pen_axis(obs)
        ang = float(np.degrees(np.arccos(np.clip(np.dot(axis, A_STAR), -1.0, 1.0))))
        pen_low = float(env._pen_lowest_z())
        row = {"point": [x, y], "plan_seed": ps, "pick_grasped": pick_grasped,
               "pick_frames": pick_frames,
               "settle_pen": pen.round(4).tolist(),
               "settle_axis": axis.round(4).tolist(),
               "pen_low_z": round(pen_low, 4),
               "grasped_end": bool(info.get("grasped", False)),
               "angle_deg": round(ang, 2),
               "shallow": bool(pick_grasped and pen_low > 0.6 and ang < 10.0)}
        rows.append(row)
        tag = "SHALLOW" if row["shallow"] else ("no-grasp" if not pick_grasped else "")
        print(f"[scan] {n + 1}/{len(todo)} p{pi} ({x:+.3f},{y:+.3f}) ps={ps} "
              f"轴偏 {ang:5.1f}° pen_low_z={pen_low:.3f} grasped={pick_grasped} {tag}",
              flush=True)
        out_file.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    env.close()
    print(f"[scan] shard {shard} 完成 {len(rows)} 条 → {out_file}")
    return 0


def merge(out_dir):
    rows, seen = [], set()
    for f in sorted(out_dir.glob("scan_poses_s*.json")):
        for r in json.loads(f.read_text()):
            k = (round(r["point"][0], 6), round(r["point"][1], 6), r["plan_seed"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
    (out_dir / "scan_poses.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    cands = sorted([r for r in rows if r["shallow"]], key=lambda r: r["angle_deg"])
    (out_dir / "scan_candidates.json").write_text(
        json.dumps(cands, ensure_ascii=False, indent=1))

    n_pick = sum(r["pick_grasped"] for r in rows)
    print(f"[merge] 共 {len(rows)} 组合：pick 成功 {n_pick}，浅轴候选 {len(cands)}")
    print("[merge] 候选（按轴偏升序，前 30）：")
    for r in cands[:30]:
        print(f"  ({r['point'][0]:+.3f},{r['point'][1]:+.3f}) ps={r['plan_seed']} "
              f"轴偏 {r['angle_deg']:5.2f}° pen_low_z={r['pen_low_z']:.3f} "
              f"axis={r['settle_axis']}")
    per_seed = {ps: sum(1 for r in cands if r["plan_seed"] == ps)
                for ps in sorted({r["plan_seed"] for r in rows})}
    print(f"[merge] 各 plan_seed 候选数: {per_seed}")

    # ---- 实验图：空间散点（每点最优轴偏）+ 轴偏直方图 + 各 seed 候选数 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    best = {}
    for r in rows:
        k = (r["point"][0], r["point"][1])
        if k not in best or r["angle_deg"] < best[k]["angle_deg"]:
            best[k] = r
    xs = [k[0] * 1000 for k in best]
    ys = [k[1] * 1000 for k in best]
    cs = [min(v["angle_deg"], 60.0) for v in best.values()]
    shallow_keys = {(r["point"][0], r["point"][1]) for r in cands}
    sc = axes[0].scatter(xs, ys, c=cs, cmap="RdYlGn_r", vmin=0, vmax=60, s=120)
    for (x, y), v in best.items():
        if (x, y) in shallow_keys:
            axes[0].annotate(f"{v['angle_deg']:.0f}°/s{v['plan_seed']}",
                             (x * 1000, y * 1000), textcoords="offset points",
                             xytext=(6, 6), fontsize=8, color="#1a7a1a")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].set_title(f"各桌面点最优轴偏（10 plan_seed 取最小）\n浅轴点 {len(shallow_keys)}/25")
    axes[0].grid(alpha=0.3)
    axes[0].set_aspect("equal")
    fig.colorbar(sc, ax=axes[0], label="轴偏角 (°)")

    angs = [r["angle_deg"] for r in rows]
    axes[1].hist(angs, bins=36, range=(0, 180), color="#4472a8", alpha=0.85)
    axes[1].axvline(10, color="r", ls="--", lw=1.5, label="浅轴阈值 10°")
    axes[1].set_xlabel("settle 笔轴与标准轴夹角 (°)")
    axes[1].set_ylabel("组合数")
    axes[1].set_title(f"轴偏分布（{len(rows)} 组合，候选 {len(cands)}）")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    seeds = sorted(per_seed)
    axes[2].bar([str(s) for s in seeds], [per_seed[s] for s in seeds], color="#4472a8")
    axes[2].set_xlabel("plan_seed")
    axes[2].set_ylabel("浅轴候选数")
    axes[2].set_title("各 plan_seed 浅轴产出")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("M4 n4 自然 pick 变体扫描：25 桌面点 × 10 plan_seed（settle 25 帧口径）")
    fig.tight_layout()
    fig.savefig(out_dir / "scan_pick.png", dpi=150)
    print(f"[merge] 图: {out_dir / 'scan_pick.png'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default="verify_out/scan_pick")
    ap.add_argument("--plan-seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()
    if args.plan_seeds is not None:
        global PLAN_SEEDS
        PLAN_SEEDS = list(args.plan_seeds)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        return merge(out_dir)
    return combo_rows(args.shard, args.shards, out_dir)


if __name__ == "__main__":
    sys.exit(main())
