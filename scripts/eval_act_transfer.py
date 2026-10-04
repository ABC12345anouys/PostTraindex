"""M4 n5 transfer ACT 闭环评测（lerobot-env / mujoco 3.7.0）。

episode = 标称 reset+expert pick+settle（与录数同协议，不喂策略）→ 策略从 settle 末
接管到 hold 判定。成功 = 左手触笔 & 右手离笔 & 笔悬空 & 线速度<阈值 连续 10 帧
（与 proto_transfer hold 口径一致）。评测变体 = 成功池去重 (point, plan_seed)。

用法:
  MUJOCO_GL=egl <lerobot-env>/bin/python scripts/eval_act_transfer.py \
      --ckpt checkpoints/act_m4/checkpoints/020000/pretrained_model --tag act_m4_20k
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import proto_transfer as PT  # noqa: E402
from envs import BimanualDexConfig  # noqa: E402
from envs.bimanual_dex_env import BimanualDexEnv, TABLE_TOP_Z  # noqa: E402
from teleop.scripted_expert import ScriptedExpert  # noqa: E402

CAMS = ("head", "left_wrist", "right_wrist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="act_m4")
    ap.add_argument("--pool", default="verify_out/transfer_success_pool.json")
    ap.add_argument("--max-frames", type=int, default=200, help="策略接管最长帧数")
    ap.add_argument("--ensemble", type=float, default=0.01)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(args.ckpt)
    if args.ensemble >= 0:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.config.temporal_ensemble_coeff = args.ensemble
        policy.temporal_ensembler = ACTTemporalEnsembler(
            args.ensemble, policy.config.chunk_size)
    policy.config.device = device
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": device}})

    # 物理变体 = 池中去重 (point, plan_seed)
    pool = json.loads((ROOT / args.pool).read_text())
    variants, seen = [], set()
    for r in pool:
        k = (r["point"][0], r["point"][1], r["plan_seed"])
        if k not in seen:
            seen.add(k)
            variants.append({"point": r["point"], "plan_seed": r["plan_seed"],
                             "angle_deg": r.get("angle_deg")})

    cfg = PT.load_cfg(ROOT / "configs/transfer_marker.yaml")
    tc = cfg["transfer"]
    poses = {k: np.asarray(v, dtype=np.float64) for k, v in cfg["hand_poses"].items()}
    n_settle = int(tc.get("settle_frames", 25))
    arm_damp = float(tc.get("arm_damping", 10.0))
    hold_need = int(tc["hold_frames"])
    hz = float(cfg.get("control_hz", 20))

    # hold_frames=10**9：禁用 env 级 any-hand 成功早停（实测右手停滞持笔 10 帧即
    # term=True 截断 episode，掩盖迟到交接）；transfer 成功由 lh&!rh 严格口径判定
    env_cfg = BimanualDexConfig(max_episode_seconds=25,
                                pen_xy=((0.0, 0.0), (0.0, 0.0)),
                                hold_frames=10 ** 9)
    env = BimanualDexEnv(config=env_cfg, render_mode="rgb_array")
    unwrapped = env
    PT.build_geom_sides(unwrapped)
    import mujoco
    damp0 = unwrapped.model.dof_damping.copy()

    def set_damping(c):
        for _side in ("l", "r"):
            for _ji in range(1, 8):
                _jid = mujoco.mj_name2id(unwrapped.model, mujoco.mjtObj.mjOBJ_JOINT,
                                         f"{_side}_joint{_ji}")
                unwrapped.model.dof_damping[unwrapped.model.jnt_dofadr[_jid]] = c

    orig_randomize = unwrapped._randomize_pen

    def fixed_pen(d):
        orig_randomize(d)
        qflat = np.array([0.7071068, 0.0, 0.7071068, 0.0])
        d.qpos[3:7] = unwrapped._quat_mul(np.array([1.0, 0.0, 0.0, 0.0]), qflat)

    unwrapped._randomize_pen = fixed_pen

    # 左手 home/张开 静态保持位（pick+settle 段与录数一致）
    cfg_l = dict(cfg)
    cfg_l["allowed_hands"] = ["left"]
    expert_meta = ScriptedExpert(unwrapped, cfg_l)
    l_q9 = unwrapped.model.key_qpos[expert_meta._key_home][expert_meta.iks["left"].qadr].copy()
    l_f18 = poses["open"][2:20].copy()
    r_f18 = poses["pinch3"][2:20].copy()

    out_dir = ROOT / "verify_out" / f"eval_transfer_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, v in enumerate(variants):
        px, py = v["point"]
        unwrapped.model.dof_damping[:] = damp0
        env_cfg.pen_xy = ((px, px), (py, py))
        obs, _ = env.reset(seed=1002)

        # ---- expert pick + settle（与录数同协议）----
        cfg_r = dict(cfg)
        cfg_r["allowed_hands"] = ["right"]
        cfg_r["plan_seed"] = int(v["plan_seed"])
        expert_r = ScriptedExpert(unwrapped, cfg_r)
        expert_r.reset(obs)
        info = {}
        while expert_r.si < 4:
            obs, _, term, trunc, info = env.step_rad(expert_r.act(obs))
        set_damping(arm_damp)
        r_q9 = unwrapped.data.qpos[expert_r.ik.qadr].copy()
        for _ in range(n_settle):
            obs, _, _, _, info = env.step_rad(PT.blend_action(l_q9, l_f18, r_q9, r_f18))
        PT.set_collision_mask("cross")   # transfer 段掩码口径与录数一致

        # ---- 策略接管 ----
        policy.reset()
        hold_cnt, pen_z_min, frames = 0, 9.9, 0
        shots = {}
        for k in range(int(args.max_frames)):
            ob = {}
            for c in CAMS:
                ob[f"observation.images.{c}"] = torch.from_numpy(
                    obs["images"][c]).permute(2, 0, 1).to(torch.float32) / 255.0
            ob["observation.state"] = torch.from_numpy(
                np.asarray(obs["state"], np.float32))
            ob["observation.object"] = torch.from_numpy(
                np.asarray(obs["object"], np.float32))
            with torch.inference_mode():
                out = post(policy.select_action(pre(ob)))
            act = out["action"] if isinstance(out, dict) else out
            a = act.squeeze(0).cpu().numpy().astype(np.float64)
            obs, _, term, trunc, info = env.step_rad(a)
            frames += 1
            pen_z_min = min(pen_z_min, float(info["pen_z"]))
            if k in (0, int(args.max_frames) // 2):
                shots[k] = obs["images"]["head"].copy()
            lh = PT.hand_pen_contact(unwrapped, "lh")
            rh = PT.hand_pen_contact(unwrapped, "rh")
            lifted = unwrapped._pen_lowest_z() > TABLE_TOP_Z + unwrapped.cfg.lift_clearance
            stable = np.linalg.norm(unwrapped.data.qvel[0:3]) < unwrapped.cfg.max_lin_vel
            hold_cnt = hold_cnt + 1 if (lh and not rh and lifted and stable) else 0
            if hold_cnt >= hold_need:
                break
            if term or trunc:
                break
        shots["end"] = obs["images"]["head"].copy()
        PT.set_collision_mask(None)
        ok = hold_cnt >= hold_need
        results.append({**v, "success": ok, "frames": frames,
                        "hold": hold_cnt, "pen_z_min": round(pen_z_min, 4)})
        from PIL import Image
        for tag, img in shots.items():
            Image.fromarray(img).save(out_dir / f"v{i:02d}_{tag}.png")
        print(f"[{i + 1:2d}/{len(variants)}] ({px:+.3f},{py:+.3f}) ps={v['plan_seed']} "
              f"{'SUC' if ok else 'FAIL'} frames={frames} hold={hold_cnt} "
              f"pen_z_min={pen_z_min:.3f}", flush=True)

    env.close()
    n = len(results)
    ns = sum(r["success"] for r in results)
    print(f"\n== transfer {args.tag}: {ns}/{n} = {ns / n:.0%} ==")
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    # ---- 成功率图（逐变体 + 总体）----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    labels = [f"({r['point'][0]*1000:+.0f},{r['point'][1]*1000:+.0f})\nps{r['plan_seed']}"
              for r in results]
    vals = [1 if r["success"] else 0 for r in results]
    fig, ax = plt.subplots(figsize=(max(7, n * 0.7), 4))
    ax.bar(range(n), vals, color=["#2ca02c" if v else "#c0392b" for v in vals])
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1])
    ax.set_title(f"M4 n5 transfer ACT 闭环成功率 {ns}/{n} = {ns / n:.0%}（{args.tag}）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "success_rate.png", dpi=150)
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
