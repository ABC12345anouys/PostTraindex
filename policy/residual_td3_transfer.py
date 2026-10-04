"""M4 n6 transfer 残差 TD3：冻结 ACT（ensemble 0.01）为参考策略，raw 关节空间残差。

跑法（lerobot-env / mujoco 3.7.0，与录数/n5 评测同物理）:
  MUJOCO_GL=egl <lerobot-env>/bin/python policy/residual_td3_transfer.py \
      --cfg configs/residual_td3_transfer.yaml --minutes 8.5   # 分段前台跑，自动续训

与 M3 母版差异（transfer 任务适配）:
  - episode = 标称 reset(yaw0) + 右手 expert pick(plan_seed) + settle 25 帧（不喂策略、
    与录数/n5 评测同协议）→ 策略+残差从 settle 末接管，最长 max_frames 帧；
  - 成功 = 左手触笔 & 右手离笔 & 笔悬空 & 线速度<阈值 连续 hold_frames 帧（proto 口径）；
  - 早停：笔最低点 < TABLE_TOP_Z+0.02 持续 10 帧判失败（省墙钟）；
  - 变体 = 成功池去重 (point, plan_seed) 循环采样；env_seed 不改变物理（yaw0+定点），
    探索多样性来自行为噪声；
  - residual_dims 支持多段切片（默认左手指+右手指 40-D：交接关键在双手指，
    臂归 ACT；M3 经验：小探索面）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from envs import BimanualDexConfig  # noqa: E402
from envs.bimanual_dex_env import BimanualDexEnv, TABLE_TOP_Z  # noqa: E402
import proto_transfer as PT  # noqa: E402
from teleop.scripted_expert import ScriptedExpert  # noqa: E402

CAMS = ("head", "left_wrist", "right_wrist")
S_DIM = 54 + 13
FULL_A_DIM = 54     # [左臂7/左手20/右臂7/右手20]
A_DIM = 54          # 残差维度，main 里按 residual_slices 重设


# ---------- 网络（与 M3 同构） ----------
class Actor(nn.Module):
    def __init__(self, bound):
        super().__init__()
        self.bound = float(bound)
        self.net = nn.Sequential(
            nn.Linear(S_DIM, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, A_DIM), nn.Tanh())
        with torch.no_grad():  # zero-init：初始行为 == ACT 基线
            self.net[-2].weight.zero_()
            self.net[-2].bias.zero_()

    def forward(self, s):
        return self.bound * self.net(s)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        def q():
            return nn.Sequential(
                nn.Linear(S_DIM + A_DIM, 1024), nn.LayerNorm(1024), nn.ReLU(),
                nn.Linear(1024, 256), nn.LayerNorm(256), nn.ReLU(),
                nn.Linear(256, 1))
        self.q1, self.q2 = q(), q()

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x), self.q2(x)


class Buffer:
    def __init__(self, cap):
        self.cap = int(cap)
        self.s = np.zeros((cap, S_DIM), np.float32)
        self.a = np.zeros((cap, A_DIM), np.float32)
        self.r = np.zeros((cap, 1), np.float32)
        self.s2 = np.zeros((cap, S_DIM), np.float32)
        self.d = np.zeros((cap, 1), np.float32)
        self.n, self.ptr = 0, 0

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i, 0], self.s2[i], self.d[i, 0] = s, a, r, s2, d
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs, rng, device):
        idx = rng.integers(0, self.n, bs)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return to(self.s), to(self.a), to(self.r), to(self.s2), to(self.d)

    def save(self, path):
        np.savez(path, s=self.s[:self.n], a=self.a[:self.n], r=self.r[:self.n],
                 s2=self.s2[:self.n], d=self.d[:self.n], ptr=self.ptr)

    def load(self, path):
        z = np.load(path)
        for k in ("s", "a", "r", "s2", "d"):
            getattr(self, k)[:len(z[k])] = z[k]
        self.n, self.ptr = len(z["s"]), int(z["ptr"])


# ---------- 冻结 ACT ----------
class ACTBase:
    def __init__(self, ckpt, ensemble, device):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
        self.device = device
        self.policy = ACTPolicy.from_pretrained(ckpt)
        if ensemble >= 0:
            self.policy.config.temporal_ensemble_coeff = ensemble
            self.policy.temporal_ensembler = ACTTemporalEnsembler(
                ensemble, self.policy.config.chunk_size)
        self.policy.config.device = device
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": device}})

    def reset(self):
        self.policy.reset()

    @torch.inference_mode()
    def action(self, obs):
        ob = {}
        for c in CAMS:
            ob[f"observation.images.{c}"] = torch.from_numpy(
                obs["images"][c]).permute(2, 0, 1).to(torch.float32) / 255.0
        ob["observation.state"] = torch.from_numpy(np.asarray(obs["state"], np.float32))
        ob["observation.object"] = torch.from_numpy(np.asarray(obs["object"], np.float32))
        out = self.post(self.policy.select_action(self.pre(ob)))
        act = out["action"] if isinstance(out, dict) else out
        return act.squeeze(0).cpu().numpy().astype(np.float64)


def s_vec(obs):
    return np.concatenate([np.asarray(obs["state"], np.float32),
                           np.asarray(obs["object"], np.float32)])


# ---------- transfer 前缀（reset+pick+settle，与录数/n5 评测同协议） ----------
class Prefix:
    def __init__(self, env, unwrapped, tcfg_path):
        import mujoco
        cfg = PT.load_cfg(tcfg_path)
        self.cfg = cfg
        tc = cfg["transfer"]
        self.poses = {k: np.asarray(v, dtype=np.float64)
                      for k, v in cfg["hand_poses"].items()}
        self.n_settle = int(tc.get("settle_frames", 25))
        self.arm_damp = float(tc.get("arm_damping", 10.0))
        self.hold_need = int(tc["hold_frames"])
        PT.build_geom_sides(unwrapped)
        self.damp0 = unwrapped.model.dof_damping.copy()
        self._mujoco = mujoco

        orig_randomize = unwrapped._randomize_pen
        qflat = np.array([0.7071068, 0.0, 0.7071068, 0.0])

        def fixed_pen(d):
            orig_randomize(d)
            d.qpos[3:7] = unwrapped._quat_mul(np.array([1.0, 0.0, 0.0, 0.0]), qflat)
        unwrapped._randomize_pen = fixed_pen

        cfg_l = dict(cfg)
        cfg_l["allowed_hands"] = ["left"]
        expert_meta = ScriptedExpert(unwrapped, cfg_l)
        self.l_q9 = unwrapped.model.key_qpos[expert_meta._key_home][
            expert_meta.iks["left"].qadr].copy()
        self.l_f18 = self.poses["open"][2:20].copy()
        self.r_f18 = self.poses["pinch3"][2:20].copy()

    def set_damping(self, unwrapped, c):
        mujoco = self._mujoco
        for side in ("l", "r"):
            for ji in range(1, 8):
                jid = mujoco.mj_name2id(unwrapped.model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{side}_joint{ji}")
                unwrapped.model.dof_damping[unwrapped.model.jnt_dofadr[jid]] = c

    def run(self, env, unwrapped, px, py, plan_seed):
        """返回 settle 末 obs（碰撞掩码已切 cross，与录数一致）。"""
        unwrapped.model.dof_damping[:] = self.damp0
        unwrapped.cfg.pen_xy = ((px, px), (py, py))
        obs, _ = env.reset(seed=1002)
        cfg_r = dict(self.cfg)
        cfg_r["allowed_hands"] = ["right"]
        cfg_r["plan_seed"] = int(plan_seed)
        expert_r = ScriptedExpert(unwrapped, cfg_r)
        expert_r.reset(obs)
        while expert_r.si < 4:
            obs, _, _, _, _ = env.step_rad(expert_r.act(obs))
        self.set_damping(unwrapped, self.arm_damp)
        r_q9 = unwrapped.data.qpos[expert_r.ik.qadr].copy()
        for _ in range(self.n_settle):
            obs, _, _, _, _ = env.step_rad(PT.blend_action(
                self.l_q9, self.l_f18, r_q9, self.r_f18))
        PT.set_collision_mask("cross")
        return obs

    def restore(self):
        PT.set_collision_mask(None)


# ---------- 环境交互 ----------
def make_env(seconds):
    # hold_frames=10**9：禁用 env 级 any-hand 成功早停（同 eval_act_transfer）
    cfg = BimanualDexConfig(max_episode_seconds=float(seconds),
                            pen_xy=((0.0, 0.0), (0.0, 0.0)),
                            hold_frames=10 ** 9)
    env = BimanualDexEnv(config=cfg, render_mode="rgb_array")
    return env, env


def scatter(da, slices):
    """多段残差撒回 54-D 全动作。"""
    full = np.zeros(FULL_A_DIM, np.float32)
    off = 0
    for i0, i1 in slices:
        full[i0:i1] = da[off:off + (i1 - i0)]
        off += i1 - i0
    return full


def run_episode(env, unwrapped, pfx, act, actor, rc, device, px, py, plan_seed,
                rng, explore, buf=None):
    """一条 episode：前缀 → ACT+残差接管。返回 (转移数, 成功?, 策略帧数)。"""
    obs = pfx.run(env, unwrapped, px, py, plan_seed)
    act.reset()
    bound = float(rc["residual_bound"])
    std = float(rc["expl_noise"]) * bound
    w = float(rc.get("shape_w", 0.0))
    clip = float(rc.get("shape_clip", 0.005))
    slices = rc["_slices"]
    max_frames = int(rc["max_frames"])
    hold_need = pfx.hold_need
    drop_z = TABLE_TOP_Z + float(rc.get("drop_clearance", 0.02))
    n_tr, frames, done, success = 0, 0, False, False
    hold_cnt, drop_cnt = 0, 0
    z_prev = None
    while not done and frames < max_frames:
        s = s_vec(obs)
        with torch.inference_mode():
            da = actor(torch.as_tensor(s, device=device).unsqueeze(0))
        da = da.squeeze(0).cpu().numpy()
        if explore:
            da = np.clip(da + rng.normal(0, std, A_DIM), -bound, bound)
        da = da.astype(np.float32)
        base = act.action(obs)
        obs, _, term, trunc, info = env.step_rad(base + scatter(da, slices))
        frames += 1
        z = float(info["pen_z"])
        r = 0.0 if z_prev is None else w * float(np.clip(z - z_prev, -clip, clip))
        z_prev = z
        lh = PT.hand_pen_contact(unwrapped, "lh")
        rh = PT.hand_pen_contact(unwrapped, "rh")
        lifted = unwrapped._pen_lowest_z() > TABLE_TOP_Z + unwrapped.cfg.lift_clearance
        stable = np.linalg.norm(unwrapped.data.qvel[0:3]) < unwrapped.cfg.max_lin_vel
        hold_cnt = hold_cnt + 1 if (lh and not rh and lifted and stable) else 0
        drop_cnt = drop_cnt + 1 if unwrapped._pen_lowest_z() < drop_z else 0
        if hold_cnt >= hold_need:
            done, success = True, True
        elif drop_cnt >= 10 or term or trunc:
            done = True
        elif frames >= max_frames:
            done = True
        if done:
            r += float(rc["r_success"]) if success else float(rc["r_fail"])
        if buf is not None:
            buf.add(s, da, r, s_vec(obs), float(done))
            n_tr += 1
    pfx.restore()
    return n_tr, success, frames


@torch.inference_mode()
def evaluate(env, unwrapped, pfx, act, actor, rc, device, variants, tag):
    """贪心评测（无噪声）：逐变体 (point, plan_seed)。"""
    results = []
    for i, v in enumerate(variants):
        _, success, frames = run_episode(
            env, unwrapped, pfx, act, actor, rc, device,
            v["point"][0], v["point"][1], v["plan_seed"], None, explore=False)
        results.append({**v, "success": success, "frames": frames})
        print(f"  [{tag} {i+1}/{len(variants)}] ({v['point'][0]:+.3f},{v['point'][1]:+.3f})"
              f" ps={v['plan_seed']} {'SUC' if success else 'FAIL'} frames={frames}",
              flush=True)
    ns = sum(r["success"] for r in results)
    return ns / len(results), results


# ---------- TD3（与 M3 同构） ----------
def td3_update(actor, actor_t, critic, critic_t, opt_a, opt_c, buf, rc, rng, device, it,
               update_actor=True):
    bs = min(int(rc["batch_size"]), buf.n)
    s, a, r, s2, d = buf.sample(bs, rng, device)
    gamma = float(rc["gamma_frame"])
    bound = float(rc["residual_bound"])
    with torch.no_grad():
        noise = (torch.randn_like(a) * float(rc["target_noise"]) * bound
                 ).clamp(-float(rc["target_noise_clip"]) * bound,
                         float(rc["target_noise_clip"]) * bound)
        a2 = (actor_t(s2) + noise).clamp(-bound, bound)
        q1t, q2t = critic_t(s2, a2)
        y = r + gamma * (1 - d) * torch.min(q1t, q2t)
    q1, q2 = critic(s, a)
    loss_c = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)
    opt_c.zero_grad()
    loss_c.backward()
    opt_c.step()
    loss_a_val = 0.0
    if update_actor and it % int(rc["policy_delay"]) == 0:
        pi = actor(s)
        l2 = float(rc.get("actor_l2", 0.0)) * (pi / bound).pow(2).mean()
        loss_a = -critic.q1(torch.cat([s, pi], -1)).mean() + l2
        opt_a.zero_grad()
        loss_a.backward()
        opt_a.step()
        loss_a_val = float(loss_a)
    tau = float(rc["tau"])
    with torch.no_grad():
        if update_actor:
            for p, pt in zip(actor.parameters(), actor_t.parameters()):
                pt.mul_(1 - tau).add_(tau * p)
        for p, pt in zip(critic.parameters(), critic_t.parameters()):
            pt.mul_(1 - tau).add_(tau * p)
    return loss_c.item(), loss_a_val


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/residual_td3_transfer.yaml")
    ap.add_argument("--minutes", type=float, default=8.5, help="本段时间预算（到点优雅落盘退出）")
    ap.add_argument("--smoke", action="store_true", help="2 条链路验证 + 测速，不写正式 ckpt")
    ap.add_argument("--finalize", action="store_true",
                    help="跳过训练，加载 ckpt 做终评并置 done")
    args = ap.parse_args()

    rc = yaml.safe_load((ROOT / args.cfg).read_text())
    global A_DIM
    slices = [(int(a), int(b)) for a, b in rc.get("residual_dims", [[0, FULL_A_DIM]])]
    A_DIM = sum(b - a for a, b in slices)
    rc["_slices"] = slices
    print(f"残差切片 {slices} -> A_DIM={A_DIM}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / rc["out_dir"]
    if args.smoke:
        out = out.parent / (out.name + "_smoke")
    out.mkdir(parents=True, exist_ok=True)
    ckpt_p, buf_p, log_p = out / "ckpt.pt", out / "buffer.npz", out / "log.jsonl"

    torch.manual_seed(int(rc["seed"]))
    actor = Actor(rc["residual_bound"]).to(device)
    actor_t = Actor(rc["residual_bound"]).to(device)
    actor_t.load_state_dict(actor.state_dict())
    critic = Critic().to(device)
    critic_t = Critic().to(device)
    critic_t.load_state_dict(critic.state_dict())
    for p in list(actor_t.parameters()) + list(critic_t.parameters()):
        p.requires_grad_(False)
    opt_a = torch.optim.Adam(actor.parameters(), lr=float(rc["actor_lr"]))
    opt_c = torch.optim.Adam(critic.parameters(), lr=float(rc["critic_lr"]))
    buf = Buffer(rc["buffer_size"])

    ep0, upd, best, done_flag = 0, 0, -1.0, False
    if ckpt_p.exists() and not args.smoke:
        z = torch.load(ckpt_p, map_location=device, weights_only=False)
        actor.load_state_dict(z["actor"])
        critic.load_state_dict(z["critic"])
        actor_t.load_state_dict(z["actor_t"])
        critic_t.load_state_dict(z["critic_t"])
        opt_a.load_state_dict(z["opt_a"])
        opt_c.load_state_dict(z["opt_c"])
        ep0, upd, best, done_flag = z["episodes"], z["updates"], z["best"], z.get("done", False)
        if buf_p.exists():
            buf.load(buf_p)
        print(f"resume: episodes={ep0} transitions={buf.n} updates={upd} best={best:.2f}",
              flush=True)
    if done_flag:
        print("已达 max_transitions 并完成终评，无需再跑。")
        return

    act = ACTBase(ROOT / rc["act_ckpt"], float(rc["ensemble"]), device)
    env, unwrapped = make_env(rc["episode_seconds"])
    pfx = Prefix(env, unwrapped, ROOT / "configs/transfer_marker.yaml")
    # 变体 = 成功池去重 (point, plan_seed)
    pool = json.loads((ROOT / rc["pool"]).read_text())
    variants, seen = [], set()
    for r in pool:
        k = (r["point"][0], r["point"][1], r["plan_seed"])
        if k not in seen:
            seen.add(k)
            variants.append({"point": r["point"], "plan_seed": r["plan_seed"]})
    print(f"变体 {len(variants)} 个", flush=True)

    if args.finalize:
        bp = out / "best_actor.pt"
        if bp.exists():
            actor.load_state_dict(torch.load(bp, map_location=device, weights_only=True))
            print(f"finalize with best_actor (best={best:.0%})", flush=True)
        force_final = True
        budget = 0.0
    else:
        force_final = False
        budget = 300.0 if args.smoke else args.minutes * 60.0
    rng = np.random.default_rng(int(rc["seed"]))
    t0 = time.time()
    max_eps = 2 if args.smoke else 10 ** 9
    grad_steps = 20 if args.smoke else int(rc["grad_steps_per_episode"])

    ep = ep0
    recent = []
    stall = 0
    while ep < max_eps and buf.n < int(rc["max_transitions"]):
        if time.time() - t0 > budget:
            break
        rng_ep = np.random.default_rng(int(rc["seed"]) * 100003 + ep)
        v = variants[ep % len(variants)]
        px, py = float(v["point"][0]), float(v["point"][1])
        explore = bool(rng_ep.random() >= float(rc["greedy_prob"]))
        n_tr, success, frames = run_episode(
            env, unwrapped, pfx, act, actor, rc, device, px, py, v["plan_seed"],
            rng_ep, explore=explore, buf=buf)
        ep += 1
        recent.append(bool(success))
        warm = buf.n < int(rc["critic_warmup_transitions"])
        if buf.n >= 8:
            for _ in range(grad_steps):
                td3_update(actor, actor_t, critic, critic_t, opt_a, opt_c,
                           buf, rc, rng, device, upd, update_actor=not warm)
                upd += 1
        wr = sum(recent[-20:]) / len(recent[-20:])
        print(f"[ep {ep}] ({px:+.3f},{py:+.3f}) ps={v['plan_seed']} "
              f"{'SUC' if success else 'FAIL'} frames={frames} tr={buf.n} "
              f"win20={wr:.0%} upd={upd} elapsed={time.time()-t0:.0f}s", flush=True)
        if not args.smoke and ep % int(rc["eval_every_episodes"]) == 0:
            rate, res = evaluate(env, unwrapped, pfx, act, actor, rc, device,
                                 variants, "pool")
            if rate > best:
                best = rate
                stall = 0
                torch.save(actor.state_dict(), out / "best_actor.pt")
            else:
                stall += 1
                if stall >= 2 and (out / "best_actor.pt").exists():
                    actor.load_state_dict(torch.load(out / "best_actor.pt",
                                                     map_location=device, weights_only=True))
                    actor_t.load_state_dict(actor.state_dict())
                    for g in opt_a.param_groups:
                        g["lr"] = max(g["lr"] * 0.5, 1e-7)
                    stall = 0
                    print(f"== rollback: restore best={best:.0%}, "
                          f"actor_lr={opt_a.param_groups[0]['lr']:.1e} ==", flush=True)
            entry = {"episodes": ep, "transitions": buf.n, "pool_rate": rate,
                     "best": best, "results": res}
            with open(log_p, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"== eval pool {int(rate*len(variants))}/{len(variants)} = {rate:.0%} "
                  f"(best {best:.0%}) ==", flush=True)
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                        "actor_t": actor_t.state_dict(), "critic_t": critic_t.state_dict(),
                        "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
                        "episodes": ep, "updates": upd, "best": best, "done": False}, ckpt_p)
            if time.time() - t0 > budget:
                break

    if not args.smoke:
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                    "actor_t": actor_t.state_dict(), "critic_t": critic_t.state_dict(),
                    "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
                    "episodes": ep, "updates": upd, "best": best,
                    "done": force_final or buf.n >= int(rc["max_transitions"])}, ckpt_p)
        buf.save(buf_p)
        print(f"saved: episodes={ep} transitions={buf.n} -> {ckpt_p}", flush=True)

    if force_final or buf.n >= int(rc["max_transitions"]):
        rate, res = evaluate(env, unwrapped, pfx, act, actor, rc, device,
                             variants, "final-pool")
        final = {"transitions": buf.n, "episodes": ep,
                 "pool": {"rate": rate, "results": res}}
        (ROOT / "verify_out" / "rl_m4_final.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=1))
        print(f"== FINAL pool {rate:.0%} ==", flush=True)

    env.close()


if __name__ == "__main__":
    main()
