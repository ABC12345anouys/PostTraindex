"""M3 残差 TD3：冻结 ACT（ensemble 0.01）为参考策略，raw 54-D 关节空间残差。

跑法（lerobot-env / mujoco 3.7.0，与 M2 同物理）:
  MUJOCO_GL=egl <lerobot-env>/bin/python policy/residual_td3.py \
      --cfg configs/residual_td3.yaml --minutes 8.5   # 分段前台跑，自动续训凑满 max_transitions

机制：每 C=chunk_exec 帧一个 RL 决策——actor(s) 输出有界残差 Δa（zero-init tanh），
C 帧内逐帧执行 ACT(ensemble)+Δa；转移为 chunk 级 (s,Δa,r,s',done)，γ=gamma_frame^C；
终局奖励：成功 r_success / 失败与超时 r_fail；TD3（双 critic、目标网络、延迟策略更新）。
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
from envs import BimanualDexConfig  # noqa: E402
from envs.bimanual_dex_env import BimanualDexEnv  # noqa: E402

CAMS = ("head", "left_wrist", "right_wrist")
S_DIM = 54 + 13
A_DIM = 54          # 残差维度，main 里按 residual_dims 切片重设
FULL_A_DIM = 54     # 全动作维（[左臂7/左手20/右臂7/右手20]）


# ---------- 网络 ----------
class Actor(nn.Module):
    def __init__(self, bound):
        super().__init__()
        self.bound = float(bound)
        self.net = nn.Sequential(
            nn.Linear(S_DIM, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, A_DIM), nn.Tanh())
        with torch.no_grad():  # zero-init 输出层：初始行为 == ACT 基线
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


# ---------- 回放缓冲（chunk 级转移） ----------
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


# ---------- 冻结 ACT（与 eval_act.py 同一套加载/观测构造） ----------
class ACTBase:
    def __init__(self, ckpt, ensemble, device):
        # 评测可复现：关 cudnn benchmark/TF32（数值抖动会翻转边界 seed，±11–22% 测量噪声）
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
        self.device = device
        self.policy = ACTPolicy.from_pretrained(ckpt)
        if ensemble >= 0:  # from_pretrained 后改 config 不建 ensembler，须手动补挂
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
        for c in CAMS:  # 与 train.py:438 一致：uint8 → CHW float32/255
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


# ---------- 环境交互 ----------
def make_env(seconds):
    cfg = BimanualDexConfig(max_episode_seconds=float(seconds))
    env = BimanualDexEnv(config=cfg, render_mode="rgb_array")
    return env, env


def run_episode(env, unwrapped, act, actor, rc, device, px, py, env_seed,
                rng, explore, buf=None):
    """跑一条 episode，返回 (转移数, 成功?, 帧数)。per-frame 决策与转移；explore 时 Δa 加噪声。"""
    orig_randomize = unwrapped._randomize_pen
    qflat = np.array([0.7071068, 0.0, 0.7071068, 0.0])

    def fixed_pen(d):
        orig_randomize(d)
        d.qpos[3:7] = unwrapped._quat_mul(np.array([1.0, 0.0, 0.0, 0.0]), qflat)

    unwrapped._randomize_pen = fixed_pen
    unwrapped.cfg.pen_xy = ((px, px), (py, py))
    obs, _ = env.reset(seed=env_seed)
    act.reset()
    bound = float(rc["residual_bound"])
    std = float(rc["expl_noise"]) * bound
    w = float(rc.get("shape_w", 0.0))
    clip = float(rc.get("shape_clip", 0.005))
    n_tr, frames, done, success = 0, 0, False, False
    z_prev = None
    while not done:
        s = s_vec(obs)
        with torch.inference_mode():
            da = actor(torch.as_tensor(s, device=device).unsqueeze(0))
        da = da.squeeze(0).cpu().numpy()
        if explore:
            da = np.clip(da + rng.normal(0, std, A_DIM), -bound, bound)
        da = da.astype(np.float32)
        full = np.zeros(FULL_A_DIM, np.float32)  # 残差只加在 residual_dims 切片，其余维归 ACT
        full[int(rc["_i0"]):int(rc["_i1"])] = da
        base = act.action(obs)
        obs, _, term, trunc, info = env.step_rad(base + full)
        frames += 1
        done = bool(term or trunc)
        z = float(info["pen_z"])
        r = 0.0 if z_prev is None else w * float(np.clip(z - z_prev, -clip, clip))
        z_prev = z
        if done:
            success = bool(info["success"])
            r += float(rc["r_success"]) if success else float(rc["r_fail"])
        if buf is not None:
            buf.add(s, da, r, s_vec(obs), float(done))
            n_tr += 1
    return n_tr, success, frames


@torch.inference_mode()
def evaluate(env, unwrapped, act, actor, rc, device, points, tag):
    """贪心评测（无噪声），与 eval_act.py 同协议：固定点 + seed 3000+i。"""
    results = []
    for i, (px, py) in enumerate(points):
        _, success, frames = run_episode(env, unwrapped, act, actor, rc, device,
                                         px, py, 3000 + i, None, explore=False)
        results.append({"x": px, "y": py, "success": success, "frames": frames})
        print(f"  [{tag} {i+1}/{len(points)}] ({px:+.3f},{py:+.3f}) "
              f"{'SUC' if success else 'FAIL'} frames={frames}", flush=True)
    ns = sum(r["success"] for r in results)
    return ns / len(results), results


# ---------- TD3 ----------
def td3_update(actor, actor_t, critic, critic_t, opt_a, opt_c, buf, rc, rng, device, it,
               update_actor=True):
    bs = min(int(rc["batch_size"]), buf.n)
    s, a, r, s2, d = buf.sample(bs, rng, device)
    gamma = float(rc["gamma_frame"])  # per-frame 转移（chunk 级 C=16 已证伪，登记偏离）
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
    # warm-up（update_actor=False）只更 critic；目标网络每步 Polyak
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
    ap.add_argument("--cfg", default="configs/residual_td3.yaml")
    ap.add_argument("--minutes", type=float, default=8.5, help="本段时间预算（到点优雅落盘退出）")
    ap.add_argument("--smoke", action="store_true", help="2 条链路验证 + 测速，不写正式 ckpt")
    ap.add_argument("--finalize", action="store_true",
                    help="跳过训练，加载 ckpt 做终评（wl9+rand20）并置 done")
    args = ap.parse_args()

    rc = yaml.safe_load((ROOT / args.cfg).read_text())
    global A_DIM
    i0, i1 = (int(x) for x in rc.get("residual_dims", [0, FULL_A_DIM]))
    A_DIM = i1 - i0
    rc["_i0"], rc["_i1"] = i0, i1
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
    points = [tuple(map(float, p)) for p in rc["points"]]
    if args.finalize:  # 直接终评：用 best_actor（若有），走段落落盘后把 done 写回
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
        rng_ep = np.random.default_rng(int(rc["seed"]) * 100003 + ep)  # 每条独立可复现
        bx, by = points[ep % len(points)]
        jit = float(rc.get("train_jitter", 0.0))  # ±jit 抖动采样（对齐 rand20 分布，提泛化）
        px = float(np.clip(bx + rng_ep.uniform(-jit, jit), -0.1, 0.1)) if jit > 0 else bx
        py = float(np.clip(by + rng_ep.uniform(-jit, jit), -0.1, 0.1)) if jit > 0 else by
        env_seed = int(rng_ep.integers(1 << 30))
        explore = bool(rng_ep.random() >= float(rc["greedy_prob"]))  # 部分条纯贪心采集
        n_tr, success, frames = run_episode(
            env, unwrapped, act, actor, rc, device, px, py, env_seed, rng_ep,
            explore=explore, buf=buf)
        ep += 1
        recent.append(bool(success))
        # 梯度更新（warm-up 期只更 critic：critic warm-up）
        warm = buf.n < int(rc["critic_warmup_transitions"])
        if buf.n >= 8:
            for _ in range(grad_steps):
                td3_update(actor, actor_t, critic, critic_t, opt_a, opt_c,
                           buf, rc, rng, device, upd, update_actor=not warm)
                upd += 1
        wr = sum(recent[-20:]) / len(recent[-20:])
        print(f"[ep {ep}] ({px:+.3f},{py:+.3f}) {'SUC' if success else 'FAIL'} "
              f"frames={frames} tr={buf.n} win20={wr:.0%} upd={upd} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        # 定期贪心评测
        if not args.smoke and ep % int(rc["eval_every_episodes"]) == 0:
            rate, res = evaluate(env, unwrapped, act, actor, rc, device, points, "wl9")
            if rate > best:  # 防后期发散：best actor 单独落盘
                best = rate
                stall = 0
                torch.save(actor.state_dict(), out / "best_actor.pt")
            else:
                stall += 1
                if stall >= 2 and (out / "best_actor.pt").exists():
                    # 高原回滚：恢复 best actor，actor_lr 减半（下限 1e-7）
                    actor.load_state_dict(torch.load(out / "best_actor.pt",
                                                     map_location=device, weights_only=True))
                    actor_t.load_state_dict(actor.state_dict())
                    for g in opt_a.param_groups:
                        g["lr"] = max(g["lr"] * 0.5, 1e-7)
                    stall = 0
                    print(f"== rollback: restore best={best:.0%}, "
                          f"actor_lr={opt_a.param_groups[0]['lr']:.1e} ==", flush=True)
            entry = {"episodes": ep, "transitions": buf.n, "wl9_rate": rate,
                     "best": best, "results": res}
            if ep % int(rc.get("eval_rand_every", 10 ** 9)) == 0:
                rng_r = np.random.default_rng(2026)
                pts20 = []
                for _ in range(20):
                    bxx, byy = points[int(rng_r.integers(len(points)))]
                    pts20.append((float(np.clip(bxx + rng_r.uniform(-0.015, 0.015), -0.1, 0.1)),
                                  float(np.clip(byy + rng_r.uniform(-0.015, 0.015), -0.1, 0.1))))
                r20, res20 = evaluate(env, unwrapped, act, actor, rc, device, pts20, "r20")
                entry["rand20_rate"] = r20
                print(f"== eval rand20 {int(r20*20)}/20 = {r20:.0%} ==", flush=True)
            with open(log_p, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"== eval wl9 {int(rate*len(points))}/{len(points)} = {rate:.0%} "
                  f"(best {best:.0%}) ==", flush=True)
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                        "actor_t": actor_t.state_dict(), "critic_t": critic_t.state_dict(),
                        "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
                        "episodes": ep, "updates": upd, "best": best, "done": False}, ckpt_p)
            if time.time() - t0 > budget:
                break

    # 段落落盘
    if not args.smoke:
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                    "actor_t": actor_t.state_dict(), "critic_t": critic_t.state_dict(),
                    "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
                    "episodes": ep, "updates": upd, "best": best,
                    "done": force_final or buf.n >= int(rc["max_transitions"])}, ckpt_p)
        buf.save(buf_p)
        print(f"saved: episodes={ep} transitions={buf.n} -> {ckpt_p}", flush=True)

    # 预算凑满或手动 finalize：终评 whitelist9 + rand20
    if force_final or buf.n >= int(rc["max_transitions"]):
        rate9, res9 = evaluate(env, unwrapped, act, actor, rc, device, points, "final-wl9")
        rng_r = np.random.default_rng(2026)
        pts20 = []
        for _ in range(20):
            bx, by = points[int(rng_r.integers(len(points)))]
            pts20.append((float(np.clip(bx + rng_r.uniform(-0.015, 0.015), -0.1, 0.1)),
                          float(np.clip(by + rng_r.uniform(-0.015, 0.015), -0.1, 0.1))))
        rate20, res20 = evaluate(env, unwrapped, act, actor, rc, device, pts20, "final-rand20")
        final = {"transitions": buf.n, "episodes": ep,
                 "whitelist9": {"rate": rate9, "results": res9},
                 "rand20": {"rate": rate20, "results": res20}}
        (ROOT / "verify_out" / "rl_m3_final.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=1))
        print(f"== FINAL wl9 {rate9:.0%} rand20 {rate20:.0%} ==", flush=True)

    env.close()


if __name__ == "__main__":
    main()
