"""M4 原型：Transfer between Hands（马克笔右手 → 左手交接）。

流程（中心点固定笔位，一次交接尝试）：
  0. 复用 ScriptedExpert 完成 pick（approach/descend/close/lift，右手 mode1 侧捏）
  1. gather：左臂移到笔的沿轴对侧承接位（pre_pinch 张开），右手死握
  2. catch ：左手 pre_pinch → pinch3 合拢，右手保持
  3. release：右手 pinch3 → open 并沿 +笔轴 撤离，左手保持
  4. hold  ：左手独握判定（左触笔 + 右离笔 + 笔悬空 3cm + 线速度 <0.2，连续 N 帧）

任务参数全部在 configs/transfer_marker.yaml（transfer 段），脚本不含任务硬编码。

用法:
  MUJOCO_GL=egl python3 scripts/proto_transfer.py [--x 0] [--y 0] [--yaw 0] [--seed 0]
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import BimanualDexConfig  # noqa: E402
from envs.bimanual_dex_env import BimanualDexEnv, TABLE_TOP_Z  # noqa: E402
from teleop.ik_solver import quat_to_R  # noqa: E402
from teleop.scripted_expert import ScriptedExpert, _min_jerk  # noqa: E402

_UP = np.array([0.0, 0.0, 1.0])


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_cfg(path: Path):
    cfg = yaml.safe_load(path.read_text())
    base = cfg.pop("base_cfg", None)
    if base:
        base_path = ROOT / base
        cfg = deep_merge(yaml.safe_load(base_path.read_text()), cfg)
    return cfg


def pen_axis(obs):
    q = np.asarray(obs["object"][3:7], dtype=np.float64)
    a = quat_to_R(q) @ np.array([0.0, 0.0, 1.0])
    return a / np.linalg.norm(a)


def grasp_frame_safe(a, mode, tilt):
    """_grasp_frame 的安全版：|a·UP|<0.7 时与原实现逐元素一致（保 seed4 口径）；
    近垂直笔轴时原式 y_col=cross(-UP,a) 退化且 R 非正交（63.7° 残差根因），
    换水平参考系构造正交基底。tilt 为 frame_tilt(rad)，绕 x（笔轴）施加。"""
    s = np.array([a[1], -a[0], 0.0])
    s = s / (np.linalg.norm(s) + 1e-9)
    x_col = (a, -a, s, -s)[mode]
    if abs(float(np.dot(a, _UP))) < 0.7:
        y_col = np.cross(-_UP, x_col)
        R0 = np.column_stack([x_col, y_col, -_UP])
    else:
        ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        y_col = np.cross(ref, x_col)
        y_col = y_col / np.linalg.norm(y_col)
        z_col = np.cross(x_col, y_col)
        R0 = np.column_stack([x_col, y_col, z_col])
    if tilt:
        ct, st = np.cos(tilt), np.sin(tilt)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
        R0 = R0 @ Rx
    return R0


def rot_about(axis, ang):
    """Rodrigues：绕单位轴 axis 转 ang 弧度的旋转矩阵。"""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) * np.cos(ang) + np.sin(ang) * K \
        + (1.0 - np.cos(ang)) * np.outer(axis, axis)


def hand_pen_contact(env, prefix):
    """prefix='lh'/'rh'：该手任一 geom 与笔接触。"""
    m, d = env.model, env.data
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom(c.geom1).bodyid) or ""
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom(c.geom2).bodyid) or ""
        if (b1 == "pen" and b2.startswith(prefix + "_")) or \
           (b2 == "pen" and b1.startswith(prefix + "_")):
            return True
    return False


import mujoco

LEFT_PREFIXES = ("lh_", "l_link")
RIGHT_PREFIXES = ("rh_", "r_link")

_mask_mode = None          # None / "cross"（只禁跨侧）/ "self"（手/臂 geom 全部禁碰，标定用）
_geom_sides = None         # 每个 geom 的侧别数组（env 创建后构建）
_geom_finger = None        # 每个 geom 是否属于手指段
_geom_islink = None        # 每个 geom 是否属于纯臂杆 l_link*/r_link*


NONFINGER_BODIES = (
    "l_link", "r_link", "lh_forearm", "rh_forearm", "lh_wrist", "rh_wrist",
    "lh_palm", "rh_palm", "lh_thbase", "rh_thbase", "lh_thhub", "rh_thhub",
)


def build_geom_sides(env):
    """预计算 geom 分类，供 contactfilter 高速查询。
    side: 0=左臂侧 1=右臂侧 2=笔 -1=其他（桌/世界）；finger: 是否手指段 geom。"""
    global _geom_sides, _geom_finger, _geom_islink
    m = env.model
    out = np.full(m.ngeom, -1, dtype=np.int8)
    fin = np.zeros(m.ngeom, dtype=bool)
    lin = np.zeros(m.ngeom, dtype=bool)
    for gi in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom(gi).bodyid)) or ""
        if bn.startswith(LEFT_PREFIXES):
            out[gi] = 0
        elif bn.startswith(RIGHT_PREFIXES):
            out[gi] = 1
        elif bn == "pen":
            out[gi] = 2
        if out[gi] in (0, 1) and not bn.startswith(NONFINGER_BODIES):
            fin[gi] = True
        if bn.startswith(("l_link", "r_link")):
            lin[gi] = True
    _geom_sides = out
    _geom_finger = fin
    _geom_islink = lin
    return out


_filter_err = 0


def _contactfilter(m, d, g1, g2):
    try:
        # MuJoCo 约定：0=保留接触，非0=过滤掉
        mode = _mask_mode
        if mode is None or _geom_sides is None:
            return 0
        s1 = int(_geom_sides[int(g1)])
        s2 = int(_geom_sides[int(g2)])
        if mode == "self":
            return 1 if (s1 in (0, 1) or s2 in (0, 1)) else 0
        if mode == "cross":
            # 跨侧身体互碰全禁
            if s1 in (0, 1) and s2 in (0, 1) and s1 != s2:
                return 1
            # 非手指臂体（臂杆/腕/掌/拇指基座）碰笔：禁，防前臂横扫扫飞笔
            if (s1 == 2 and s2 in (0, 1)) or (s2 == 2 and s1 in (0, 1)):
                arm = int(g2) if s1 == 2 else int(g1)
                return 0 if bool(_geom_finger[arm]) else 1
            # 纯臂杆（l_link*/r_link*）碰桌/世界：放行，防横移时上臂刮桌产生撞击振荡
            if s1 == -1 and s2 in (0, 1) and bool(_geom_islink[int(g2)]):
                return 1
            if s2 == -1 and s1 in (0, 1) and bool(_geom_islink[int(g1)]):
                return 1
            return 0
        return 0
    except Exception:
        globals()["_filter_err"] += 1
        return 0


try:
    mujoco.set_mjcb_contactfilter(_contactfilter)
except Exception:
    pass


def set_collision_mask(mode):
    """mode=None 全开；'cross' 只禁左↔右身体互碰（保留手-笔/手-桌）；
    'self' 手/臂 geom 对所有物体禁碰（静置标定用）。"""
    global _mask_mode
    _mask_mode = mode


# 兼容旧名（静置标定：手/臂全部禁碰）
def set_arms_hands_collision(env, on):
    set_collision_mask(None if on else "self")


def set_cross_body_collision(env, on, also_self=False):
    # 兼容名：动作段跨侧禁碰
    set_collision_mask(None if on else ("self" if also_self else "cross"))

def _fk_site(ik, q):
    """只读 FK：给定臂 7+腕 2 关节角，返回 grasp_site 世界位姿（恢复 data）。"""
    m, d = ik.model, ik.data
    saved = d.qpos.copy()
    d.qpos[ik.qadr] = np.asarray(q, dtype=np.float64)
    mujoco.mj_kinematics(m, d)
    p = d.site(ik.site_id).xpos.copy()
    R = d.site(ik.site_id).xmat.reshape(3, 3).copy()
    d.qpos[:] = saved
    mujoco.mj_kinematics(m, d)
    return p, R


def _quat_slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = (1 - t) * q0 + t * q1
        return q / np.linalg.norm(q)
    th0 = np.arccos(dot)
    s0 = np.sin((1 - t) * th0) / np.sin(th0)
    s1 = np.sin(t * th0) / np.sin(th0)
    return s0 * q0 + s1 * q1


def build_cartesian_chain(expert, ik, q_start, q_end, n_points=24, arch=0.10):
    """沿笛卡尔直线（叠加抛物线抬臂拱高）滚动 warm-start IK，
    生成关节连续的路径，避开关节空间直线穿越肘奇异导致的甩臂。"""
    P0, R0 = _fk_site(ik, q_start)
    P1, R1 = _fk_site(ik, q_end)
    Q0 = np.zeros(4)
    Q1 = np.zeros(4)
    mujoco.mju_mat2Quat(Q0, R0.reshape(9))
    mujoco.mju_mat2Quat(Q1, R1.reshape(9))
    qs = [q_start.copy()]
    qprev = q_start.copy()
    worst = 0.0
    for i in range(1, n_points + 1):
        t = i / n_points
        P = (1 - t) * P0 + t * P1
        P[2] += arch * 4.0 * t * (1.0 - t)          # 中点最高，抬臂离桌
        Qt = _quat_slerp(Q0, Q1, t)
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, Qt)
        q, ep, eo = expert._solve_from(ik, qprev, P, Rm.reshape(3, 3))
        worst = max(worst, ep)
        qs.append(q)
        qprev = q
    print(f"[plan] gather 笛卡尔滚动链 {n_points} 段 arch={arch} 末段最差 pos 残差={worst*100:.2f}cm")
    return qs


ELBOW_IDX = 3          # l_joint4/r_joint4（肘，range≈[-3.07,-0.07]，接近 -0.07=伸直奇异）
ELBOW_BENT = -1.15     # 期望肘弯角（home≈-1.30，右手持笔≈-1.19）


def tuck_solve(ik, seed, tpos, tR, elbow_tgt=ELBOW_BENT,
               elbow_gain=1.0, iters=500):
    """强零空间屈肘重解：q_seed 除肘外=seed（零空间梯度≈0），只把肘拉向 elbow_tgt，
    位置/朝向主任务由 9-DoF 冗余的其余关节承担。返回 (q9, pos_err_m, ori_err_r)。"""
    d = ik.data
    c = ik.cfg
    saved_cfg = (c.damping, c.pos_gain, c.ori_gain, c.null_gain, c.max_dq,
                 c.pos_tol, c.ori_tol)
    saved_seed = ik.q_seed.copy()
    saved_qpos = d.qpos.copy()
    try:
        ik.q_seed = seed.copy()
        ik.q_seed[ELBOW_IDX] = elbow_tgt
        c.damping, c.pos_gain, c.ori_gain, c.null_gain, c.max_dq = 0.05, 1.0, 1.0, elbow_gain, 0.3
        # 起点已是收敛解，必须关掉提前退出，否则零空间屈肘项一次都不会执行
        c.pos_tol, c.ori_tol = -1.0, -1.0
        d.qpos[ik.qadr] = seed
        q = ik.solve(tpos, tR, iters)
        ep, eo = ik.errors(tpos, tR)
        return q, ep, eo
    finally:
        (c.damping, c.pos_gain, c.ori_gain, c.null_gain, c.max_dq,
         c.pos_tol, c.ori_tol) = saved_cfg
        ik.q_seed = saved_seed
        d.qpos[:] = saved_qpos


def plan_left_goal(expert_l, ik_l, poses, C, R, hand_pose):
    """多起点 DLS + 静置补偿；末段强零空间屈肘，避免肘近伸直极限导致伺服奇异振荡。
    标定期间关闭双臂/双手 geom 碰撞（跨臂互穿会炸仿真）；恢复调用前的掩码模式。"""
    _prev_mask = _mask_mode
    set_collision_mask("self")
    try:
        q, _, _ = expert_l._solve_robust(ik_l, C, R)
        rp = ro = np.inf
        expert_l._blocked_axes = None
        for _ in range(3):
            p_need, R_need, rp, ro = expert_l._compensated_target(q, hand_pose, C, R)
            q, ep, eo = expert_l._solve_from(ik_l, q, p_need, R_need)
        # 屈肘重解 + 2 轮带肘偏好的静置补偿（普通 _solve_from 会把肘推回伸直）
        for _ in range(3):
            p_need, R_need, rp, ro = expert_l._compensated_target(q, hand_pose, C, R)
            q, ep, eo = tuck_solve(ik_l, q, p_need, R_need)
            if q[ELBOW_IDX] < -0.55 and ep < 0.015:
                break
        return q, rp, ro
    finally:
        set_collision_mask(_prev_mask)


def chain_at(qs, u):
    """整条 q 路径上按累积弧长线性取 q(u)，u∈[0,1]（段间速度连续）。"""
    u = min(1.0, max(0.0, u))
    i = int(u * (len(qs) - 1))
    if i >= len(qs) - 1:
        return qs[-1].copy()
    f = u * (len(qs) - 1) - i
    return (1 - f) * qs[i] + f * qs[i + 1]


def blend_action(l_q9, l_f18, r_q9, r_f18):
    return np.concatenate([l_q9[:7], l_q9[7:9], l_f18,
                           r_q9[:7], r_q9[7:9], r_f18]).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/transfer_marker.yaml")
    ap.add_argument("--x", type=float, default=0.0)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1002)  # wl9 协议：(0,0) 点=index2 → 1000+2
    ap.add_argument("--plan-seed", type=int, default=4)  # 已验证成功的 pick IK seed
    ap.add_argument("--out", default="verify_out/transfer_proto")
    ap.add_argument("--no-normalize", action="store_true", help="关闭笔轴规范化")
    ap.add_argument("--perturb-xy", type=float, default=0.0, help="抓持态笔心 xy 扰动幅值(m)")
    ap.add_argument("--perturb-deg", type=float, default=0.0, help="抓持态笔轴扰动幅值(deg)")
    ap.add_argument("--perturb-seed", type=int, default=0)
    ap.add_argument("--perturb-settle", type=int, default=15, help="扰动后再静置帧数")
    ap.add_argument("--nudge-deg", type=float, default=0.0, help="手腕刚性微旋幅值(deg)，绕笔心随机轴")
    ap.add_argument("--nudge-trans", type=float, default=0.0, help="手腕刚性微移幅值(m)")
    ap.add_argument("--nudge-seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[],
                    help="覆盖 transfer 段参数，如 --set handoff_along=-0.11")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(ROOT / args.cfg)
    tc = cfg["transfer"]
    for kv in args.set:
        k, v = kv.split("=", 1)
        tc[k] = type(tc.get(k, 0.0))(float(v)) if not isinstance(tc.get(k), str) else v
    hz = float(cfg.get("control_hz", 20))
    poses = {k: np.asarray(v, dtype=np.float64) for k, v in cfg["hand_poses"].items()}

    env_cfg = BimanualDexConfig(max_episode_seconds=20,
                                pen_xy=((args.x, args.x), (args.y, args.y)))
    env = BimanualDexEnv(config=env_cfg, render_mode="rgb_array")
    unwrapped = env
    build_geom_sides(unwrapped)          # contactfilter 侧别表（标定前必须就位）
    arm_damp = float(tc.get("arm_damping", 10.0))

    def apply_arm_damping(c):
        # qfrc_bias 重力前馈让臂"失重"但 position actuator 缺阻尼会激发极限环（home
        # 末端 ±4cm 振荡）；给 7 个臂关节补阻尼可零下垂且稳定。pick 之后才施加，
        # 避免改变 M1 已验证的 pick 接触抓取姿态。
        if c <= 0:
            return
        for _side in ("l", "r"):
            for _ji in range(1, 8):
                _jid = mujoco.mj_name2id(unwrapped.model, mujoco.mjtObj.mjOBJ_JOINT,
                                         f"{_side}_joint{_ji}")
                unwrapped.model.dof_damping[unwrapped.model.jnt_dofadr[_jid]] = c
    orig_randomize = unwrapped._randomize_pen

    def fixed_pen(d):
        orig_randomize(d)
        c, s = np.cos(args.yaw / 2), np.sin(args.yaw / 2)
        qflat = np.array([0.7071068, 0.0, 0.7071068, 0.0])
        qyaw = np.array([c, 0.0, 0.0, s])
        d.qpos[3:7] = unwrapped._quat_mul(qyaw, qflat)

    unwrapped._randomize_pen = fixed_pen
    obs, _ = env.reset(seed=args.seed)

    # ---------- 阶段 0：pick（右手专家，到 lift 结束即停） ----------
    cfg_r = copy.deepcopy(cfg)
    cfg_r["allowed_hands"] = ["right"]
    cfg_r["plan_seed"] = int(args.plan_seed)
    expert_r = ScriptedExpert(unwrapped, cfg_r)
    expert_r.reset(obs)
    print(f"[pick] hand={expert_r.side} mode={expert_r.mode}")
    pick_frames = 0
    while expert_r.si < 4:           # 0..3 = approach/descend/close/lift
        obs, _, term, trunc, info = env.step_rad(expert_r.act(obs))
        pick_frames += 1
        if term or trunc:
            # success/fell 只改奖励与终止标志，物理继续；原型刻意跑到 lift 末
            pass
    print(f"[pick] {pick_frames} 帧, grasped={info['grasped']} pen_z={info['pen_z']:.3f}")
    pick_grasped = info["grasped"]
    apply_arm_damping(arm_damp)   # 交接前才开臂阻尼（保持原生 pick 抓取姿态）

    # 右手 lift 末关节/手指状态（9 维臂腕 + 18 手指）
    r_q9 = unwrapped.data.qpos[expert_r.ik.qadr].copy()
    # 握力加固（问题 38）：pinch3 是边际握持，扰动后笔在 carry 途中重力 pivot 下垂；
    # 沿 pre_pinch→pinch3 合拢方向外推 tighten 比例提高法向力（位置 actuator 超程加压）。
    _tighten = float(tc.get("grip_tighten", 0.0))
    r_f18 = (poses["pinch3"] + _tighten * (poses["pinch3"] - poses["pre_pinch"]))[2:20].copy()
    # 左手初始（home 张开）
    cfg_l = copy.deepcopy(cfg)
    cfg_l["allowed_hands"] = ["left"]
    cfg_l["n_restarts"] = int(tc["plan_restarts"])
    expert_l = ScriptedExpert(unwrapped, cfg_l)   # 不 reset：仅借 IK 与标定方法
    ik_l = expert_l.iks["left"]
    l_q9 = unwrapped.model.key_qpos[expert_l._key_home][ik_l.qadr].copy()
    l_f18 = poses["open"][2:20].copy()

    # ---------- pick 后持笔静置：加阻尼后 lift 截断时笔仍在转动，先收敛再规划 ----------
    n_settle = int(tc.get("settle_frames", 25))
    for _ in range(n_settle):
        obs, _, _, _, _ = env.step_rad(blend_action(l_q9, l_f18, r_q9, r_f18))
    print(f"[settle] {n_settle}帧 pen={np.asarray(obs['object'][:3]).round(3)} "
          f"axis={pen_axis(obs).round(3)} pen_z={env._pen_lowest_z():.3f}")
    obs_pre = obs

    # ---------- 抓持态扰动：笔在右手中注入小位姿噪声（录数多样性来源，问题 35） ----------
    perturb_info = {"xy": float(args.perturb_xy), "deg": float(args.perturb_deg),
                    "seed": int(args.perturb_seed), "ran": False}
    if args.perturb_xy > 0.0 or args.perturb_deg > 0.0:
        prng = np.random.default_rng(args.perturb_seed)
        dxy = prng.uniform(-args.perturb_xy, args.perturb_xy, 2)
        dax = np.radians(prng.uniform(-args.perturb_deg, args.perturb_deg, 2))
        dd = unwrapped.data
        dd.qpos[0] += dxy[0]
        dd.qpos[1] += dxy[1]
        q = dd.qpos[3:7].copy()
        for _ax, _an in ((np.array([1.0, 0.0, 0.0]), dax[0]),
                         (np.array([0.0, 1.0, 0.0]), dax[1])):
            _c, _s = np.cos(_an / 2), np.sin(_an / 2)
            _dq = np.array([_c, _ax[0] * _s, _ax[1] * _s, _ax[2] * _s])
            q = unwrapped._quat_mul(_dq, q)
        dd.qpos[3:7] = q
        dd.qvel[0:6] = 0.0
        mujoco.mj_forward(unwrapped.model, dd)
        for _ in range(int(args.perturb_settle)):
            obs, _, _, _, _ = env.step_rad(blend_action(l_q9, l_f18, r_q9, r_f18))
        perturb_info.update({"ran": True, "dxy": dxy.round(5).tolist(),
                             "ddeg": np.degrees(dax).round(3).tolist()})
        print(f"[perturb] dxy={dxy.round(4)} ddeg={np.degrees(dax).round(2)} "
              f"pen={np.asarray(obs['object'][:3]).round(3)} axis={pen_axis(obs).round(3)} "
              f"pen_z={env._pen_lowest_z():.3f}")
        obs_pre = obs

    # ---------- 手腕刚性 nudge：整手带笔做小旋转/平移（形闭包内部相对位姿不变，问题 39） ----------
    # 与 teleport 扰动（改笔相对手的位姿→破坏接触构型→carry 重力 pivot 下垂，问题 35）
    # 的根本区别：nudge 只改"手+笔"整体在世界系中的位姿，抓持接触完全保留。
    nudge_info = {"deg": float(args.nudge_deg), "trans": float(args.nudge_trans),
                  "seed": int(args.nudge_seed), "ran": False}
    if args.nudge_deg > 0.0 or args.nudge_trans > 0.0:
        nrng = np.random.default_rng(args.nudge_seed)
        _ax = nrng.normal(size=3)
        _ax = _ax / np.linalg.norm(_ax)
        _ang = np.radians(nrng.uniform(-args.nudge_deg, args.nudge_deg))
        _tr = nrng.uniform(-args.nudge_trans, args.nudge_trans, 3)
        R_d = rot_about(_ax, _ang)
        pc_n = np.asarray(obs["object"][:3], dtype=np.float64)
        p_s0, R_s0 = _fk_site(expert_r.ik, r_q9)
        p_ng = pc_n + R_d @ (p_s0 - pc_n) + _tr
        R_ng = R_d @ R_s0
        q_ng, ep_ng, _ = expert_r._solve_from(expert_r.ik, r_q9, p_ng, R_ng, iters=80)
        print(f"[nudge] ang={np.degrees(_ang):.2f}° tr={(_tr*1000).round(1)}mm "
              f"IK 残差 pos={ep_ng*100:.2f}cm")
        if ep_ng < 0.02:
            nq = build_cartesian_chain(expert_r, expert_r.ik, r_q9, q_ng, n_points=8, arch=0.0)
            for k in range(15):
                u = _min_jerk(k / 15.0)
                obs, _, _, _, _ = env.step_rad(blend_action(l_q9, l_f18, chain_at(nq, u), r_f18))
            for _ in range(5):
                obs, _, _, _, _ = env.step_rad(blend_action(l_q9, l_f18, q_ng, r_f18))
            r_q9 = q_ng
            nudge_info.update({"ran": True, "ang_deg": round(float(np.degrees(_ang)), 3),
                               "trans_mm": (_tr * 1000).round(2).tolist()})
            print(f"[nudge] 完成 pen={np.asarray(obs['object'][:3]).round(3)} "
                  f"axis={pen_axis(obs).round(3)} pen_z={env._pen_lowest_z():.3f}")
            obs_pre = obs
        else:
            print("[nudge] IK 残差过大，跳过")

    # ---------- 笔轴规范化：在右手中用腕自由度把笔转到标准朝向 ----------
    # 不同 pick plan_seed 的笔轴 a 差异大，handoff 会合点完全由 a 决定（问题 29）：
    # 闭环迭代把笔绕笔心最小旋转到标准轴 a_star（每步限幅 normalize_max_step_deg，
    # 大步旋转会在 pinch3 握持中打滑/甩飞），后续规划基于规范化后的实测位姿。
    a_star = np.asarray(tc.get("normalize_axis", [0.903, -0.224, -0.366]), dtype=np.float64)
    a_star = a_star / np.linalg.norm(a_star)
    norm_info = {"axis_star": a_star.round(4).tolist(), "ran": False, "iters": 0}
    a0 = pen_axis(obs)
    norm_info["axis_before"] = a0.round(4).tolist()
    norm_info["angle_before_deg"] = round(float(np.degrees(
        np.arccos(np.clip(np.dot(a0, a_star), -1.0, 1.0)))), 2)
    if bool(tc.get("normalize_enable", True)) and not args.no_normalize:
        _min_deg = float(tc.get("normalize_min_deg", 3.0))
        _max_step = float(tc.get("normalize_max_step_deg", 30.0))
        _max_iter = int(tc.get("normalize_max_iter", 4))
        n_norm = int(round(float(tc.get("normalize_seconds", 2.0)) * hz))
        n_set = int(tc.get("normalize_settle", 10))
        for _it in range(_max_iter):
            a0 = pen_axis(obs)
            ang0 = float(np.degrees(np.arccos(np.clip(np.dot(a0, a_star), -1.0, 1.0))))
            # 笔已落桌（近垂直弱握会持续下滑，转不动）则停止
            if ang0 < _min_deg or np.asarray(obs["object"][:3])[2] < 0.55:
                break
            v = np.cross(a0, a_star)
            if np.linalg.norm(v) < 1e-9:
                break
            v = v / np.linalg.norm(v)
            th = np.radians(min(ang0, _max_step))
            K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            R_d = np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)
            p_pen0 = np.asarray(obs["object"][:3], dtype=np.float64)
            p_site0, R_site0 = _fk_site(expert_r.ik, r_q9)
            p_tgt = p_pen0 + R_d @ (p_site0 - p_pen0)
            R_tgt = R_d @ R_site0
            q_norm, ep_n, eo_n = expert_r._solve_from(expert_r.ik, r_q9, p_tgt, R_tgt)
            print(f"[normalize] it{_it} 轴偏 {ang0:.1f}° 步进 {np.degrees(th):.1f}° "
                  f"IK 残差 pos={ep_n*100:.1f}cm ori={np.degrees(eo_n):.1f}°")
            if ep_n >= 0.03:
                print("[normalize] IK 残差过大，停止规范化")
                break
            norm_q = build_cartesian_chain(
                expert_r, expert_r.ik, r_q9, q_norm,
                n_points=int(tc.get("normalize_chain_points", 16)),
                arch=float(tc.get("normalize_arch", 0.0)))
            for k in range(n_norm):
                u = _min_jerk(k / max(n_norm, 1))
                obs, _, _, _, info = env.step_rad(blend_action(
                    l_q9, l_f18, chain_at(norm_q, u), r_f18))
            for _ in range(n_set):
                obs, _, _, _, info = env.step_rad(blend_action(
                    l_q9, l_f18, q_norm, r_f18))
            r_q9 = q_norm
            norm_info["ran"] = True
            norm_info["iters"] = _it + 1
    a1 = pen_axis(obs)
    ang1 = float(np.degrees(np.arccos(np.clip(np.dot(a1, a_star), -1.0, 1.0))))
    norm_info["axis_after"] = a1.round(4).tolist()
    norm_info["angle_after_deg"] = round(ang1, 2)
    print(f"[normalize] ran={norm_info['ran']} iters={norm_info['iters']} "
          f"轴偏 {norm_info['angle_before_deg']:.1f}°→{ang1:.1f}° "
          f"pen={np.asarray(obs['object'][:3]).round(3)} rh={hand_pen_contact(unwrapped,'rh')}")
    Image.fromarray(obs["images"]["head"]).save(out / "normalize.png")

    # ---------- 交接几何规划 ----------
    pick_pen = np.asarray(obs["object"][:3], dtype=np.float64)
    a = pen_axis(obs)
    s = np.array([a[1], -a[0], 0.0]); s /= np.linalg.norm(s) + 1e-9

    # handoff 会合笔位（右手先把笔搬过来），笔轴方向近似不变（只平移）
    pen_pos = (pick_pen + a * float(tc["handoff_along"])
               + _UP * float(tc["handoff_vertical"]))
    # carry 目标朝向用 grasp_frame 理想系（非 FK 实际系）：左承接 R_l 同出 grasp_frame(a)，
    # 拖腕修正可让会合点实际笔姿态对齐左手规划假设；FK 纯平移实测 catch 抓不上（已证伪）
    R_r = grasp_frame_safe(a, expert_r.mode, expert_r.frame_tilts.get("right", 0.0))
    q_handoff, ep_h, eo_h = expert_r._solve_from(expert_r.ik, r_q9, pen_pos, R_r)
    print(f"[plan] handoff C={pen_pos.round(3)} R: pos={ep_h*100:.1f}cm ori={np.degrees(eo_h):.1f}°")
    # carry 右手链：沿笛卡尔路径 warm-start 滚动 IK（抛物线抬臂防甩笔）
    carry_q = build_cartesian_chain(
        expert_r, expert_r.ik, r_q9, q_handoff,
        n_points=int(tc.get("carry_chain_points", 24)),
        arch=float(tc.get("carry_arch", 0.10)))
    # 右手在 handoff 位的实际 site（撤离规划基准）
    savedq = unwrapped.data.qpos.copy()
    unwrapped.data.qpos[expert_r.ik.qadr] = q_handoff
    mujoco.mj_kinematics(unwrapped.model, unwrapped.data)
    p_handoff_site = unwrapped.data.site(expert_r.ik.site_id).xpos.copy()
    unwrapped.data.qpos[:] = savedq
    mujoco.mj_kinematics(unwrapped.model, unwrapped.data)

    expert_l.side = "left"
    expert_l.ik = ik_l
    expert_l.off = 0
    expert_l.mode = int(tc["left_mode"])
    C_l = (pen_pos + a * float(tc["along"]) + s * float(tc["lateral"])
           + _UP * float(tc["vertical"])
           + _UP * float(tc.get("left_goal_dz", 0.0)))   # 左承接位 z 微调（补偿右 site-笔中心几何偏移）
    # 圆柱笔绕轴 1-DoF 冗余：网格搜索绕轴扭转角 φ，取左臂 IK 残差最小的抓握朝向
    # （近垂直笔轴下默认 frame 常落在左腕/肘不可达区，问题 29/30 根因之一）
    R_l0 = grasp_frame_safe(a, expert_l.mode, expert_l.frame_tilts.get("left", 0.0))
    # 默认优先、奇异才搜索：φ=0 粗朝向残差超阈值才在 ±range 窗口内找可达扭转，
    # 避免搜到"左手包抄右手"的几何（右手无法撤离，已实测失败）
    _, _ep0, _eo0 = expert_l._solve_from(ik_l, l_q9, C_l, R_l0, iters=200)
    R_l = R_l0
    if np.degrees(_eo0) > float(tc.get("left_twist_trigger_deg", 25.0)):
        n_tw = int(tc.get("left_twist_search", 12))
        _tw_rng = np.radians(float(tc.get("left_twist_range_deg", 90.0)))
        tw_best = None
        for phi in np.linspace(-_tw_rng, _tw_rng, n_tw):
            R_phi = rot_about(a, phi) @ R_l0
            _, _ep, _eo = expert_l._solve_from(ik_l, l_q9, C_l, R_phi, iters=200)
            _sc = _ep + 0.02 * _eo
            if tw_best is None or _sc < tw_best[0]:
                tw_best = (_sc, phi, R_phi, _ep, _eo)
        if tw_best[4] < _eo0:
            R_l = tw_best[2]
            print(f"[plan] left twist 触发(φ0 ori={np.degrees(_eo0):.1f}°) "
                  f"φ*={np.degrees(tw_best[1]):.0f}° 粗残差 ori={np.degrees(tw_best[4]):.1f}°")
    # 承接臂目标按 catch 末【执行手型 pinch3】静置标定：
    # lh_grasp_site 挂在拇指基座上，手型不同 site 世界位漂移可达 7cm，
    # 必须按最终握持手型补偿（与 pick 专家 descend 同原则）

    # 左手等待/预抓位：沿笔轴在承接位靠左手侧(-a)外 approach_dist 处，朝向与最终
    # 抓握 R_l 一致；该位在笔自由端之外，右手 carry 全程不与其干涉。
    _appr = float(tc.get("axis_approach", 0.13))
    C_ret = C_l - a * _appr
    q_retreat, rp_ret, ro_ret = expert_l._solve_from(ik_l, l_q9, C_ret, R_l)
    if rp_ret > 0.02:
        q_retreat, rp_ret, ro_ret = expert_l._solve_robust(ik_l, C_ret, R_l)
    print(f"[plan] left axis-pre C={C_ret.round(3)} d={_appr:.2f} pos={rp_ret*100:.1f}cm ori={np.degrees(ro_ret):.1f}°")
    retreat_q = build_cartesian_chain(expert_l, ik_l, l_q9, q_retreat,
                                      n_points=int(tc.get("retreat_chain_points", 16)),
                                      arch=float(tc.get("retreat_arch", 0.12)))

    # ---------- handoff 时间表（右手 carry 送笔 + 左手撤到沿轴预抓位）----------
    f_pre = poses[tc["gather_hand"]][2:20]
    f_catch = poses[tc["catch_hand"]][2:20]
    f_open_r = poses[tc["release_hand"]][2:20]

    T = int(round(float(tc.get("handoff_seconds", 2.5)) * hz))
    t_ret = float(tc.get("t_retreat_done", 0.30))
    t_carry = float(tc.get("t_carry_done", 0.55))
    TL_q, TL_f, TR_q, TR_f = [], [], [], []
    for k in range(T):
        tt = k / max(T - 1, 1)
        TR_q.append(chain_at(carry_q, _min_jerk(min(1.0, tt / t_carry))))
        TR_f.append(r_f18.copy())
        if tt < t_ret:
            TL_q.append(chain_at(retreat_q, _min_jerk(tt / t_ret)))
        else:
            TL_q.append(q_retreat.copy())
        TL_f.append(f_pre.copy())
    print(f"[plan] 串行时间表 T={T}帧 t_ret={t_ret} t_carry={t_carry}")

    # transfer 动作段：禁跨侧身体互碰（左手指碰右臂炸仿真，软组织近似）
    set_collision_mask("cross")

    result = {"pick_frames": pick_frames, "phases": [], "success": False,
              "normalize": norm_info,
              "pick_grasped": bool(pick_grasped),
              "settle_axis": np.asarray(pen_axis(obs_pre)).round(4).tolist(),
        "perturb": perturb_info,
        "nudge": nudge_info}
    hold_need = int(tc["hold_frames"])
    hold_cnt = 0
    cur_l = (l_q9, l_f18)
    cur_r = (r_q9, r_f18)

    def run_phases(phases):
        nonlocal obs, hold_cnt, cur_l, cur_r
        for name, sec, l0, l1, r0, r1, extra in phases:
            n = max(1, int(round(sec * hz)))
            l0q, l0f = l0; l1q, l1f = l1
            r0q, r0f = r0; r1q, r1f = r1
            phase_info = {"name": name, "frames": n}
            kind = extra[0] if extra else None
            qchain = extra[1] if extra else None
            fchain = extra[2] if extra else None
            if name == "catch":
                _hd = float(tc.get("hold_arm_damping", 10.0))
                for _ji in range(1, 8):
                    _jid = mujoco.mj_name2id(unwrapped.model, mujoco.mjtObj.mjOBJ_JOINT,
                                             f"r_joint{_ji}")
                    unwrapped.model.dof_damping[unwrapped.model.jnt_dofadr[_jid]] = _hd
            for k in range(n):
                u = _min_jerk(k / max(n, 1))
                if kind == "timed":
                    cur_l = (extra[1][k], extra[2][k])
                    cur_r = (extra[3][k], extra[4][k])
                elif kind == "chain_l":
                    cur_l = (chain_at(qchain, u), chain_at(fchain, u))
                    cur_r = ((1 - u) * r0q + u * r1q, (1 - u) * r0f + u * r1f)
                elif kind == "track_l":
                    # 闭环承接：每帧按实测笔心重解左臂 IK（朝向固定标称 R_l）。
                    # 前 track_approach_frac 段目标点从起始 site 按 min-jerk 混入
                    # （无剖面直接跟踪=首帧瞬移 13cm 撞笔，已证伪），之后纯跟踪；
                    # 笔从右手滑落即变成移动目标，左手跟随接住而非开环撞笔。
                    R_lt, f0t, f1t, cst = extra[1], extra[2], extra[3], extra[4]
                    if k == 0:
                        p_start_l = _fk_site(ik_l, cur_l[0])[0]
                    pc = np.asarray(obs["object"][:3], dtype=np.float64)
                    p_pen = pc + _UP * float(tc.get("left_goal_dz", 0.0))
                    vv = k / max(n - 1, 1)
                    ta = float(tc.get("track_approach_frac", 0.5))
                    mj = _min_jerk(min(1.0, vv / ta))
                    tgt_p = (1 - mj) * p_start_l + mj * p_pen
                    # 朝向跟踪（问题 36）：标称凹口对抓持态扰动零容差（±1mm 即顶飞），
                    # 每帧按实测轴重建承接 frame（叠加规划时可达性扭转偏置 φ*）。
                    # 大轴偏（>10°）实测系左腕不可达——该模式只服务小扰动包络。
                    if str(tc.get("track_orient", "nominal")) == "meas":
                        a_m = pen_axis(obs)
                        R_use = rot_about(a_m, twist_phi) @ grasp_frame_safe(
                            a_m, expert_l.mode, expert_l.frame_tilts.get("left", 0.0))
                    else:
                        R_use = R_lt
                    q_new, ep_t, _ = expert_l._solve_from(ik_l, cur_l[0], tgt_p, R_use, iters=80)
                    if ep_t < 0.04:
                        dq = q_new - cur_l[0]
                        mxd = float(tc.get("track_max_dq", 0.12))
                        nrm = np.abs(dq).max()
                        lq = cur_l[0] + dq * (mxd / nrm) if nrm > mxd else q_new
                    else:
                        lq = cur_l[0]
                    ww = 0.0 if vv < cst else min(1.0, (vv - cst) / (1.0 - cst))
                    cur_l = (lq, (1 - ww) * f0t + ww * f1t)
                    cur_r = ((1 - u) * r0q + u * r1q, (1 - u) * r0f + u * r1f)
                elif kind == "chain_r":
                    cur_r = (chain_at(qchain, u), chain_at(fchain, u))
                    cur_l = ((1 - u) * l0q + u * l1q, (1 - u) * l0f + u * l1f)
                else:
                    cur_l = ((1 - u) * l0q + u * l1q, (1 - u) * l0f + u * l1f)
                    cur_r = ((1 - u) * r0q + u * r1q, (1 - u) * r0f + u * r1f)
                obs, _, _, _, info = env.step_rad(blend_action(cur_l[0], cur_l[1],
                                                               cur_r[0], cur_r[1]))
                if k % 10 == 0 or k == n - 1:
                    dd2 = unwrapped.data
                    sp = dd2.site(ik_l.site_id).xpos
                    xtra = ""
                    if k == n - 1:
                        pp = []
                        for ci in range(dd2.ncon):
                            cc = dd2.contact[ci]
                            c1 = mujoco.mj_id2name(unwrapped.model, mujoco.mjtObj.mjOBJ_BODY,
                                                   int(unwrapped.model.geom(cc.geom1).bodyid)) or "?"
                            c2 = mujoco.mj_id2name(unwrapped.model, mujoco.mjtObj.mjOBJ_BODY,
                                                   int(unwrapped.model.geom(cc.geom2).bodyid)) or "?"
                            pp.append(f"{c1}-{c2}")
                        xtra = f" pairs={sorted(set(pp))}"
                    print(f"  {name[:2]}{k:2d} z={info['pen_z']:.3f} lh={hand_pen_contact(unwrapped,'lh')} "
                          f"rh={hand_pen_contact(unwrapped,'rh')} lsite={np.round(sp,3).tolist()}{xtra}")
                if name == "hold":
                    lh = hand_pen_contact(unwrapped, "lh")
                    rh = hand_pen_contact(unwrapped, "rh")
                    lifted = unwrapped._pen_lowest_z() > TABLE_TOP_Z + unwrapped.cfg.lift_clearance
                    stable = np.linalg.norm(unwrapped.data.qvel[0:3]) < unwrapped.cfg.max_lin_vel
                    ok = lh and not rh and lifted and stable
                    hold_cnt = hold_cnt + 1 if ok else 0
                    if k == n - 1:
                        phase_info.update(lh=lh, rh=rh, lifted=bool(lifted),
                                          stable=bool(stable), hold=hold_cnt)
            if name == "catch":
                md, dd = unwrapped.model, unwrapped.data
                pc = dd.body(unwrapped._pen_body).xpos
                qa = quat_to_R(np.asarray(dd.body(unwrapped._pen_body).xquat)) @ np.array([0., 0., 1.])
                qa /= np.linalg.norm(qa)
                for nm in ("lh_ffdistal", "lh_mfdistal", "lh_thdistal", "rh_ffdistal"):
                    bid = mujoco.mj_name2id(md, mujoco.mjtObj.mjOBJ_BODY, nm)
                    fp = dd.body(bid).xpos
                    tv = np.clip(np.dot(pc - fp, qa), -0.075, 0.075)
                    closest = fp + qa * tv
                    d_ax = np.linalg.norm(closest - pc)
                    print(f"    {nm}: 轴垂距={d_ax*100:.1f}cm 端距={np.linalg.norm(fp-pc)*100:.1f}cm")
            Image.fromarray(obs["images"]["head"]).save(out / f"{name}.png")
            pen_z = float(unwrapped.data.body(unwrapped._pen_body).xpos[2])
            phase_info["pen_z_end"] = round(pen_z, 3)
            result["phases"].append(phase_info)
            print(f"[{name}] {n}帧 pen_z={pen_z:.3f} "
                  f"lh={hand_pen_contact(unwrapped,'lh')} rh={hand_pen_contact(unwrapped,'rh')}")

    twist_phi = 0.0   # replan 扭转搜索的可达性偏置（track_orient=meas 时叠加）

    # ---- 执行 handoff（左手按标称轴撤到沿轴预抓位，右手送笔）----
    run_phases([("handoff", float(T) / hz,
                 (l_q9.copy(), l_f18.copy()), (q_retreat.copy(), f_pre),
                 (r_q9.copy(), r_f18.copy()), (q_handoff.copy(), r_f18.copy()),
                 ("timed", TL_q, TL_f, TR_q, TR_f))])

    # ---------- handoff 后按【实测位置 + 标称轴朝向】在线重规划 ----------
    # 朝向规划仍用 carry 前标称轴 a（成功机制是左手按浅轴形成承接凹口，笔从右手
    # 滑落入凹口自动对正；carry 后实测轴偏陡 30°+ 时左腕不可达，已证伪）；
    # 位置用实测值（消除 carry 落点漂移导致的走廊错位撞笔）。
    a_meas = pen_axis(obs)
    pen_pos2 = np.asarray(obs["object"][:3], dtype=np.float64)
    l_now = unwrapped.data.qpos[ik_l.qadr].copy()
    r_now = unwrapped.data.qpos[expert_r.ik.qadr].copy()
    print(f"[replan] 实测 pen={pen_pos2.round(3)} axis_meas={a_meas.round(3)} 规划轴={a.round(3)}")
    C_l = (pen_pos2 + a * float(tc["along"]) + s * float(tc["lateral"])
           + _UP * float(tc["vertical"])
           + _UP * float(tc.get("left_goal_dz", 0.0)))
    R_l0 = grasp_frame_safe(a, expert_l.mode, expert_l.frame_tilts.get("left", 0.0))
    _, _ep0, _eo0 = expert_l._solve_from(ik_l, l_now, C_l, R_l0, iters=200)
    R_l = R_l0
    if np.degrees(_eo0) > float(tc.get("left_twist_trigger_deg", 25.0)):
        n_tw = int(tc.get("left_twist_search", 12))
        _tw_rng = np.radians(float(tc.get("left_twist_range_deg", 90.0)))
        tw_best = None
        for phi in np.linspace(-_tw_rng, _tw_rng, n_tw):
            R_phi = rot_about(a, phi) @ R_l0
            _, _ep, _eo = expert_l._solve_from(ik_l, l_now, C_l, R_phi, iters=200)
            _sc = _ep + 0.02 * _eo
            if tw_best is None or _sc < tw_best[0]:
                tw_best = (_sc, phi, R_phi, _ep, _eo)
        if tw_best[4] < _eo0:
            R_l = tw_best[2]
            twist_phi = float(tw_best[1])
            print(f"[replan] left twist 触发 φ*={np.degrees(tw_best[1]):.0f}° "
                  f"粗残差 ori={np.degrees(tw_best[4]):.1f}°")
    # 右手撤离：从当前实际 site 沿标称 +笔轴 平移
    p_site_now, _ = _fk_site(expert_r.ik, r_now)
    C_rel = p_site_now + a * float(tc["release_along"])
    q_rel, rp_rel, ro_rel = expert_r._solve_from(expert_r.ik, r_now, C_rel, R_r)
    print(f"[replan] right release C={C_rel.round(3)} 残差 pos={rp_rel*100:.1f}cm ori={np.degrees(ro_rel):.1f}°")

    # ---- catch（闭环跟踪）/ release / hold ----
    # 右手静止目标必须用规划链末点 carry_q[-1] 而非实测 r_now：笔是靠"伺服仍在向
    # 链末收敛"的残余压力握住的，冻结实测值会卸压掉笔（已实测 ca10 内掉落）。
    run_phases([
        ("catch", float(tc["catch_seconds"]),
         (l_now.copy(), f_pre), (l_now.copy(), f_catch),
         (carry_q[-1].copy(), r_f18.copy()), (carry_q[-1].copy(), r_f18.copy()),
         ("track_l", R_l, f_pre, f_catch, float(tc.get("catch_close_start", 0.6)))),
    ])
    l_end = unwrapped.data.qpos[ik_l.qadr].copy()
    f_end = cur_l[1]
    run_phases([
        ("release", float(tc["release_seconds"]),
         (l_end.copy(), f_end.copy()), (l_end.copy(), f_end.copy()),
         (carry_q[-1].copy(), r_f18.copy()), (q_rel.copy(), f_open_r), None),
        ("hold", float(tc["hold_seconds"]),
         (l_end.copy(), f_end.copy()), (l_end.copy(), f_end.copy()),
         (q_rel.copy(), f_open_r), (q_rel.copy(), f_open_r), None),
    ])

    result["success"] = hold_cnt >= hold_need
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\nTRANSFER success={result['success']}（hold 连续 {hold_cnt}/{hold_need} 帧）")
    print(f"截图/结果: {out}")
    set_collision_mask(None)
    env.close()
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
