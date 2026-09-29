"""配置驱动的三段式脚本专家（张开接近 → 预弯下降 → 合拢 → 抬升保持）。

策略（关节空间路点，稳定不抖）：
1. reset 时按笔位 yaw，从 8 个候选（左右手 × 4 手掌朝向）中选下降路点 IK
   残差最小者（顺握/横握各 ±，覆盖 Franka A4/A6 单侧限位下的全部笔朝向，
   左手负责远左角、右手负责远右角）；
2. 每个阶段入口对末端笛卡尔位姿做多起点 DLS-IK 得到关节终态
   （home 种子失败时扰动肩肘重启，跳出局部极小）；
3. 阶段内 9 维臂腕 + 18 维手指目标做五次多项式插值，env 自身做 π/18 限幅。

阶段由 configs/*.yaml 声明：
  target = 笔位 + pen_axis*along + side_axis*lateral + z_axis*vertical
  4 候选朝向均 z=竖直向下，x=±笔轴 或 ±笔侧向。
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.ik_solver import ArmIK, IKConfig, quat_to_R  # noqa: E402

_UP = np.array([0.0, 0.0, 1.0])


def _min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


class ScriptedExpert:
    def __init__(self, env, cfg: dict):
        self.env = env
        self.cfg = cfg
        self.hz = cfg.get("control_hz", 20)
        allowed = cfg.get("allowed_hands", ["left", "right"])
        self.iks = {s: ArmIK(env, s, IKConfig(**cfg["ik"]),
                             include_wrist=True, wrist_seed=cfg.get("wrist", [0.0, 0.0]))
                    for s in allowed}
        self.poses = {k: np.asarray(v, dtype=np.float64) for k, v in cfg["hand_poses"].items()}
        self.stages = cfg["stages"]
        self.plan_iters = int(cfg.get("plan_iters", 250))
        self.n_restarts = int(cfg.get("plan_restarts", 6))
        # 捏取要求拇指-手指对捏方向横跨笔轴：仅 mode0/1（手掌 x 沿 ±笔轴）
        self.grasp_modes = list(cfg.get("grasp_modes", [0, 1]))
        self.ori_weight = float(cfg.get("select_orient", {}).get("ori_weight", 0.7))
        # 抓取帧绕手掌 x 轴（笔轴）的额外倾角（rad）：把正下俯捏改成侧捏，
        # 让拇指与食/中指在细笔两侧等高会合（Shadow 拇指指尖自然高约 5cm）。
        # 支持标量（双手同值）或 {right: …, left: …} 逐手配置（左右镜像倾角反号）
        ft = cfg.get("frame_tilt", 0.0)
        if isinstance(ft, dict):
            self.frame_tilts = {k: float(v) for k, v in ft.items()}
        else:
            self.frame_tilts = {"right": float(ft), "left": float(ft)}
        self.frame_tilt = float(ft) if not isinstance(ft, dict) else 0.0
        self._rng = np.random.default_rng(int(cfg.get("plan_seed", 0)))
        home = env.home_action_rad
        self._home = home.copy()
        self._key_home = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.side = allowed[0]
        self.ik = self.iks[self.side]
        self.mode = 0
        self.cand_scores = {}
        self.plan_errors = []
        self._anchor = None

    # ------------------------------------------------------------------ reset
    def reset(self, obs=None):
        self.si = 0
        self.k = 0
        self.plan_errors = []
        # 接触反应式横向微调（沿握点侧向轴 s 的偏移 o）
        self._react_o = 0.0
        self._react_locked = False
        self._react_t = 0
        self._react_side_runs = [0, 0]  # [连续仅拇指, 连续仅食中]
        if obs is None:
            self.mode = 0
            self._anchor = None
            return
        pen_pos, pen_q = self._pen_pose(obs)
        axis = quat_to_R(pen_q) @ np.array([0.0, 0.0, 1.0])
        axis /= np.linalg.norm(axis)
        self._anchor = (pen_pos.copy(), axis.copy())
        self._select_hand_mode(obs)
        self.ik = self.iks[self.side]
        self.off = 27 if self.side == "right" else 0
        self._begin_stage(obs)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _pen_pose(obs):
        o = obs["object"]
        return np.asarray(o[:3], dtype=np.float64), np.asarray(o[3:7], dtype=np.float64)

    def _fingers_goal(self, name):
        return self.poses[name][2:20].copy()

    @staticmethod
    def _axes(pen_axis):
        a = pen_axis / (np.linalg.norm(pen_axis) + 1e-9)
        s = np.array([a[1], -a[0], 0.0])
        s /= np.linalg.norm(s) + 1e-9
        return a, s

    def _grasp_frame(self, pen_axis, mode):
        a, s = self._axes(pen_axis)
        x_col = (a, -a, s, -s)[mode]
        y_col = np.cross(-_UP, x_col)
        R0 = np.column_stack([x_col, y_col, -_UP])
        t = self.frame_tilts.get(self.side, 0.0)
        if t:
            # 绕手掌 x（笔轴方向）旋转：y/z 平面倾斜
            ct, st = np.cos(t), np.sin(t)
            Rx = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
            R0 = R0 @ Rx
        return R0, s

    def _waypoint(self, st, pen_pos, pen_axis, anchor, mode):
        if st["reference"] == "anchor":
            pen_pos, pen_axis = anchor
        R, s = self._grasp_frame(pen_axis, mode)
        a, _ = self._axes(pen_axis)
        # 阶段字段 vertical/along/lateral 均相对【握点】；排名用粗略 site 目标，
        # 真正执行的下降/抬升构型在求解后做物理静置补偿
        pos = pen_pos + a * st["along"] + s * st["lateral"] + _UP * st["vertical"]
        return pos, R

    def _current_q9(self):
        return self.env.data.qpos[self.ik.qadr].copy()

    def _current_fingers(self):
        return np.asarray(self.env.data.actuator_length[self.off + 9:self.off + 27],
                          dtype=np.float64).copy()

    def _solve_from(self, ik, seed_q, tpos, tR, iters=None):
        # 从指定种子做一次 DLS 求解，返回 (q9, ep, eo)，恢复 data 状态
        m, d = ik.model, ik.data
        saved = d.qpos.copy()
        d.qpos[ik.qadr] = np.asarray(seed_q, dtype=np.float64)
        self._aggressive(ik)
        try:
            q = ik.solve(tpos, tR, iters=iters or self.plan_iters)
            d.qpos[ik.qadr] = q
            mujoco.mj_kinematics(m, d)
            return q, *ik.errors(tpos, tR)
        finally:
            d.qpos[:] = saved
            mujoco.mj_kinematics(m, d)
            self._restore_cfg(ik)

    def _solve_robust(self, ik, tpos, tR):
        # home 种子 + 肩肘扰动多起点，返回 (q9, pos_err, ori_err)
        m, d = ik.model, ik.data
        home_q = m.key_qpos[self._key_home][ik.qadr]

        def attempt(seed_q, iters):
            return self._solve_from(ik, seed_q, tpos, tR, iters)

        best = attempt(d.qpos[ik.qadr].copy(), self.plan_iters)

        def score(r):
            return r[1] + self.ori_weight * r[2]

        if score(best) > 0.03:
            for _ in range(self.n_restarts):
                q0 = home_q.copy()
                q0[:7] += self._rng.uniform(-1.2, 1.2, 7)
                q0[:7] = np.clip(q0[:7], ik.lo[:7] + 0.05, ik.hi[:7] - 0.05)
                cand = attempt(q0, self.plan_iters)
                if score(cand) < score(best):
                    best = cand
        return best

    @staticmethod
    def _aggressive(ik):
        c = ik.cfg
        c._saved = (c.damping, c.pos_gain, c.ori_gain, c.null_gain, c.max_dq)
        c.damping, c.pos_gain, c.ori_gain, c.null_gain, c.max_dq = 0.03, 1.0, 1.0, 0.03, 0.6

    @staticmethod
    def _restore_cfg(ik):
        c = ik.cfg
        c.damping, c.pos_gain, c.ori_gain, c.null_gain, c.max_dq = c._saved

    def _settle_pose(self, q9, hand_pose, settle_steps=800):
        # 物理静置到 (q9, 指定手型)，返回 (指令site位姿, 实际site位姿, 实际握点局部坐标)
        m, d = self.ik.model, self.ik.data
        ik = self.ik
        off = self.off
        prefix = "rh" if self.side == "right" else "lh"
        tip_names = [f"{prefix}_ffdistal", f"{prefix}_mfdistal",
                     f"{prefix}_rfdistal", f"{prefix}_lfdistal",
                     f"{prefix}_thdistal"]
        tip_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                   for n in tip_names]
        saved_q = d.qpos.copy()
        saved_v = d.qvel.copy()
        # 标定期间禁用笔的所有碰撞（否则自由落体砸手会污染静置结果/炸仿真）
        pen_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pen")
        saved_mask = []
        for gi in range(m.ngeom):
            if m.geom(gi).bodyid == pen_b:
                saved_mask.append((gi, int(m.geom(gi).contype[0]),
                                   int(m.geom(gi).conaffinity[0])))
                m.geom(gi).contype[:] = 0
                m.geom(gi).conaffinity[:] = 0
        # 指令 FK 位姿
        d.qpos[ik.qadr] = q9
        mujoco.mj_kinematics(m, d)
        p_cmd = d.site(ik.site_id).xpos.copy()
        R_cmd = d.site(ik.site_id).xmat.reshape(3, 3).copy()
        # 笔移开、手置位
        d.qpos[0:3] = [0.0, 0.0, 2.0]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[:] = 0.0
        hand_jadr = ik.qadr[0] + 7
        for j in range(m.njnt):
            qa = m.jnt_qposadr[j]
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE and hand_jadr <= qa < hand_jadr + 24:
                d.qpos[qa] = 0.0
        d.qpos[ik.qadr] = q9
        pose = self.poses[hand_pose]
        d.ctrl[off:off + 7] = q9[:7]
        d.ctrl[off + 7:off + 9] = q9[7:9]
        d.ctrl[off + 9:off + 27] = pose[2:20]
        for _ in range(settle_steps):
            # 与 env._step_rad 一致的重力/科氏前馈（实测符号为 +qfrc_bias），
            # 否则静置测出的"下垂"是执行环境中不存在的伪下垂，补偿会过度修正
            mujoco.mj_forward(m, d)
            d.qfrc_applied[6:] = d.qfrc_bias[6:]
            mujoco.mj_step(m, d)
        d.qfrc_applied[6:] = 0.0
        p_act = d.site(ik.site_id).xpos.copy()
        R_act = d.site(ik.site_id).xmat.reshape(3, 3).copy()
        tips = np.array([d.body(b).xpos for b in tip_ids])
        # 0..3=食/中/无/小，4=拇；按手型定义对捏中心
        if hand_pose in ("pinch2",):
            fcenter = tips[0]
        elif hand_pose in ("pinch3",):
            fcenter = tips[:2].mean(axis=0)
        else:
            fcenter = tips[:4].mean(axis=0)
        g_world = 0.5 * (fcenter + tips[4])
        g_loc = R_act.T @ (g_world - p_act)
        d.qpos[:] = saved_q
        d.qvel[:] = saved_v
        for gi, ct, ca in saved_mask:
            m.geom(gi).contype[:] = ct
            m.geom(gi).conaffinity[:] = ca
        mujoco.mj_kinematics(m, d)
        return p_cmd, R_cmd, p_act, R_act, g_loc

    def _compensated_target(self, q9, hand_pose, C, R_des):
        # 不动点标定：静置（含桌面接触）后直接读"实际握点"g_world，
        # 把 site 指令平移 (C - g_world)，迭代到实际握点落在 C。
        # 注意：不能用 site 下垂量补偿——重力补偿下下垂≈0，实测位移主要是
        # 手指被桌面顶住产生的平衡位移，按下垂反向补偿只会越压越深。
        p_cmd, R_cmd, p_act, R_act, g_loc = self._settle_pose(q9, hand_pose)
        g_world = p_act + R_act @ g_loc
        # 桌面硬约束：若某个方向（典型为 -z）实际握点已无法靠近 C，
        # 该方向不做修正，避免把指令压入桌面导致整臂卡死。
        delta = C - g_world
        # 每轮修正限幅，防止 open 手型下 g_world 飘忽导致发散
        nd = np.linalg.norm(delta)
        if nd > 0.05:
            delta *= 0.05 / nd
        tol = 0.005
        blocked = getattr(self, "_blocked_axes", None)
        if blocked is None:
            blocked = np.zeros(3, dtype=bool)
        else:
            blocked = blocked | (np.abs(delta) > np.abs(getattr(self, "_last_delta", delta)) + tol)
        delta = np.where(blocked, 0.0, delta)
        self._blocked_axes = blocked
        self._last_delta = C - g_world
        p_cmd_need = p_cmd + delta
        # 总偏移限幅：指令握点偏离目标最多 12cm（超出即标定失败，回退到未补偿目标）
        off = p_cmd_need - C
        no = np.linalg.norm(off)
        if no > 0.12:
            p_cmd_need = C + off * (0.12 / no)
        rp = float(np.linalg.norm(C - g_world))
        cos = np.clip((np.trace(R_des.T @ R_act) - 1.0) / 2.0, -1.0, 1.0)
        ro = float(np.arccos(cos))
        return p_cmd_need, R_des, rp, ro

    def _select_hand_mode(self, obs):
        pen_pos, pen_q = self._pen_pose(obs)
        axis = quat_to_R(pen_q) @ np.array([0.0, 0.0, 1.0])
        axis /= np.linalg.norm(axis)
        anchor = (pen_pos.copy(), axis.copy())
        st = self.stages[1]
        # 8 候选：粗解 -> 2 轮物理静置补偿 -> 按补偿后可达性排名
        self.cand_scores = {}
        best_key, best_score = None, np.inf
        a, svec = self._axes(axis)
        C = pen_pos + a * st["along"] + svec * st["lateral"] + _UP * st["vertical"]
        calib_pose = self.stages[1]["hand"]  # descend 实际手型（张开），保证执行几何一致
        for hand, ik in self.iks.items():
            self.side, self.ik = hand, ik
            self.off = 27 if hand == "right" else 0
            for mode in self.grasp_modes:
                R, _ = self._grasp_frame(axis, mode)
                q, _, _ = self._solve_robust(ik, C, R)
                rp, ro, p_need, R_need = np.inf, np.inf, C, R
                self._blocked_axes = None
                for _ in range(3):
                    p_need, R_need, rp, ro = self._compensated_target(q, calib_pose, C, R)
                    q, _, _ = self._solve_from(ik, q, p_need, R_need)
                # 按实际静置握点残差排名（含桌面硬约束），而非 IK 残差
                margin = np.minimum(q - ik.lo, ik.hi - q).min()
                score = rp + self.ori_weight * ro + 0.05 * max(0.0, 0.1 - margin)
                self.cand_scores[(hand, mode)] = score
                if score < best_score:
                    best_score, best_key = score, (hand, mode)
                    best_q, best_t = q, (p_need, R_need)
        self.side, self.mode = best_key
        self.ik = self.iks[self.side]
        self.off = 27 if self.side == "right" else 0
        self._q_desc = best_q
        self._t_desc = best_t
        self._desc_C = C.copy()

    def _begin_stage(self, obs):
        st = self.stages[self.si]
        self.n = max(1, int(round(st["duration"] * self.hz)))
        pen_pos, pen_q = self._pen_pose(obs)
        axis = quat_to_R(pen_q) @ np.array([0.0, 0.0, 1.0])
        axis /= np.linalg.norm(axis)
        if st["reference"] == "anchor" and self._anchor is None:
            self._anchor = (pen_pos.copy(), axis.copy())
        tR0 = self._grasp_frame(axis, self.mode)[0]
        a, svec = self._axes(axis)
        if st["reference"] == "anchor":
            ref_pos, ref_axis = self._anchor
        else:
            ref_pos, ref_axis = pen_pos, axis
        C = ref_pos + a * st["along"] + svec * st["lateral"] + _UP * st["vertical"]
        self.q_start = self._current_q9()
        self.f_start = self._current_fingers()
        if st["name"] in ("descend", "close"):
            self.q_goal = self._q_desc.copy()
            p_need, R_need = self._t_desc
            ep = eo = 0.0
        else:
            seed = self._q_desc if self.si == 0 else self._prev_goal
            self.q_goal, _, _ = self._solve_from(self.ik, seed, C, tR0)
            ep, eo, p_need, R_need = np.inf, np.inf, C, tR0
            self._blocked_axes = None
            for _ in range(3):
                p_need, R_need, rp, ro = self._compensated_target(
                    self.q_goal, st["hand"], C, tR0)
                self.q_goal, ep, eo = self._solve_from(self.ik, self.q_goal,
                                                       p_need, R_need)
            ep, eo = rp, ro
        self.f_goal = self._fingers_goal(st["hand"])
        self.plan_errors.append((st["name"], ep, eo, p_need, R_need))
        self._prev_goal = self.q_goal.copy()
        self._stage_C = C.copy()
        self._stage_R = tR0
        self._stage_s = svec.copy()
        self._stage_name = st["name"]
        if st["name"] == "descend":
            self._react_o = 0.0
            self._react_locked = False
            self._react_t = 0
            self._desc_stopped = False

    def _pen_contact_sides(self):
        # 返回 (拇指触笔, 食/中指触笔)
        d = self.env.data
        m = self.env.model
        p = self.side[0]  # r / l
        th_n = f"{p}h_thdistal"
        ff_n = (f"{p}h_ffdistal", f"{p}h_mfdistal")
        th = fin = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom(c.geom1).bodyid)
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom(c.geom2).bodyid)
            other = None
            if b1 == "pen":
                other = b2
            elif b2 == "pen":
                other = b1
            if other is None:
                continue
            if other == th_n:
                th = True
            if other in ff_n:
                fin = True
        return th, fin

    def _side_table_contact(self):
        # 本手任一 geom 触桌（world）即 True（descend 下探过头/偏移压桌的信号）
        d = self.env.data
        m = self.env.model
        pre = self.side[0] + "h_"
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom(c.geom1).bodyid)
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom(c.geom2).bodyid)
            if b1 == "world" and b2 and b2.startswith(pre):
                return True
            if b2 == "world" and b1 and b1.startswith(pre):
                return True
        return False

    # ------------------------------------------------------------------ policy
    def act(self, obs):
        u = _min_jerk(self.k / self.n)
        q9 = (1 - u) * self.q_start + u * self.q_goal
        finger_v = (1 - u) * self.f_start + u * self.f_goal
        # 接触反应式横向微调（仅 close/lift/hold）：
        # 都没碰到 -> 慢速正弦扫描扩大捕捉域；单侧碰到 -> 朝该侧对中；双侧 -> 锁定。
        sname = self.stages[self.si]["name"]
        if sname in ("close", "lift", "hold"):
            # 迟滞反馈：close 后半段、连续 3 帧仅单侧触笔才小幅对中；双侧即锁定。
            if sname == "close" and not self._react_locked and self.k > 0.4 * self.n:
                th, fin = self._pen_contact_sides()
                runs = self._react_side_runs
                runs[0] = runs[0] + 1 if (th and not fin) else 0
                runs[1] = runs[1] + 1 if (fin and not th) else 0
                if th and fin:
                    self._react_locked = True
                else:
                    step = float(self.cfg.get("react_step", 0.003))
                    cap = float(self.cfg.get("react_cap", 0.012))
                    if runs[0] >= 3:
                        self._react_o = float(np.clip(self._react_o + step, -cap, cap))
                        runs[0] = 0
                    elif runs[1] >= 3:
                        self._react_o = float(np.clip(self._react_o - step, -cap, cap))
                        runs[1] = 0
            if abs(self._react_o) > 1e-6:
                p_target = self._stage_C + self._stage_s * self._react_o
                q9, _, _ = self._solve_from(self.ik, q9, p_target, self._stage_R,
                                            iters=25)
        # 接触引导下降：descend 后段一旦手指碰笔或本手触桌，就地冻结当前位姿
        # （其余插值目标=当前值），下一帧提前进入 close，避免压桌/扫飞
        sname_desc = self.stages[self.si]["name"]
        if (sname_desc == "descend" and self.cfg.get("desc_pen_stop", False)
                and not getattr(self, "_desc_stopped", False)
                and self.k > int(0.5 * self.n)):
            th_c, fin_c = self._pen_contact_sides()
            if th_c or fin_c:
                # 可选：手指碰到笔即冻结深度提前合拢。实测净负收益（高处闭合抓笔顶），
                # 默认关闭，保留开关供实验。
                self._desc_stopped = True
                self.q_goal = q9.copy()
                self.f_goal = finger_v.copy()
                self.n = max(self.k, 1)
        hand_v = np.concatenate([q9[7:9], finger_v])
        arm_q = q9[:7]

        if self.side == "right":
            out = np.concatenate([self._home[0:7], self._home[7:27], arm_q, hand_v])
        else:
            out = np.concatenate([arm_q, hand_v, self._home[27:34], self._home[34:54]])

        self.k += 1
        if self.k >= self.n and self.si < len(self.stages) - 1:
            self.si += 1
            self.k = 0
            self._begin_stage(obs)
        return out


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
