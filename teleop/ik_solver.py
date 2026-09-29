"""7-DoF Panda 臂（+2-DoF Shadow 腕）阻尼最小二乘（DLS）逆运动学。

每次控制步调用 `ArmIK.solve`，在当前 qpos 附近做若干次 DLS 迭代，
带 ready 位姿零空间偏置（保持自然构型、避免翻转）。默认把 2 个腕关节也纳入
求解（9 维），让笔轴朝向所需的 roll 主要由手腕承担。
"""
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKConfig:
    damping: float = 0.08
    pos_gain: float = 0.9
    ori_gain: float = 0.6
    null_gain: float = 0.2
    iters_per_step: int = 4
    max_dq: float = 0.25          # 单次迭代关节步长上限（rad）
    pos_tol: float = 0.005
    ori_tol_deg: float = 5.0

    def __post_init__(self):
        self.ori_tol = np.radians(self.ori_tol_deg)


def quat_to_R(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def axis_angle_between(R_cur, R_des):
    """旋转误差（世界系）：0.5 * sum(xc×xd + yc×yd + zc×zd)。

    符号与 mj_jacSite 的角速度雅可比配套：qdot 沿 J^T e 方向可使当前姿态收敛到目标。
    """
    return 0.5 * (
        np.cross(R_cur[:, 0], R_des[:, 0])
        + np.cross(R_cur[:, 1], R_des[:, 1])
        + np.cross(R_cur[:, 2], R_des[:, 2])
    )


class ArmIK:
    def __init__(self, env, side: str, cfg: IKConfig | None = None,
                 include_wrist: bool = True, wrist_seed=(0.0, 0.0)):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.cfg = cfg or IKConfig()
        self.include_wrist = include_wrist
        pfx_arm, pfx_hand = (("l", "lh") if side == "left" else ("r", "rh"))
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, f"{pfx_hand}_grasp_site")
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                  f"{pfx_arm}_joint{i}") for i in range(1, 8)]
        if include_wrist:
            jids += [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                       f"{pfx_hand}_WRJ2"),
                     mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                       f"{pfx_hand}_WRJ1")]
        self.qadr = np.array([self.model.jnt_qposadr[i] for i in jids])
        self.dadr = np.array([self.model.jnt_dofadr[i] for i in jids])
        self.lo = self.model.jnt_range[jids, 0]
        self.hi = self.model.jnt_range[jids, 1]
        # ready 零空间目标（臂=home 关键帧；腕=配置名义位）
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.q_seed = self.model.key_qpos[key][self.qadr].copy()
        if include_wrist:
            self.q_seed[7:9] = np.asarray(wrist_seed, dtype=np.float64)
        self.n = len(self.qadr)
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def current_pose(self):
        d = self.data
        return d.site(self.site_id).xpos.copy(), d.site(self.site_id).xmat.reshape(3, 3).copy()

    def solve(self, target_pos, target_R, iters: int | None = None):
        """在线 IK：从当前 qpos 出发逼近目标，返回 n 维绝对关节目标（默认 9 维）。"""
        m, d = self.model, self.data
        iters = iters or self.cfg.iters_per_step
        orig_qpos = d.qpos.copy()
        try:
            for _ in range(iters):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                self._jacp.fill(0.0); self._jacr.fill(0.0)
                mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self.site_id)
                J = np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])
                p = d.site(self.site_id).xpos
                R = d.site(self.site_id).xmat.reshape(3, 3)
                ep = np.asarray(target_pos) - p
                eo = axis_angle_between(R, np.asarray(target_R))
                if np.linalg.norm(ep) < self.cfg.pos_tol and \
                   np.linalg.norm(eo) < self.cfg.ori_tol:
                    break
                e = np.concatenate([self.cfg.pos_gain * ep, self.cfg.ori_gain * eo])
                Jt = J.T
                JJt = J @ Jt + (self.cfg.damping ** 2) * np.eye(6)
                dq = Jt @ np.linalg.solve(JJt, e)
                # ready 零空间偏置
                qn = d.qpos[self.qadr]
                dq += (np.eye(self.n) - Jt @ np.linalg.solve(JJt, J)) @ \
                      (self.cfg.null_gain * (self.q_seed - qn))
                mag = np.linalg.norm(dq)
                if mag > self.cfg.max_dq:
                    dq *= self.cfg.max_dq / mag
                d.qpos[self.qadr] = np.clip(qn + dq, self.lo + 1e-4, self.hi - 1e-4)
                mujoco.mj_kinematics(m, d)
            return d.qpos[self.qadr].copy()
        finally:
            d.qpos[:] = orig_qpos
            mujoco.mj_kinematics(m, d)

    def errors(self, target_pos, target_R):
        """不修改状态，返回当前 (位置误差 m, 姿态误差角 rad)。"""
        m, d = self.model, self.data
        orig = d.qpos.copy()
        try:
            mujoco.mj_kinematics(m, d)
            p = d.site(self.site_id).xpos
            R = d.site(self.site_id).xmat.reshape(3, 3)
            ep = np.linalg.norm(np.asarray(target_pos) - p)
            eo = np.linalg.norm(axis_angle_between(R, np.asarray(target_R)))
            return ep, eo
        finally:
            d.qpos[:] = orig
            mujoco.mj_kinematics(m, d)
