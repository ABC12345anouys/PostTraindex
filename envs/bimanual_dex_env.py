"""Gymnasium 环境：双 Panda + 双 Shadow Hand（MuJoCo），任务 Pick Up Marker。

动作空间
  (54,) float32，归一化 [-1, 1]，按 actuator ctrlrange 映射为**绝对关节位置
  指令**；相邻控制帧之间在弧度空间限幅（默认 π/18 rad/帧，对齐论文安全约束）。
  执行器顺序：[左臂7, 左手20, 右臂7, 右手20]。

观测空间（Dict）
  state      (54,)  执行器空间位置（手臂=关节角，手=腱长度 J1+J2，单位 rad）
  state_vel  (54,)  执行器空间速度（rad/s）
  object     (13,)  笔：位置3 + 四元数4(wxyz) + 线速度3 + 角速度3
  images     Dict   head / left_wrist / right_wrist 三路 224x224x3 uint8
                    （仅 render_mode="rgb_array" 时渲染）

注意（相对论文 54-DoF 的差异，见 PLAN/PROGRESS）
  Shadow Hand 每只手 24 个关节 / 20 个驱动器（J1/J2 由固定腱耦合），
  故机器人本体状态为 62-DoF，但 action / 执行器空间反馈为 54-D，与论文接口对齐。
"""

import os
from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCENE = os.path.join(ROOT, "assets", "bimanual_scene.xml")
TABLE_TOP_Z = 0.50
PEN_RADIUS = 0.008
PEN_HALF_LEN = 0.075

# 执行器分组顺序（与 XML 一致）
ARM_L = slice(0, 7)
HAND_L = slice(7, 27)
ARM_R = slice(27, 34)
HAND_R = slice(34, 54)


@dataclass
class BimanualDexConfig:
    scene_path: str = DEFAULT_SCENE
    control_hz: float = 20.0            # 论文/MVP：20 Hz
    image_size: int = 224
    cameras: tuple = ("head", "left_wrist", "right_wrist")
    action_rate_limit: float = np.pi / 18   # rad / 控制帧
    max_episode_seconds: float = 10.0
    # reset 随机化（笔在桌面上的位置/朝向范围）
    pen_xy: tuple = ((-0.12, 0.12), (-0.18, 0.18))
    settle_seconds: float = 0.4        # 落笔后静置
    reset_max_tries: int = 20
    # 成功判定：笔最低点离桌沿 + 稳定持握帧数
    lift_clearance: float = 0.03       # 笔整体抬离桌面 3 cm
    max_lin_vel: float = 0.20          # 持握稳定线速度阈值 m/s
    hold_frames: int = 10              # 连续 0.5 s
    # 掉落/越界失败
    fall_z: float = 0.35
    xy_limit: float = 0.55
    reward_type: str = "sparse"        # "sparse"（论文：成功0/失败-1）或 "dense"（偏离项）
    init_jitter: float = 0.0           # reset 时机器人关节微小扰动（rad），0=严格 home
    # 偏离论文标注：对机器人全部 DOF 注入 qfrc_bias（重力+科氏，本版本符号约定）前馈，等效
    # 工业臂标配的重力补偿，消除弱位置伺服的静态下垂；笔不补偿。
    gravity_comp: bool = True
    # 偏离论文/Menagerie 标注：关闭手部 geom 之间的碰撞（手-桌/笔/臂仍碰撞）。
    # Shadow 硬 geom 自碰撞会阻止人类可实现的指尖对捏（真实软组织可贴合），
    # 是桌面细物对捏不可达的主要原因。手部 geom：contype=2/conaffinity=1，
    # 其余保持 1/1，于是手只与 bit1（桌/笔/臂）碰撞、不与其他手 geom 碰撞。
    hand_selfcollide: bool = False


class BimanualDexEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config: BimanualDexConfig | None = None,
                 render_mode: str | None = None):
        super().__init__()
        self.cfg = config or BimanualDexConfig()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(self.cfg.scene_path)
        self.data = mujoco.MjData(self.model)
        assert self.model.nu == 54, f"expect 54 actuators, got {self.model.nu}"

        self.frame_skip = int(round(1.0 / (self.cfg.control_hz * self.model.opt.timestep)))
        assert abs(self.frame_skip * self.model.opt.timestep - 1.0 / self.cfg.control_hz) < 1e-6

        self._build_index()
        if not self.cfg.hand_selfcollide:
            self._disable_hand_selfcollision()
        self._key_home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._pen_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pen_geom")
        self._pen_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pen")

        self.action_space = spaces.Box(-1.0, 1.0, shape=(54,), dtype=np.float32)
        state_box = spaces.Box(-np.inf, np.inf, shape=(54,), dtype=np.float32)
        obs = {
            "state": state_box,
            "state_vel": state_box,
            "object": spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32),
        }
        if render_mode == "rgb_array":
            img = spaces.Box(0, 255, shape=(self.cfg.image_size, self.cfg.image_size, 3),
                             dtype=np.uint8)
            obs["images"] = spaces.Dict({c: img for c in self.cfg.cameras})
            self._renderer = mujoco.Renderer(self.model, self.cfg.image_size,
                                             self.cfg.image_size)
            self._cam_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c)
                             for c in self.cfg.cameras]
        self.observation_space = spaces.Dict(obs)

        self.max_steps = int(self.cfg.max_episode_seconds * self.cfg.control_hz)
        self.step_count = 0
        self._hold_count = 0
        self._target = np.zeros(54)          # 当前弧度制位置指令
        self._home_target = self.model.key_ctrl[self._key_home].copy()
        self._first_obs = None

    # --------------------------------------------------------------- indices
    def _disable_hand_selfcollision(self):
        # bit2 仅戴在手上；手的 affinity 只含 bit1，故手-手不碰，手-环境正常。
        for gi in range(self.model.ngeom):
            bid = self.model.geom(gi).bodyid
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith(("lh_", "rh_")):
                self.model.geom(gi).contype[:] = 2
                self.model.geom(gi).conaffinity[:] = 1

    def _build_index(self):
        m = self.model
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        expect = [f"l_actuator{i}" for i in range(1, 8)] + \
                 ["lh_A_WRJ2", "lh_A_WRJ1", "lh_A_THJ5", "lh_A_THJ4", "lh_A_THJ3",
                  "lh_A_THJ2", "lh_A_THJ1", "lh_A_FFJ4", "lh_A_FFJ3", "lh_A_FFJ0",
                  "lh_A_MFJ4", "lh_A_MFJ3", "lh_A_MFJ0", "lh_A_RFJ4", "lh_A_RFJ3",
                  "lh_A_RFJ0", "lh_A_LFJ5", "lh_A_LFJ4", "lh_A_LFJ3", "lh_A_LFJ0"] + \
                 [f"r_actuator{i}" for i in range(1, 8)] + \
                 [n.replace("lh_", "rh_") for n in
                  ["lh_A_WRJ2", "lh_A_WRJ1", "lh_A_THJ5", "lh_A_THJ4", "lh_A_THJ3",
                   "lh_A_THJ2", "lh_A_THJ1", "lh_A_FFJ4", "lh_A_FFJ3", "lh_A_FFJ0",
                   "lh_A_MFJ4", "lh_A_MFJ3", "lh_A_MFJ0", "lh_A_RFJ4", "lh_A_RFJ3",
                   "lh_A_RFJ0", "lh_A_LFJ5", "lh_A_LFJ4", "lh_A_LFJ3", "lh_A_LFJ0"]]
        assert names == expect, "actuator order mismatch:\n%r" % names
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
        self.ctrl_mid = 0.5 * (self.ctrl_lo + self.ctrl_hi)
        self.ctrl_half = 0.5 * (self.ctrl_hi - self.ctrl_lo)

        # 手部碰撞体集合（用于抓持判定）
        self._hand_geom_body_ids = set()
        for gid in range(m.ngeom):
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom(gid).bodyid)) or ""
            if bn.startswith(("lh_", "rh_")):
                self._hand_geom_body_ids.add(int(m.geom(gid).bodyid))

        # 机器人 qpos/qvel 切片（62-DoF 本体状态，供专家/IK 使用）
        def joints(prefix):
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_joint{i}")
                   for i in range(1, 8)]
            return ids
        self._arm = {}
        for side, pfx in (("l", "l"), ("r", "r")):
            ids = joints(pfx)
            self._arm[side] = (
                np.array([m.jnt_qposadr[i] for i in ids]),
                np.array([m.jnt_dofadr[i] for i in ids]),
            )
        self._hand_qpos = {}
        for pfx in ("lh", "rh"):
            ids = [j for j in range(m.njnt)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(pfx + "_")]
            ids.sort(key=lambda j: m.jnt_qposadr[j])
            self._hand_qpos[pfx] = (
                np.array([m.jnt_qposadr[j] for j in ids]),
                np.array([m.jnt_dofadr[j] for j in ids]),
            )

    @property
    def robot_qpos(self) -> np.ndarray:
        """62-D 机器人关节位置：[左臂7, 左手24, 右臂7, 右手24]。"""
        d = self.data
        al, _ = self._arm["l"]; ar, _ = self._arm["r"]
        hl, _ = self._hand_qpos["lh"]; hr, _ = self._hand_qpos["rh"]
        return np.concatenate([d.qpos[al], d.qpos[hl], d.qpos[ar], d.qpos[hr]])

    @property
    def robot_qvel(self) -> np.ndarray:
        d = self.data
        _, al = self._arm["l"]; _, ar = self._arm["r"]
        _, hl = self._hand_qpos["lh"]; _, hr = self._hand_qpos["rh"]
        return np.concatenate([d.qvel[al], d.qvel[hl], d.qvel[ar], d.qvel[hr]])

    @property
    def home_action_rad(self) -> np.ndarray:
        """home 关键帧对应的弧度制指令（手臂 home、手张开）。"""
        return self._home_target.copy()

    # --------------------------------------------------------------- helpers
    def _pen_pose(self):
        d = self.data
        return np.concatenate([d.body(self._pen_body).xpos,
                               d.body(self._pen_body).xquat,
                               d.qvel[0:3], d.qvel[3:6]])

    def _pen_lowest_z(self) -> float:
        """笔几何体最低点的世界坐标 z（考虑任意姿态）。"""
        d = self.data
        center = d.body(self._pen_body).xpos
        R = d.body(self._pen_body).xmat.reshape(3, 3)
        axis_z = abs(R[2, 2])                       # 圆柱局部轴（z）的世界竖直分量
        half_vertical = PEN_HALF_LEN * axis_z + PEN_RADIUS * np.sqrt(1.0 - axis_z ** 2)
        return float(center[2] - half_vertical)

    def _hand_touching_pen(self) -> bool:
        m, d = self.model, self.data
        for c in d.contact[: d.ncon]:
            b1, b2 = int(m.geom(c.geom1).bodyid), int(m.geom(c.geom2).bodyid)
            if (b1 == self._pen_body and b2 in self._hand_geom_body_ids) or \
               (b2 == self._pen_body and b1 in self._hand_geom_body_ids):
                return True
        return False

    def _pen_on_table(self) -> bool:
        d = self.data
        z = d.body(self._pen_body).xpos[2]
        return TABLE_TOP_Z - 0.02 < z < TABLE_TOP_Z + 0.05

    def get_obs(self):
        d = self.data
        obs = {
            "state": d.actuator_length.astype(np.float32),
            "state_vel": d.actuator_velocity.astype(np.float32),
            "object": self._pen_pose().astype(np.float32),
        }
        if self.render_mode == "rgb_array":
            obs["images"] = self._render_images()
        return obs

    def _render_images(self):
        imgs = {}
        for name, cid in zip(self.cfg.cameras, self._cam_ids):
            self._renderer.update_scene(self.data, camera=cid)
            imgs[name] = self._renderer.render().copy()
        return imgs

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self._render_images()

    # --------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.cfg
        for _ in range(cfg.reset_max_tries):
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._key_home)
            d = self.data
            if cfg.init_jitter > 0:
                al, _ = self._arm["l"]; ar, _ = self._arm["r"]
                d.qpos[al] += self.np_random.uniform(-cfg.init_jitter, cfg.init_jitter, size=7)
                d.qpos[ar] += self.np_random.uniform(-cfg.init_jitter, cfg.init_jitter, size=7)
            self._randomize_pen(d)
            mujoco.mj_forward(self.model, d)
            # 静置落笔（手臂保持 home）
            d.ctrl[:] = self._home_target
            for _ in range(int(cfg.settle_seconds / self.model.opt.timestep)):
                mujoco.mj_step(self.model, d)
            if self._pen_on_table() and not self._hand_touching_pen():
                break
        else:
            raise RuntimeError("reset: 无法生成合法的笔位置（检查随机范围/碰撞）")

        self.step_count = 0
        self._hold_count = 0
        self._target = d.actuator_length.copy()
        d.ctrl[:] = self._target
        obs = self.get_obs()
        return obs, {"pen_pose": self._pen_pose()}

    def _randomize_pen(self, d):
        (xlo, xhi), (ylo, yhi) = self.cfg.pen_xy
        x = self.np_random.uniform(xlo, xhi)
        y = self.np_random.uniform(ylo, yhi)
        yaw = self.np_random.uniform(0.0, 2 * np.pi)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        # q_yaw @ q_flat（q_flat 把圆柱轴 z 转到 x 平放）
        qflat = np.array([0.7071068, 0.0, 0.7071068, 0.0])  # wxyz
        qyaw = np.array([c, 0.0, 0.0, s])
        quat = self._quat_mul(qyaw, qflat)
        d.qpos[0:3] = [x, y, TABLE_TOP_Z + PEN_RADIUS + 0.002]
        d.qpos[3:7] = quat
        d.qvel[0:6] = 0.0

    @staticmethod
    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    # --------------------------------------------------------------- step
    def step(self, action):
        return self._step_rad(self._action_to_rad(np.asarray(action, dtype=np.float32)))

    def step_rad(self, target_rad):
        """弧度制绝对位置指令接口（供脚本专家/teleop 使用，同样限幅）。"""
        return self._step_rad(np.asarray(target_rad, dtype=np.float64))

    def _action_to_rad(self, a):
        a = np.clip(a, -1.0, 1.0)
        return self.ctrl_mid + a * self.ctrl_half

    def _step_rad(self, desired):
        cfg = self.cfg
        desired = np.clip(desired, self.ctrl_lo, self.ctrl_hi)
        delta = np.clip(desired - self._target, -cfg.action_rate_limit, cfg.action_rate_limit)
        self._target = np.clip(self._target + delta, self.ctrl_lo, self.ctrl_hi)

        d = self.data
        d.ctrl[:] = self._target
        for _ in range(self.frame_skip):
            if cfg.gravity_comp:
                # 实测（mj_inverse 验证）：保持该状态所需关节力矩 == +qfrc_bias
                # （本 MuJoCo 版本 bias 符号约定），故前馈 +bias；上一子步值滞后 0.4ms
                d.qfrc_applied[6:] = d.qfrc_bias[6:]
            mujoco.mj_step(self.model, d)
        self.step_count += 1

        pen = d.body(self._pen_body).xpos
        grasped = self._hand_touching_pen()
        lifted = self._pen_lowest_z() > TABLE_TOP_Z + cfg.lift_clearance
        stable = np.linalg.norm(d.qvel[0:3]) < cfg.max_lin_vel
        if grasped and lifted and stable:
            self._hold_count += 1
        else:
            self._hold_count = 0

        success = self._hold_count >= cfg.hold_frames
        fell = pen[2] < cfg.fall_z or abs(pen[0]) > cfg.xy_limit or abs(pen[1]) > cfg.xy_limit
        timeout = self.step_count >= self.max_steps
        terminated = bool(success or fell)
        truncated = bool(timeout and not terminated)

        if cfg.reward_type == "sparse":
            # 论文设定：成功 0；其余终局（掉落/超时）-1；中间步 0
            reward = 0.0 if success else (-1.0 if (fell or timeout) else 0.0)
        else:
            # 偏离论文的提速项：高度/接触 shaping（M3 消融时与 sparse 区分报告）
            lift_frac = np.clip((self._pen_lowest_z() - TABLE_TOP_Z) / cfg.lift_clearance, 0, 1)
            reward = 0.4 * lift_frac + (0.3 if grasped else 0.0)
            if success: reward += 1.0
            if fell: reward -= 1.0

        info = {
            "success": success, "grasped": grasped, "lifted": lifted,
            "pen_z": float(pen[2]), "hold": self._hold_count,
        }
        return self.get_obs(), float(reward), terminated, truncated, info

    def close(self):
        if hasattr(self, "_renderer"):
            self._renderer.close()
