# PostTraindex

高自由度双臂灵巧操作的 VLA 后训练 MVP（MuJoCo 仿真版）。

复现论文 *Towards High-DoF Dexterous Manipulation through VLA Post-Training*（Wuji Tech / 上海科大，arXiv 2609.19666）的工程流水线，并做两项核心简化：

- **VLA 用 ACT 替代 π0.5**（LeRobot 现成实现，单卡 RTX 4090 可训）
- **技能提升环节保留残差强化学习**：冻结 ACT 作为参考策略 π_ref，TD3 学习有界残差 Δa

## 论文流水线与 MVP 取舍

论文四步：手部动作 codec（20→9/手）→ π0.5 SFT → DAgger 修正 → latent 残差 RL。

| 论文章节 | MVP 决策 |
|---|---|
| π0.5 (3.35B, flow matching) | **换成 ACT**，放弃语言泛化（留接口） |
| Step 1 codec（20→9/手） | **跳过**，ACT 直接输出 54-D 关节 chunk；Phase-B 可选补 |
| Step 2 SFT | LeRobot ACT 微调，50–200 条仿真示教（脚本专家自动产数据） |
| Step 3 DAgger | **跳过**（仿真可用脚本专家自动接管，列为 Phase-B） |
| Step 4 latent 残差 RL | **保留**，简化为 raw 关节空间残差 TD3：冻结 ACT，actor 输出有界 Δa 加到 ACT chunk 上 |

平台全部用现成开源方案：MuJoCo 3.x + MuJoCo Menagerie（franka_panda ×2 + shadow_hand 左右手）+ LeRobot + TD3。

## 观测 / 动作接口（对齐论文）

- **观测**：3 路 224×224 RGB（头 + 双腕，MuJoCo offscreen 渲染）+ 54-D 执行器空间状态（`actuator_length`，腱长对齐动作量纲）+ 笔 13-D 位姿（RL 用）；完整关节状态经 `robot_qpos/qvel` 暴露
- **动作**：54-D 绝对关节位置指令（2×7 臂 + 2×20 手，归一化 [-1,1] 映射 ctrlrange），限幅 π/18 rad/帧
- **控制**：20 Hz（frame_skip=25），chunk H=32、执行 C=16 帧后重规划
- **成功判定**（仿真自动）：手接触笔且笔整体抬离桌面 3 cm，连续稳定 0.5 s
- **关键闸门**：ACT 基线成功率 ≥40% 才进入残差 RL（论文结论：<20% 时 RL 无效）

## 里程碑

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 | 环境与双臂 MuJoCo 场景（Gymnasium 环境、3 相机、成功判定、verify 脚本） | ✅ 已完成（2026-09-28，`scripts/verify_env.py` 7/7 PASS，单步 24.5 ms） |
| M1 | 三段式脚本示教 100–200 条 + LeRobot v3 数据集 | ⬜ |
| M2 | ACT 模仿学习，闭环成功率 ≥40% | ⬜ |
| M3 | 冻结 ACT + 残差 TD3（zero-init tanh 头、稀疏终局奖励、critic warm-up），成功率 ≥90%，产出 success-vs-transitions 曲线 | ⬜ |
| M4 | 可选 Phase-B：chunk-VAE codec + latent vs raw 残差对比 / Sim-DAgger / Transfer between Hands | ⬜ |

建议周期 4–6 周，单人 + 1×RTX 4090 级别 GPU。

## 目录结构

```
bimanual_dex_mvp/
├── assets/                  # Menagerie 拼装后的 MJCF 场景
│   └── bimanual_scene.xml
├── configs/                 # 任务配置（pick_up_marker.yaml 等）
├── envs/                    # Gymnasium 环境 BimanualDex-v0
├── teleop/                  # 三段式脚本示教（手型库 + IK + 接触微调）
├── policy/                  # (M2/M3) ACT 配置 / residual_td3.py
├── scripts/                 # gen_scene.py / verify_env.py / tune_*.py / eval_*.py
└── utils/                   # 坐标变换等工具
```

## 快速开始

```bash
pip install "mujoco>=3.2" gymnasium
python scripts/verify_env.py   # 环境自检（相机渲染 / 驱动开合 / 单步耗时）
```

## 偏离论文 / 原始 Menagerie 设定（登记备查）

> 原则：仅在「位置控制机器人 + 最快打通 MVP」需要时偏离。

1. Shadow Hand 为 24 关节 / 20 驱动器（J1/J2 腱耦合），本体共 62-DoF（nq=69 含笔）；动作仍 54-D。
2. 全部 DOF 注入 `+qfrc_bias` 重力/科氏前馈（等价工业臂重力补偿）。
3. 伺服加强：腕 kp 10/8→60/40，手指 kp 0.4–1.5→3–4（`scripts/gen_scene.py`）。
4. 关闭手部自碰撞（指尖对捏需要软组织贴合，硬 geom 自碰撞会封锁）。
5. 侧捏 `frame_tilt=−86.9°` 使拇指/食指指尖等高会合。
6. 脚本专家工程化：高位接近、慢速抬升、close 阶段接触反应式横向微调（示教手段，不改变任务与成功判定）。

## 参考资料

- 论文：*Towards High-DoF Dexterous Manipulation through VLA Post-Training*（arXiv 2609.19666）
- [LeRobot](https://github.com/huggingface/lerobot)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Dexora](https://github.com/dexoravla/Dexora)（双臂灵巧 VLA，MuJoCo 数据管线参考）
- [RoboPianist](https://github.com/google-research/robopianist)（双 Shadow Hand MuJoCo 工程参考）
- [ACT](https://github.com/tonyzhaozh/act)
