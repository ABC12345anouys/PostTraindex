# MVP 复现计划：高自由度双臂灵巧操作的 VLA 后训练流水线（MuJoCo 仿真版）

> 参考论文：`wuji6-DOF-1.pdf` — *Towards High-DoF Dexterous Manipulation through VLA Post-Training*（Wuji Tech / 上海科大，arXiv 2609.19666）
> 目标：在 MuJoCo 中复现论文的四步后训练流水线，VLA 用 **ACT** 替代 π0.5，skill 提升环节用**残差强化学习**，双臂灵巧平台全部采用现成开源方案，以最快路径做出 MVP。
---

## 1. 论文流程梳理

论文的四步流水线（真实机器人：天机 7-DoF 双臂 × 2 + Wuji 20-DoF 灵巧手 × 2，共 54 DoF，30 Hz 控制）：
| 步骤 | 内容 | 关键设计 |
| **Step 1: 手部动作 Codec** | 每只手 20-D 关节动作 chunk（H=32）压缩到 9-D latent（chunk 级 VAE，时序 Transformer 编解码器） | 双手 9×2 + 双臂 7×2 = **32-D latent 动作空间**，与 π0.5 动作接口对齐；Huber + 时序差分损失（k=0,1,2）+ KL |
| **Step 2: SFT** | π0.5 flow-matching VLA 微调为输出 32-D latent chunk（H=32，执行前 C=25 帧后重新观测） | latent 空间 flow matching + 解码后关节空间辅助损失（λ_arm=1.0, λ_lat=0.25, λ_dec=0.75） |
| **Step 3: DAgger** | 人工接管修正，回滚 5 s 缓冲、位姿对齐、2 s 指令混合，避免手势跳变污染修正数据 | 每轮 50 条修正，α_new=0.5，从 π0.5 重新训 30k 步 |
| **Step 4: Latent Residual RL** | 冻结 π_ref，TD3 在 latent 空间学习有界残差 Δz（RL token 读出生成 2048-D 观测摘要） | zero-init tanh 输出层（Δz=0 即原策略）、任务 mask、residual 正则 + 时序平滑正则 + critic warm-up；3000 transitions 在线交互 |
任务：5 个马克笔操作任务（抓取、双手传递、手中旋转、拔帽、写字）。结果：SFT/DAgger/RL 逐级把每个任务推到 100%（20 次试验）。

## 2. MVP 范围与简化决策

MVP 只复现**骨架**：codec/SFT/DAgger/RL 四步 → 简化为 **ACT 模仿学习 + 残差 RL** 两步核心，其余作为可选扩展。

| 论文章节 | MVP 决策 | 理由 |
|---|---|---|
| π0.5 (3.35B, flow matching) | **换成 ACT**（用户指定） | LeRobot 有现成 ACT，单卡 4090 可训；放弃语言泛化，语言条件留作接口 |
| Step 1 codec（20→9/手） | **MVP 跳过**，ACT 直接输出 54-D 关节 chunk | ACT 没有 32-D 动作接口限制；codec 作为 Phase-B 可选项（论文核心对比 latent vs raw residual 时可补） |
| 相机（头 1 + 腕部 2） | 保留 3 路 RGB（MuJoCo offscreen 渲染） | 与论文观测一致，方便以后换 VLA |
| Step 2 SFT | LeRobot ACT 微调，50–200 条仿真示教 | 仿真里用脚本策略自动产数据 |
| Step 3 DAgger | **MVP 跳过**（仿真可用脚本专家做自动接管，列为 Phase-B） | 论文中 RL 单独就能把 50% 提到 100%，DAgger 不是达成 MVP 的关键路径 |
| Step 4 latent residual RL | **保留，简化为 raw 关节空间残差 TD3**：冻结 ACT，actor 输出有界 Δa 加到 ACT chunk 上 | 保留论文核心思想（冻结参考策略 + 有界残差 + 稀疏终局奖励），工程最简单 |
| RL token (822M 读出头) | 替换为轻量观测编码（关节状态 + 目标物体位姿，可选 ResNet 图像特征） | MVP 状态域 RL 足够验证流水线 |
| 奖励/成败标注 | **仿真自动判定**（无需人工标注） | 仿真最大红利 |
| 任务 | **Pick Up Marker**（MVP）→ Transfer between Hands（扩展） | 抓取是其余任务的前置 |
| 控制参数 | 20 Hz，chunk H=32，每 chunk 执行 C=16 帧后重规划 | 论文 H=32/C=25/30Hz，MVP 降低推理频率要求 |
| 动作空间 | 54-D（2×7 臂 + 2×20 手） | **M0 实测偏差**：Shadow Hand 为 24 关节/20 驱动器/手（J1/J2 固定腱耦合），故机器人本体 62-DoF（nq=69 含笔）；action 仍为 54-D，本体反馈用 `actuator_length`（腱长=J1+J2）对齐动作量纲 |

## 3. 技术选型（全部现成开源）

### 推荐组合
| 模块 | 选型 | 说明 |
|---|---|---|
| 学习框架 | **LeRobot**（HF） | ACT/Diffusion/π0 策略现成实现、数据集格式（v3）、内置 MuJoCo 支持、训练/评测脚本齐全 |
| 仿真器 | **MuJoCo 3.x** | 论文 RL 也偏 off-policy，MuJoCo 接触仿真质量高 |
| 机器人模型 | **MuJoCo Menagerie** 拼装：`franka_panda` ×2 + `shadow_hand` E3M5 左/右手 | Shadow Hand 20 驱动器/手，与论文一致；半天可拼好双臂场景 |
| 双臂灵巧参考项目 | **Dexora**（github.com/dexoravla/Dexora，ICRA 2026） | 全开源双臂灵巧 VLA：2×6-DoF 臂 + 2×12-DoF 手（36-DoF），12.2K 真机 + **100K MuJoCo 仿真轨迹**（LeRobot 格式），其 MuJoCo 场景搭建/数据管线可直接抄；备选参考：**RoboPianist**（Menagerie 双 Shadow Hand 的可运行 MuJoCo 工程） |
| RL 算法 | **TD3**（chunk 级残差） | 与论文一致；实现可基于 Stable Baselines3 或自写（actor/critic 各 ~1M 参数的小 MLP，自写更快对齐论文公式 18–21） |
| 示教数据生成 | 脚本策略（固定抓取姿态库 + mocap 关键帧） | 灵巧手抓取示教无法像夹爪那样纯脚本，用「预定义手型 + 接近-合拢-抬升」三段式 |

### 备选（不推荐作为主线）
- **Bi-DexHands**（PKU-MARL）：双臂灵巧 RL 任务库现成，但基于 Isaac Gym 而非 MuJoCo，违背仿真器约束。
- **GR00T N1.7**（NVIDIA，已接入 LeRobot）：VLA 本体更强，但定位人形、微调吃显存/数据格式转换麻烦，对 MVP 是过度工程。
- **MuJoCo Playground**（JAX/MJX）：自带 LEAP Hand 重定向等灵巧操作 RL 示例，可作为单臂灵巧子技能的备选训练场，但双臂场景需自己搭。

## 4. 实施步骤与里程碑

建议周期 **4–6 周**（单人、1×RTX 4090 级别 GPU 即可）。

### M0：环境与机器人场景（第 1 周）✅ 已完成（2026-09-28）

- [x] conda/pip 环境：`mujoco>=3.2`（实测 3.5.0）、`gymnasium` 1.1.1、`torch`/`lerobot`（M2 用现成 conda env）；EGL 离屏渲染已通（RTX 4090）
- [x] 用 Menagerie 的 franka_panda + shadow_hand（左右手）拼双臂 MuJoCo XML：桌子 + 马克笔（cylinder，平放）；双臂镜像布局（基座 ±0.62 m），CEM 求无碰撞 ready 位姿排除自/互碰撞
- [x] 定义 `BimanualDexEnv`（Gymnasium 接口，注册 `BimanualDex-v0`，见 [bimanual_dex_env.py](bimanual_dex_mvp/envs/bimanual_dex_env.py)）：
  - 观测：3 路 224×224 RGB（头 + 双腕）+ 54-D **执行器空间**状态（`actuator_length`）+ 笔 13-D 位姿（RL 用）；完整 62-D 关节状态经 `robot_qpos/qvel` 暴露
  - 动作：54-D 绝对关节位置指令（归一化 [-1,1] 映射 ctrlrange），20 Hz（frame_skip=25），限幅 π/18 rad/帧（实测一致）
  - reset：马克笔 xy/yaw 随机落位 + 0.4 s 静置 + 合法性重试；成功判定：手接触笔且笔整体抬离桌面 3 cm 且连续稳定 0.5 s（自动）
- [x] 验收：`scripts/verify_env.py` 7/7 PASS（驱动双手张开/闭合、3 路相机渲染、单步 24.5 ms < 50 ms 实时预算）；完整抓取动作链路由 M1 脚本专家验收

### M1：脚本示教 + LeRobot 数据集（第 2 周）

- [x] 三段式脚本专家：手型库（预录 2–3 个抓取手型）→ 臂接近（IK/路点）→ 合拢 → 抬升；失败自动重试（2026-09-29 固定笔位 success=True，笔抬至 0.61–0.67）
- [x] 录制 100–200 条成功 episode（LeRobot v3 格式，含三路视频），8:2 划分训练/验证（160 条 = train 130 / val 30）
- [x] 验收：数据集体检通过；脚本专家宽包络 ≥80% 未达，按偏离 9 改「好点白名单 9 点 + plan_seed 重试 + 脏成功过滤」采集（用户确认）

### M2：ACT 模仿学习（第 2–3 周）✅ 已完成（2026-09-30）

- [x] 用 LeRobot ACT 配置适配 54-D 动作（chunk H=32）：ResNet18×3 视角 + 54-D state + 13-D object，Transformer 动作头（52M 参数）
- [x] 训练至收敛：130 条 train、20k 步（bs8/lr1e-5，~32 min），loss 12.2→0.048
- [x] 仿真闭环评测（mujoco 3.7.0 同物理）：**rand20 12/20=60%**、白名单 9 点 7/9=78%（开 temporal ensemble 0.01；关闭时仅 1/9——chunk 边界复合误差）
- [x] 失败归因：首 chunk 与专家 L1 0.03 rad（拟合充分），失败均来自 chunk 边界 + 接触刀尖窗口；ensemble 后残余失败与专家同 seed 失败点重合

### M3：残差 RL（第 3–5 周）——论文 Step 4 的复现 ✅ 已完成（2026-09-30）

- [x] 冻结 ACT 为 π_ref（20k checkpoint + temporal ensemble 0.01 手动补挂）；per-frame 决策（chunk 级恒定残差证伪，偏离 #10）
- [x] TD3 残差（对齐论文式 14–21 的简化版；`policy/residual_td3.py` + `configs/residual_td3.yaml`）：
  - actor：MLP(1024-256)，输入（54-D 状态 + 13-D 笔位姿），zero-init tanh 头，输出有界 Δa；**最终只作用右手 20 指**（bound 0.02；raw 54-D 持平证伪，偏离 #11）
  - critic ×2（LayerNorm 抑 Q 高估）：(s, a_ref+Δa) → Q；per-frame 转移，γ=0.99/frame
  - 奖励：终局（成功 0 / 失败 −1）+ pen_z 每帧微 shaping（w=2×clip±0.005，偏离 #10），仿真自动标注
  - 安全：Δa bound 硬限幅（tanh×bound）+ env π/18 帧限幅；高原回滚（stall2 恢复 best_actor + lr 减半）
  - 探索：高斯噪声（σ=0.5×bound）+ 50% 纯贪心采集 + train_jitter 0.015 起点抖动
  - 每 episode 后 100 梯度步，buffer 200k，critic warm-up 500 转移后开 actor
- [x] 预算：实际 ~30k per-frame 转移（4 个 8.5 分钟前台段落，ckpt+buffer 断点续训）
- [x] 验收：任务分布 wl9 9/9=**100%**（基线 89%）✅ ≥90%；泛化 rand20 55% 持平基线（死区点归因，上限 ~75–80%）；success-vs-transitions 曲线 `verify_out/rl_m3_curve.png`（14 eval 点）

### M4（可选 Phase-B，按兴趣选做）

- [ ] 实现论文 Step 1 的 chunk-VAE codec（每手 20→9，0.43M 参数小模型，训练简单）→ latent residual RL 对比 raw residual（复现论文核心实验 Fig.9 latent vs raw）
- [ ] Sim-DAgger：脚本专家自动接管（失败检测触发 + 状态回滚 + 动作混合）→ 数据回流重训 ACT（对应论文 Step 3，用混合代替人工 2 s blend）
- [~] 第二个任务 Transfer between Hands（双手传递）——**脚本原型已跑通（2026-10-01，确定性口径）**
  - 新增 `scripts/proto_transfer.py` + `configs/transfer_marker.yaml`，串行四阶段 handoff/catch/release/hold；
  - 两个决定性设计：① 右手静止/撤离目标用 carry 笛卡尔链实际末点 `carry_q[-1]`（独立 IK 解 q_handoff 是不可达奇异构型，阶跃切入会令右臂在前馈下发散，见 PROGRESS 问题 27）；② 左手不从侧方横切（掌骨先撞落笔）而从笔自由端外**沿笔轴 +a 套入** pre_pinch 凹口、末段合拢 pinch3（问题 28）；
  - 已验证：reset(0,0) yaw0 seed1002 + 右手 plan_seed4（CLI 默认）→ hold 连续 16/10 帧，两次复现数值一致；
  - 遗留：跨 pick 姿态鲁棒——会合点随 pick 后笔轴 a 漂移，seed2 触发左肘奇异 / seed3 会合点到桌沿（问题 29/30）；下一步在 carry 前做笔轴朝向规范化，再录 transfer 示教并接 ACT/残差 RL 训练。
- [ ] 换 Dexora 的 DiT policy 或接入其 100K 仿真预训练数据

## 5. 建议目录结构

```
bimanual_dex_mvp/
├── assets/                  # Menagerie 拼装后的 MJCF 场景
│   └── bimanual_scene.xml
├── envs/
│   └── bimanual_dex_env.py  # Gymnasium 环境（观测/动作/奖励/成功判定）
├── teleop/
│   └── scripted_expert.py   # 三段式脚本示教
├── policy/
│   ├── act_lerobot/         # LeRobot 训练配置与 launch 脚本
│   └── residual_td3.py      # 残差 actor/critic + 训练循环
├── eval/
│   └── eval_policy.py       # 20 次试验成功率评测 + 曲线绘制
└── data/                    # LeRobot 格式数据集
```

## 6. 风险与降级方案

| 风险 | 应对 |
|---|---|
| Shadow Hand 54 维动作 + 接触丰富，ACT 收敛慢 | 先降难度：固定手型只学臂（14 维）打通流水线，再放开全 54 维 |
| 脚本示教覆盖不够 → SFT 基线过低（论文指出 <20% 时残差 RL 无效） | 加大脚本专家随机化（笔位姿/手型扰动）；确保 M2 基线 ≥40% 再进 M3 |
| 灵巧手接触仿真不稳（穿透/抖动） | 调 solver 迭代/接触参数；必要时换 LEAP Hand（Menagerie 有，16-DoF，更稳） |
| LeRobot ACT 对 54-D 高维动作的 chunk 配置不友好 | LeRobot 支持自定义 action shape；真不行就自写 ACT（代码量 ~300 行） |
| 纯稀疏终局奖励训不动 | 加高度/接触 shaping 作为辅助，最终报告里与论文设定区分 |

## 7. MVP 验收标准

1. MuJoCo 双臂灵巧场景可复现论文观测/动作接口（3 相机 + 54-D 状态 + 54-D 动作）；
2. ACT 在 Pick Up Marker 上从仿真示教收敛，闭环成功率 ≥40%；
3. 冻结 ACT + 残差 TD3 后成功率 ≥90%，并给出 success-vs-online-transitions 曲线（**M3 达成**：任务分布 wl9 100% vs 基线 89%，曲线 14 点）；
4. 全流程代码与数据格式（LeRobot v3）规范，可一键复跑。


## 9. 偏离论文 / 原始 Menagerie 设定清单（持续更新）

> 原则：仅在「位置控制机器人 + 最快打通 MVP」需要时偏离，并在此登记，M2/M3 报告里区分说明。

1. **Shadow 手 24 关节/20 驱动器**（J1/J2 腱耦合），论文手为 20-DoF；动作仍 54-D（M0 已记）。
2. **重力/科氏前馈**：对机器人全部 DOF 注入 `+qfrc_bias`（本 MuJoCo 3.5.0 符号约定，经 mj_inverse 验证），等效工业臂重力补偿（`gravity_comp=True`）。
3. **伺服加强**：腕 kp（10/8→60/40）+ kv、手指 kp（0.4–1.5→3–4）与 forcerange 提高（`scripts/gen_scene.py`），匹配「位置控制手」。
4. **关闭手部自碰撞**（`hand_selfcollide=False`）：手 geom 用 contype=2/conaffinity=1，手-手不碰、手-桌/笔/臂正常。原因：硬 geom 自碰撞封锁人类可实现的指尖对捏（实测独立手极限 1.08cm，场景内退化到 3.8cm）；近似真实软组织贴合。
5. **侧捏 frame_tilt=−86.9°**：正下俯捏时拇指指尖自然高于食指 3–7cm，绕手掌 x 轴侧旋使两侧等高会合；对应 pinch3 手型（THJ2 负值、食/中指深屈）。
6. **脚本专家工程化动作**：approach 从正上方 10cm 高位接近（低位接近会扫飞偏移笔）、descend −1.7cm、lift 放慢至 2.2s；close 阶段带接触反应式横向微调（3 帧迟滞、3mm/步、±12mm 限位，可选无接触扫描）。均为示教脚本工程手段，不改变任务/成功判定。
7. home 姿态 q2/q4 抬高 0.25rad 使手离桌（M0 ready 位姿的调整）。
8. **全链物理统一 mujoco 3.7.0**：录数/M2 评测/M3 RL 交互均在 lerobot-env（lerobot_mod 0.5.2 + mujoco 3.7.0）；专家标定与历史数字基于 3.5.0，3.7 下接触动力学漂移致好区重排，白名单在 3.7 下重测绘。
9. **M1 专家验收口径**：宽包络成功率 ≥80% 未达（3.5 下 narrow9@重试5 为 5/9，3.7 下 3/9）；改「好点白名单（3.7 实测 9 点）+ plan_seed≤5 重试 + 甩飞脏成功过滤」采集，策略泛化范围=白名单邻域（用户确认）。
10. **M3 RL 转移粒度改 per-frame + 奖励 shaping**：论文/原计划为 chunk 级转移 C=16；实测恒定 16 帧残差偏移对接近阶段毁灭性（噪声 0.03 rad 全灭），且 0/-1 稀疏终局奖励在 chunk 粒度信用分配过难，actor 两次死于 Q 高估漂移（0/9）。改为 per-frame 决策与转移（γ=0.99/frame 不变）、加 pen_z 每帧微 shaping（w=2×clip±0.005），终局奖励 成功 0/失败 -1 不变。
11. **M3 残差只作用右手 20 指**（`residual_dims: [34, 54]`，动作序 [左臂7/左手20/右臂7/右手20]）：原计划 raw 54-D 全维残差；实测 54-D 下 bound 受 Q 高估约束只能 0.005（≈1mm/帧），修正不动 cm 级手部对位误差，35k 转移与基线持平。改手部 20-D 后 bound 放宽至 0.02，wl9 100%。另加 `train_jitter: 0.015`（采集起点抖动，泛化探索用，rand20 收益已饱和仍 55%）。
12. **M4 Transfer 原型为确定性脚本、串行交接、固定成功口径**：当前 `proto_transfer.py` 是手工时序的脚本专家（非学习策略），用于先验证任务物理可行性与采集原型；采用**串行**交接（右手先送到位静止 → 左手再沿笔轴套入 → 右手释放）而非双臂协同动态交接（后者在 ±1cm 精度要求下暴露重力前馈极限环，已通过 pick 后臂阻尼 10 缓解）；成功口径限定 reset seed1002 + 右手 plan_seed4（plan_seed 2/3 因 pick 后笔轴朝向不同失败），跨姿态泛化待笔轴规范化后再谈，暂不作为成功率结论。任务目标与成功判定（左手独握、笔抬离桌面、稳定）不变。

## 8. 参考资料

- 论文：wuji6-DOF-1.pdf（式 1–21、Table 3/5/7 为超参主要来源）
- LeRobot: https://github.com/huggingface/lerobot
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- Dexora: https://github.com/dexoravla/Dexora （双臂灵巧 VLA，MuJoCo 仿真数据管线可参考）
- RoboPianist: https://github.com/google-research/robopianist （双 Shadow Hand MuJoCo 工程参考）
- ACT 官方: https://github.com/tonyzhaozh/act
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
