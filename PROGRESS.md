# 项目跟进记录 — 双臂灵巧 VLA 后训练 MVP（MuJoCo 复现）

> 主计划见 `PLAN.md`，需求与约束见 skill（`~/.agents/skills/bimanual-dex-mvp/SKILL.md`）。
> 规则：**每次工作都更新本文件**——进度打勾、新增问题与解决方案、追加更新日志。

## 一、当前状态总览

| 里程碑 | 内容 | 状态 | 备注 |
|---|---|---|---|
| M0 | 环境与双臂 MuJoCo 场景 | ✅ 完成（2026-09-28） | 7/7 验收通过；单步 24.5 ms（20 Hz 预算 50 ms） |
| M1 | 脚本示教 + LeRobot 数据集（100–200 条） | ✅ 完成（2026-09-29） | 160 条 LeRobot v3（train 130 / val 30）；白名单好点 9 点 + plan_seed 重试 + 脏成功过滤；宽包络 80% 未达（偏离已登记，用户确认） |
| M2 | ACT 模仿学习（基线 ≥40%） | ✅ 完成（2026-09-30） | 20k 步 loss 0.048；闭环 ensemble0.01：白名单 7/9=78%、rand20 12/20=60%，过 ≥40% 闸门 |
| M3 | 残差 TD3（目标 ≥90%） | ⬜ 未开始 | 冻结 ACT + 有界残差 |
| M4 | Phase-B 可选项（codec 对比 / Sim-DAgger / Transfer 任务） | ⬜ 未开始 | 按兴趣选做 |

**当前阶段**：M0 已完成。场景 54 执行器（左臂7+左手20+右臂7+右手20）、Gymnasium 环境 `BimanualDex-v0`、
3 路 224² 相机、20 Hz 控制（frame_skip=25，sim dt=0.002）、π/18 限幅、reset 笔随机 + 静置、
成功判定（手接触+整体抬离桌面 3 cm+连续 0.5 s 稳定）全部就绪。

**下次工作入口**：M3 — 冻结 ACT 20k checkpoint（推理必须开 temporal ensemble coeff=0.01），
残差 TD3（raw 54-D 关节空间、chunk 级转移 C=16、终局奖励），lerobot-env 物理，目标 ≥90%。
基线 rand20=60% 处论文「≥50% RL 稳定有效」区间。

### M0 验收记录

`MUJOCO_GL=egl python scripts/verify_env.py` → **7/7 PASS**：
gym 注册/空间、reset 合法随机化（10 次）、动作限幅实测 0.1745 rad、home 保持 5 s 笔不动、
双手闭合（手动 3.08 rad / 臂漂移 0.057 rad）、单步（3 相机渲染+物理）24.5 ms。
拼图：`verify_out/env_reset.png`、`env_hold.png`、`env_grasp.png`。

## 二、问题与解决方案记录

> 格式：问题描述 → 现象/根因 → 解决方案 → 状态。每次遇到新问题追加一条。

| # | 问题 | 根因/现象 | 解决方案 | 状态 |
|---|---|---|---|---|
| 1 | LeRobot 数据集 v3 格式与某些工具不兼容 | 部分工具只认 v2.1 | 统一用 v3，必要时写转换脚本 | 待验证 |
| 2 | 初始 home 关键帧严重穿透 | 双臂「平伸向中央」位姿下：右拇指穿桌 8.9 mm、双手指穿自身 link3/4（最深 25 mm）、双方法兰互穿 35.7 mm；位置保持 2 s 后最大漂移 0.33 rad | 对 14 个臂关节做 CEM（碰撞穿透软惩罚+手掌/指尖目标位姿），得无碰撞 ready 位姿（手掌 ±0.30 m、z≈0.69、指尖距桌 3 cm），写入 `gen_scene.py` 的 `ARM_HOME` | ✅ 解决，保持 2 s 无臂漂移 |
| 3 | Shadow Hand 自由度数与论文不一致 | Shadow 每只手 **24 关节/20 驱动器**（J1/J2 由固定腱耦合）；机器人本体 nq=62，加笔自由关节 nq=69/nv=68，动作仍是 54 | action 用 54 执行器；本体状态反馈用 `data.actuator_length`（臂=关节角，手=腱长 J1+J2），54-D 与动作量纲对齐；完整 62-D qpos/qvel 经 `env.robot_qpos/qvel` 暴露给专家/IK。**属偏离论文设定（论文手为 20-DoF），已标注** | ✅ 解决（已标注偏离） |
| 4 | 离屏渲染 136 ms/帧，3 路 402 ms/步 | 默认阴影贴图 4096 × 2 盏方向灯，阴影 pass 占 129 ms | 场景 `<visual><quality shadowsize="1024">`，保留阴影观感；降到 6.4 ms/帧，3 路+物理共 24.5 ms/步（<50 ms 实时预算） | ✅ 解决 |
| 5 | 腕相机画面被手腕壳体占满 | 相机挂在前臂 (0.01,0.045,0.06) 朝手指看，掌根/腕壳体正对镜头 | 改挂**手掌背部**（palm frame y=+0.055），俯视指尖方向并前倾 15°，FOV 75；ready 位可见桌面，左腕画面已能看到笔；接近位姿下的最终效果留 M1 用真实轨迹确认 | ✅ 解决（M1 复检） |
| 6 | mujoco 3.5 绑定 `geom.bodyid` 返回 0 维 ndarray | 放入 set/比较报 `unhashable type: ndarray` | 全部 `int(m.geom(gid).bodyid)` 转换 | ✅ 解决 |
| 7 | 零控时手指在重力下微张（最大 ~0.17 rad） | Shadow 默认位置增益很小（0.4–1.5） | ready 位指尖悬高 3 cm 不接触物体，暂不影响；若 M1 接触阶段抖动明显再加 gravcomp/调增益 | ⏳ 观察中 |

| 8 | 弱位置伺服在重力下静态下垂（臂 0.012、腕指 <0.002 量级） | MuJoCo 3.5.0 符号约定反直觉：经 `mj_inverse`(qacc=0) 权威验证，**保持静止所需力矩 = +qfrc_bias**（非常见的 -bias） | `BimanualDexConfig.gravity_comp=True`：`_step_rad` 每个 mj_step 前 `qfrc_applied[6:]=qfrc_bias[6:]`（跳过笔 6 个自由 DOF）；脚本专家 `_settle_pose` 静置循环同步加前馈（否则测出伪下垂污染标定） | ✅ 解决（偏离论文，已标注） |
| 9 | home 零位手指下垂触桌摩擦锁定 | 手离桌太近 | `scripts/gen_scene.py` ARM_HOME q2/q4 各 +0.25 抬高，重生成场景；home 静置 500 帧唯一接触为 pen-world | ✅ 解决 |
| 10 | 接触统计虚高（"24 帧双接触"实际为 0） | 遍历 contact 时把手指间自接触误归类为碰笔；且阶段归属必须在 `act()` 调用前取 ex.si/ex.k（act 内会推进） | 严格按 (geom1==pen & geom2==手指) 或反之判定；统计前先记录阶段 | ✅ 解决 |
| 11 | 静置下垂补偿越修越差（descend 跟踪差 17cm） | 重力补偿后下垂≈0，实测 site 位移主因是**手指被桌面顶住的平衡位移**，按下垂反向补偿等于把臂更用力压向桌面 | 废弃 site 下垂补偿，改为**握点不动点标定**：静置（含桌面）后读实际握点 g_world，site 指令平移 C−g_world 迭代；并对越修越差的轴（桌面硬约束）停止修正 | ✅ 解决 |
| 12 | 极限对捏构型（THJ2≈−0.56 反向）在场景中退化到 3.8cm 间隙 | Shadow 硬 geom 的**手内自碰撞**（thmiddle-ffdistal 等）阻止人类可实现的指尖贴合（独立手模型极限 1.08cm） | 新增 `hand_selfcollide=False`：手 geom 用 contype=2/conaffinity=1，手-手不碰、手-桌/笔/臂正常（模拟软组织贴合的常见简化，**偏离 Menagerie，已标注**） | ✅ 解决（已标注偏离） |
| 13 | 单手 overhead 对捏拇指尖总高于食指 3–7cm | Shadow 拇指基座高，正下俯捏两侧不等高 | `frame_tilt=-1.517rad(-86.9°)` 侧捏；随机搜索得到 pinch3（THJ2=−0.198 反向、食中指深屈 J3≈1.5、J0≈0.5/1.2），pre_pinch 同腕拇但食中张开；lift 放慢到 2.2s；approach vertical 改 0.1（低位接近会扫飞偏移笔） | ✅ 固定笔位 success=True（笔抬至 0.61–0.64） |
| 14 | 专家对笔位/朝向极敏感（随机 5%，缩小包络 39%） | 接触是刀尖式窗口：横向偏 1cm 即错过/扫飞；mode0 横握未标定；无/小指蜷起如"高跷"架高手掌 | 已加 close 阶段**接触反应式横向微调**（连续 3 帧仅单侧触笔才向该侧对中，双侧锁定，step 3mm/cap 12mm，可选无接触扫描）；包络/多手多模式鲁棒化仍在进行 | ⏳ 进行中 |
| 15 | 中心点成功无法复现（回退全部实验后仍失败） | 拇指实验引入 `pre_pinch` 后 descend 手型一直未回退；成功时代（中心 0.6309）descend 用 `pre`，用 pre_pinch 时跨笔标定整体劣化（descend 末朝向误差 21°、握持边缘化、hold 滑脱） | yaml descend `hand: pre_pinch → pre`，中心复测 0.647 SUC（187 帧） | ✅ 解决 |
| 16 | 网格结果对任何参数微调呈"混沌洗牌"（全网格重排、无法渐进改善） | `_solve_robust` 共享 `self._rng`，而 `_select_hand_mode` 按候选顺序（左→右）消耗随机数：改动候选集合（如禁左手）会改变右手标定的随机抽签 → 所有网格点结果重排。标定在接触丰富静置下是混沌平衡，微调只洗牌不跃升 | 停止无方向微调；建 `scripts/eval_expert_grid.py` 正式前台网格评测（`--tries` 换 plan_seed 重试取首次成功），用"确定性坏点/坏带"测绘代替盲调 | ✅ 认知固化 |
| 17 | 窄包络内存在确定性失败带，宽包络 80% 难达 | 部分点跨 5 个 plan_seed 同签名失败（lift/hold squirt 至 ~0.506 或 close 踢飞），重试无效；y>0 半区系统性失败（右臂跨越伸展几何偏置，mode0 拇指换边 0/9 严格劣于 mode1）；x=+0.025 整列坏带夹在两好列之间（显示 +0.03 为舍入） | 不硬冲宽包络：用户确认改走「好点白名单 + plan_seed 重试 + 脏成功过滤」采集（偏离已登记 PLAN §9）；3.7.0 物理下白名单重测绘 9 点（问题 18），已录满 160 条 | ✅ 解决（口径变更） |
| 18 | 录数环境 lerobot-env 物理为 mujoco 3.7.0，专家好区在 3.7 下漂移失效 | colfine 复验 2/4、narrow9 3/9：接触动力学随版本变化，混沌标定整体重排（RNG 顺序敏感同源）；录数/评测/RL 必须与训练数据同物理 | 全链统一 3.7.0（lerobot-env，用户确认）：3.7 下重测绘白名单 9 点（x∈{-0.025,0,0.05,0.075}，剔除 (0.025,0.025) 脏成功 max_z=0.973）；M2/M3 评测与 RL 交互均在 lerobot-env | ✅ 解决（偏离已登记） |
| 19 | LeRobot v3 建库 video feature names=None，训练 make_policy 报 `names[2] not subscriptable` | 该 mod 版 `dataset_to_policy_features` 要求 video names[2]=channels 以 HWC→CHW | 两库 meta/info.json 补 names=["height","width","channels"]（无需重录），录制脚本同步修复 | ✅ 解决 |
| 20 | 分段续训 CLI 试错：`--checkpoint_path` 不被接受、resume 报缺 config_path | draccus 仅暴露 `--resume` 与 `--config_path`；config_path 须指 `checkpoints/<step>/pretrained_model/train_config.json`（框架自动定位 last 与 training_state） | `policy/act_lerobot/train.sh` 自动 glob 最新数字步目录拼 config_path；离线须 HF_HUB_OFFLINE=1、root=数据集目录本身 | ✅ 解决 |
| 21 | 闭环推理报 uint8 溢出 / 维度 224≠3 | 训练循环手动做图像 uint8→float32/255（train.py:438），且 dataset 图像为 CHW（非 env 的 HWC）；preprocessor 只做归一化不含转换 | eval 喂入前 `permute(2,0,1).float()/255`，batch 维交给 AddBatch；post 返回裸 tensor（非 dict） | ✅ 解决 |
| 22 | ACT 20k 闭环仅 1/9，但首 chunk 与专家 L1=0.03 rad | 策略拟合准确（loss 0.048、量纲正确），失败来自 chunk 边界复合误差：n_action_steps=16 每 16 帧硬切换，接触任务放大 0.03–0.07 rad 偏差 | 开 **temporal ensemble coeff=0.01**（推理时参数，checkpoint 需手动补挂 ACTTemporalEnsembler）：白名单 1/9→**7/9=78%**、rand20 **12/20=60%** | ✅ 解决（M2 过闸） |

### 已识别的风险预案（开工前）

1. **ACT 基线过低（<40%）→ 残差 RL 无效**（论文明确指出 <20% 时 RL 无改善，50% 以上才稳定有效）
   → 预防：脚本专家加随机化（笔位姿/手型扰动）；M1 录数据前先在空场景验证专家成功率 ≥80%；M2 不达标时先降难度（固定手型只学 14 维臂）打通链路再放开。
2. **灵巧手接触仿真不稳（穿透/抖动）**
   → 预防：调 solver 迭代数与 contact 参数（solref/solimp/condim）；备选换 LEAP Hand（Menagerie 有模型，16-DoF 更稳）。
3. **LeRobot ACT 对 54 维高维动作 chunk 支持不直接**
   → 预案：LeRobot 支持自定义 action shape；实在不行自写 ACT（~300 行）。
4. **纯稀疏终局奖励训不动**
   → 预案：加高度/接触 shaping 加速收敛，报告中与论文设定明确区分（env 已内置 `reward_type="sparse"/"dense"` 开关）。
5. **双手/双臂自碰撞**
   → 预防：CEM 求无碰撞 ready 位姿（已落地）；reset 静置后做笔-手接触检查；M1 轨迹规划时保留安全间距。

## 三、更新日志

> 每次工作结束追加一条：日期 + 做了什么 + 结果 + 下一步。

- **2026-09-28（1）**：通读论文 `wuji6-DOF-1.pdf`（四步流水线：codec→SFT→DAgger→latent residual RL，54-DoF 双臂，5 个马克笔任务）；核实开源选型（LeRobot/ACT、MuJoCo Menagerie、Dexora、RoboPianist、MuJoCo Playground）；完成主计划 `PLAN.md` 与本跟进文档；建立需求 skill。→ 下一步：M0 开工，搭环境与场景。
- **2026-09-28（2）M0 完成**：
  1. `scripts/gen_scene.py` 拼装双 Panda + 双 Shadow Hand（54 执行器，顺序 [左臂7/左手20/右臂7/右手20]），含桌、笔（平放 cylinder）、3 相机、2 盏方向灯；
  2. 体检发现 home 位姿 4 类穿透，CEM 重求 ready 位姿；确认 nq=69/nv=68/nu=54，`actuator_length` 与 ctrl 同量纲；
  3. 腕相机改装手掌背部，shadowsize 4096→1024（渲染 136→6.4 ms/帧）；
  4. 新增 `envs/bimanual_dex_env.py`（Gymnasium `BimanualDex-v0`）：54-D 归一化动作（弧度限幅 π/18）、20 Hz（skip=25）、3×224² RGB + 54-D 状态 + 13-D 笔位姿、reset 随机/静置/合法性重试、成功=接触+抬离 3 cm+稳定 0.5 s、sparse/dense 奖励开关；
  5. `scripts/verify_env.py` 验收 7/7 PASS，env 单步 24.5 ms（EGL/RTX 4090）。
  → 下一步：M1，IK + 三段式脚本专家，固定笔位先打通「接近-抓-举」并验证成功判定能被触发。

- **2026-09-29 M1 抓取调试（1）**：
  1. 重力补偿：`BimanualDexConfig.gravity_comp`，`qfrc_applied[6:]=+qfrc_bias[6:]`（mj_inverse 验证符号），专家静置同步前馈；home q2/q4 +0.25 抬高离桌（重生成 `assets/bimanual_scene.xml`）；
  2. 握点标定：废弃 site 下垂补偿 → **握点不动点标定 + 桌面硬约束轴停修**（`_compensated_target`）；descend 按实际执行手型 pre_pinch 标定；
  3. 碰撞简化：新增 `hand_selfcollide` 配置（默认关，手 geom contype=2/conaffinity=1），解除 Shadow 自碰撞对极限对捏的封锁；
  4. 侧捏手型：随机搜索（静置→固定笔高→持握拉力测试）+ pipeline 联合搜索得到 `frame_tilt=-86.9°`、pinch3（THJ2 负值、食中深屈）、pre_pinch，lift 2.2s，approach vertical 0.1；
  5. **固定笔位 (0,0,yaw0) 正式打通**：`tune_expert.py success=True`，笔稳定抬至 z≈0.61–0.64，触发成功判定（181 帧）；
  6. 鲁棒性现状（右手/mode1 固定）：全随机 1/20；中包络（x[-0.05,0.10]/y±0.08/yaw±30°）14/36（39%，加迟滞反馈前后同分——失效多为"完全没对准/descend 扫飞"，不是单侧擦碰，反馈帮不上）；窄包络（x±3cm/y±4cm/yaw±15°）18/27（67%），失败点散布，接触窗口刀尖式。**专家 ≥80% 目标未达，列为 M1 收尾首要问题**；
  7. 关键坑新增：接触分类 bug、阶段归属需在 act() 前记录、actuator FFJ0 等为腱驱动（trntype=TENDON，trnid 是腱 id 不是关节）、批量脚本曾把 `m=env.model` 写成 `m,d=env.model` 浪费排查。
  → 下一步：把专家成功率做到 ≥80%（接触反馈/多手镜像/分模式手型），再录 100–200 条 LeRobot v3。

- **2026-09-29 M1 专家鲁棒性攻坚（2）**（指令：①descend 改接触引导 ②逐手标定侧捏参数，冲 80%）：
  1. **中心点回归根因**：descend 手型 pre_pinch 遗留未回退 → 改回 `pre` 修复（问题 15），中心 0.647 SUC；
  2. **RNG 顺序敏感统一解释**（问题 16）：全部"参数微调即全网格洗牌"现象归因于共享 `_rng` 被候选顺序消耗；teleop/scripted_expert.py 本会话所有新增（观测式笔跟踪 4 轮、双种子 approach、阶段偏置伺服、lift 清零等）证伪后已完整回退，文件等价会话开始状态；
  3. **证伪清单**（mid12 yaw0 前台，基线 3/12）：观测式笔跟踪 ×4 轮全 0/12（全阶段/平滑伺服/保剖面/lift 清零，track_pen 开关保留默认 false）；plan_restarts 6→12 得 1/12（坏左手静置分虚高误选）；close 2.5→3.5s 得 1/12（慢合拢踢飞更多）；grasp_modes [1]→[0,1] 自动选优得 2/12（settle 分数与抓取成败不相关）；approach vertical 0.15、react_step/cap 0.02 均净负回退；
  4. 新建 `scripts/eval_expert_grid.py` 正式前台网格评测器（narrow/mid12/mid36/good9/colfine 网格，--hand/--mode/--diag，**--tries 换 plan_seed 重试取首次成功**）；
  5. **结构测绘**（narrow9 yaw0@tries5 5/9=56%）：y>0 半区系统性失败（右臂跨越伸展几何偏置）；mode0 强制 0/9（右手 mode1 严格占优）；good9（好区加密）6/9=67%，**x≈+0.03 整列确定性坏带**（5 种子同签名，重试无效）夹在 x∈{0,+0.05} 两好列之间；
  6. **采集包络确定**（问题 17）：x∈{0,+0.05}×y∈[-0.05,0] yaw0 共 10 点 **10/10@tries≤5**（单次 7/10）——按此包络+失败重试只收成功条，可满足录数需求；宽包络 80% 需左手 IK 不稳定分支修复或抓取机制重设计，暂缓。
  → 下一步：g5 按包络录 100–200 条 LeRobot v3（8:2），失败重试只收成功条；PLAN.md 偏离清单补记"窄分布+重试采集"方案。
- **2026-09-29 M1 数据集录制（3）**（用户确认：按好区包络录数）：
  1. 发现 lerobot-env（lerobot_mod 0.5.2）物理为 mujoco 3.7.0，专家好区漂移（问题 18）→ 全链统一 3.7.0，白名单在 3.7 下重测绘（新增 expand12 探测网格），定稿 9 好点；
  2. 新增 `configs/record_pick_up_marker.yaml` + `scripts/record_dataset.py`（**lerobot_mod 续录必须 `LeRobotDataset.resume()`**，直接构造只读；白名单采样、plan_seed≤5 重试、max_z≤0.75 脏成功过滤、8:2 随机划分、断点续录）；
  3. **录满 160 条**：`data/bimanual_pen_train` 130 条/24852 帧、`bimanual_pen_val` 30 条/5765 帧（episode 长 183–200 帧，~11s/条含 SVT-AV1 编码），体检通过（笔位落点=白名单、y 静置漂移 -0.002、action 幅值 ≤3.4 rad）；
  4. M1 验收口径变更（用户确认）：宽包络 ≥80% → 好点白名单+重试采集（PLAN §9 偏离 8/9）。
  → 下一步：M2 — lerobot-env 下 ACT 训练（54-D chunk H=32），闭环评测同物理，目标 ≥40%。
- **2026-09-30 M2 ACT 模仿学习（4）**：
  1. 训练：lerobot-env（lerobot_mod 0.5.2），`policy/act_lerobot/train.sh`（draccus CLI，自动 resume）；ACT 52M 参数（ResNet18×3 ImageNet + Transformer 512 + VAE latent32），chunk H=32/C=16，bs8，lr 1e-5；20k 步（6.4 epoch，~32 分钟，12 step/s，显存 6.3GB），loss 12.2→0.261(5k)→0.098(10k)→0.062(15k)→0.048(20k)；
  2. 数据坑：video names 补全（问题 19）、resume config_path（问题 20）、闭环图像 CHW float/255 约定（问题 21）；
  3. **闭环评测** `scripts/eval_act.py`：同物理（mujoco 3.7.0）白名单 9 点固定 seed——无 ensemble 1/9，开 **temporal ensemble 0.01 后 7/9=78%**（同 seed 专家 7/9，持平；2 个失败点专家对 seed 也敏感）；rand20（白名单 ±1.5cm 抖动）**12/20=60%**；
  4. 首 chunk 诊断：ACT vs 专家动作 L1=0.032 rad（左臂/左手零误差），证拟合充分、误差在 chunk 边界复合（问题 22）；
  5. **M2 ≥40% 闸门达成**（60%），可进 M3 残差 TD3（论文：≥50% 基线 RL 稳定有效）。
  → 下一步：M3 — 冻结 ACT + ensemble，raw 54-D 残差 TD3，目标 ≥90%，画 success-vs-transitions 曲线。

## 四、关键数字速查（对齐论文）

- 动作空间：**执行器 54-D**（2×7 臂 + 2×20 手）；本体关节 **62-DoF**（Shadow 手 24 关节/手）；
  观测：3 相机 224² + 54-D actuator 状态 + 13-D 笔位姿（位置3+四元数4+速度6）
- 模型：nq=69（含笔自由关节 7），nv=68，nu=54，ncam=3；sim dt=0.002（500 Hz），控制 20 Hz，frame_skip=25
- chunk H=32，执行 C=25（论文）/ MVP 用 C=16 @20Hz；限幅 π/18 rad/帧（env 实测一致）
- 场景：桌 0.9×0.8 m（顶面 z=0.50），基座 ±0.62 m，ready 手掌 ±0.30 m/z≈0.69；笔 r=0.008、半长 0.075、0.015 kg
- RL：TD3，chunk 级转移，γ=0.99/frame，batch 128，buffer 200k，Polyak 5e-3，actor lr 1e-4 / critic 5e-5，终局奖励（成功 0 / 失败 −1），预算 3000–5000 转移
- DAgger（若做）：α_new=0.5，每轮 50 条，30k 步重训
- 性能：物理 5.9 ms / 3 相机渲染 18.6 ms / env 单步 24.5 ms（RTX 4090 EGL，224²）
