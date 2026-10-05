# 项目跟进记录 — 双臂灵巧 VLA 后训练 MVP（MuJoCo 复现）

> 主计划见 `PLAN.md`，需求与约束见 skill（`~/.agents/skills/bimanual-dex-mvp/SKILL.md`）。
> 规则：**每次工作都更新本文件**——进度打勾、新增问题与解决方案、追加更新日志。

## 一、当前状态总览

| 里程碑 | 内容 | 状态 | 备注 |
|---|---|---|---|
| M0 | 环境与双臂 MuJoCo 场景 | ✅ 完成（2026-09-28） | 7/7 验收通过；单步 24.5 ms（20 Hz 预算 50 ms） |
| M1 | 脚本示教 + LeRobot 数据集（100–200 条） | ✅ 完成（2026-09-29） | 160 条 LeRobot v3（train 130 / val 30）；白名单好点 9 点 + plan_seed 重试 + 脏成功过滤；宽包络 80% 未达（偏离已登记，用户确认） |
| M2 | ACT 模仿学习（基线 ≥40%） | ✅ 完成（2026-09-30） | 20k 步 loss 0.048；闭环 ensemble0.01：白名单 7/9=78%、rand20 12/20=60%，过 ≥40% 闸门 |
| M3 | 残差 TD3 | ✅ 完成（2026-09-30） | 手部 20-D 残差：任务分布 wl9 9/9=100%（基线 89%）达成 ≥90%；泛化 rand20 55% 持平基线（死区归因，上限 ~75–80%）；success-vs-transitions 曲线 14 点已产出 |
| M4 | Phase-B：Transfer between Hands 完整流水线 | 🟡 n10 完成（2026-10-05） | n8 ensemble 基线 58%（确定性）；n10 残差 TD3 整体 58%→65%（p=0.47）但**局部修复变体6（0/5→5/6，p=0.015）**；4 个恒败变体未解，90% 未达 |

**当前阶段**：M0–M3 已完成；M4 Transfer 任务 n1–n5 已收官脚本专家→扫描→录数→ACT 四段流水线。
场景 54 执行器（左臂7+左手20+右臂7+右手20）、Gymnasium 环境 `BimanualDex-v0`、
3 路 224² 相机、20 Hz 控制（frame_skip=25，sim dt=0.002）、π/18 限幅、reset 笔随机 + 静置、
成功判定（手接触+整体抬离桌面 3 cm+连续 0.5 s 稳定）全部就绪。

**下次工作入口（n11 决策点）**：手指残差只能修 1/5 个失败变体。候选：
① n11 残差空间扩到双臂关节（54-D，config 已备 configs/residual_td3_transfer_k3_arms.yaml），
   4 个恒败变体都败在 handoff 接近几何，臂残差是最便宜的对症实验；
② n9 扩示教+Sim-DAgger 提基线（工程量中）；③ codec latent 残差（论文 Fig.9，工程量最大）。

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
| 23 | chunk 级恒定残差毁灭接近阶段：Δa 恒定 16 帧，噪声 0.03 rad 即全灭；且 0/-1 稀疏奖励 chunk 粒度信用分配过难 | 改 per-frame 决策与转移（γ=0.99/frame），加 pen_z 每帧微 shaping（w=2×clip±0.005），终局奖励不变 | ✅ 已修（偏离 #10） |
| 24 | Q 高估漂移死亡螺旋（三波）：actor 追 critic 外推伪峰→0/9 | ①噪声 0.3→0.05+greedy 混合 ②bound 0.1→0.005 硬约束+actor lr 1e-5+L2 锚 0.5+LayerNorm critic ③高原回滚（stall2 恢复 best+lr 减半）；TD3+BC 归一在 Q≈0 域爆炸已证伪 | ✅ 已修（v6 配方） |
| 25 | 评测数值噪声：cudnn benchmark/TF32 致 ACT 推理微抖，翻转边界 seed，wl9 同 actor 两次测 78% vs 89%（±11–22%） | eval_act.py / residual_td3.py 均加确定性 flags（benchmark=False, deterministic=True, 禁 TF32）；基线重测：wl9 8/9=89%、rand20 11/20=55% | ✅ 已修 |
| 26 | raw 54-D 残差与基线持平，≥90% 未达 | bound 受 Q 高估约束只能 0.005（≈1mm/帧），修正力被 54 维稀释、修不动 cm 级手部对位误差（35k 转移 wl9 89%/rand20 60% ≈ 基线 89%/55%） | 残差只作用右手 20 指（residual_dims [34,54]，动作序 [左臂7/左手20/右臂7/右手20]），bound 放宽 0.02：win20 90–95%、wl9 两次 100%；终评 wl9 9/9=100% ✅、rand20 55% 持平（~4–5 失败点落已知死区 y>0 / y<−0.05 包络外 / x≈0.025 坏带，上限 ~75–80%，非 RL 失效）；train_jitter 0.015 续训无进一步收益（lr 已 4 次回滚降至 1.6e-6），收官 | ✅ 解决 |

| 27 | M4 交接 catch 段右臂剧烈发散（右腕 y 单帧瞬移 ~40 cm、笔被甩高 ~15 cm、穿模），与左臂是否运动无关 | 排查中先后证伪两条假设：重力前馈 `qfrc_bias[6:]` 的双臂 qvel 科氏耦合（放慢 catch 一倍无效）、欠阻尼慢不稳定（阻尼 10→30 反而更早炸）；**冻结左臂纯静止仍 h5 炸**才定位为**右手静止目标阶跃**——catch 把右手指令从 carry 链实际末点切到独立 IK 解 `q_handoff`，而 `handoff_along=-0.13` 下 q_handoff 是 pos 6.8cm/ori 26.5° 残差的不可达奇异构型，位置伺服强追不可达目标在重力前馈下发散 | catch/release 右手静止与撤离基准统一改用 carry 笛卡尔链实际末点 `carry_q[-1]`（不再用独立解 q_handoff），消除 handoff→catch 目标跳变；右腕随即锁死、笔长时间稳定 | ✅ 解决 |
| 28 | 左手从侧方(-y)水平横切承接，grasp_site 距笔还有 10 cm 时左手小指掌骨（lh_lfmetacarpal）先撞落笔；open 直伸手型指尖还低于 site，会从笔下方铲入挑高 | `lh_grasp_site` 挂在拇指基座（thbase，手掌桡侧），手整体沿 -y 平移时掌尺侧/掌骨先于 pinch 指尖到达笔杆；open 手型四指伸直、指尖几何远低于 pinch3 标定位 | 改**沿笔轴套入**：预抓位=承接位沿 -a 退 `axis_approach=0.13`m（笔自由端之外，朝向直接用最终抓握 R_l），catch 保持朝向沿 +a 直线平移，笔杆滑进 pre_pinch 张开的拇指-食/中指凹口，套入后 40%（`catch_close_start=0.6`）才合拢 pinch3；撤退/套入全程用 pre_pinch 而非 open | ✅ 解决（plan_seed4 成功） |
| 29 | 交接对 pick 后笔轴朝向 a 高度敏感：plan_seed 2/3/4 仅 4 成功（1/3） | handoff 会合点 `pen_pos = pick_pen + a·handoff_along` 完全由 pick 后笔轴 a 决定：seed2 的 a 使会合点落身体正中 x≈-0.01，触发左肘伸直奇异（左承接朝向 IK 残差 67.6°，套入仅小指节刮笔、抓空）；seed3 会合点偏到 y+0.24/z0.52 桌沿，carry 阶段即掉笔；seed4 a=[0.90,-0.22,-0.37] 会合点在左臂舒适侧 x=-0.105 才成立 | 当前以**确定性协议** reset seed1002 + 右手 plan_seed4 作为 M4 原型成功口径（两次复现数值完全一致）；跨姿态鲁棒需在 carry 前加"在右手中把笔旋转规范化到标准轴朝向"步骤（利用腕自由度），列为下一阶段 | ⏳ 待做（已限定成功口径） |
| 30 | 左臂在身体中央会合点 IK 解出肘 `l_joint4≈-0.079`（距 range 上限 -0.07 仅 0.6°，近伸直奇异、无刚度），右手同位置肘=-1.19 弯而稳 | 7-DoF 逆解零空间偏向该构型；且 `ArmIK.solve` 的 pos_tol=5mm/ori_tol=5° 提前退出会让零空间屈肘项一次都不执行 | 加 `tuck_solve()`：置 `pos_tol/ori_tol=-1` 强制跑满 500 迭代 + 零空间屈肘重解，肘从 -0.079 拉到 -0.67；保留于 plan_left_goal（最终由沿轴套入+选 seed4 共同规避中央奇异） | ✅ 缓解（根因仍需笔轴规范化） |

| 31 | 笔轴规范化（右手中腕旋转）对弱握/近垂直轴失效 | pinch3 位置伺服夹持靠摩擦传扭：大步旋转（30°/2s）打滑甩飞（seed3 45.8°→97° 笔落桌），小步慢速（10°/3s）对近垂直轴由重力 pivot 主导、笔在夹持中持续下滑 pivot 失控（seed2 65.8°→36.7° 后落桌）；seed3 甚至在 settle 阶段笔已滑到桌上 | `normalize_enable:false` 默认关闭（保留闭环限步实现）：承接凹口机制本身容忍实测轴偏（认知见问题 34），规范化收益为负 | ✅ 认知固化（功能保留默认关） |
| 32 | 近垂直笔轴左手承接朝向 IK 残差 63.7°（完全抓空） | `_grasp_frame` 在 a≈±UP 时 `y_col=cross(-UP,a)` 退化，且 R0 各列非正交（z 列固定 -UP） | `grasp_frame_safe`：|a·UP|<0.7 与原式逐元素一致（保 seed4 口径），否则用水平参考系构造正交基；p0 左承接残差 63.7°→2.4° | ✅ 解决 |
| 33 | handoff 后在线重规划引入两个回归 | ①`plan_left_goal` finally 硬编码 `set_collision_mask(None)`，把动作段 cross 掩码清掉 → 左手接近时撞右臂 geom 把笔打落（catch 首 10 帧即落桌）；②右手静止目标用实测 `r_now` 会卸掉"伺服仍向规划链末收敛"的残余夹持压力，笔 10 帧内滑落 | ①plan_left_goal 改为保存/恢复调用前掩码模式；②右手静止/撤离基准保持规划链末 `carry_q[-1]`，实测值仅作 IK 种子 | ✅ 解决 |
| 34 | 按 carry 后实测笔轴规划左手朝向反而必败（37.7° 不可达） | 承接成功机制实证：左手按**标称浅轴**形成承接凹口，笔从右手滑落入凹口自动对正（catch 末 lh 轴垂距 2.5–3.3cm 松接触）；carry 后实测轴常偏陡 30°+，2-DoF 腕在该位置对陡轴朝向不可达，且绕轴扭转搜索 12 候选全部 ~37.7°（误差与扭转无关） | 在线重规划 = **实测位置 + 标称轴朝向**（位置消 carry 落点漂移走廊错位，朝向保浅轴可达凹口）；seed4 回归 16/10 ✅ | ✅ 认知固化 |

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

- **2026-09-30 M3 残差 TD3（6）**：
  1. 新建 `policy/residual_td3.py` + `configs/residual_td3.yaml`：冻结 ACT 20k（ensemble 0.01）+ raw 54-D 残差 TD3（actor 1024-256 zero-init tanh、双 critic LayerNorm、buffer 200k、分段续训、best_actor 快照、高原回滚）；
  2. 三轮配方证伪（问题 23/24）：chunk 级恒定残差全灭 → per-frame；噪声/bound 过大两波死亡螺旋 → bound 0.005+lr 1e-5+L2+rollback（v6）后行为健康（win20 65%）；
  3. 35k per-frame 转移 / 16k 更新：eval 峰 9/9=100%（后证实为测量噪声），**确定性复测：TD3 wl9 8/9=89%、rand20 12/20=60% vs ACT 基线 8/9=89%、11/20=55%**——持平略好，≥90% 未达；
  4. 根因判断：bound 0.005（≈1mm）太小修正不动 cm 级抓取失误；下一步选项：手部 20-D 残差 / bound 0.01+lr 1e-6 / 转 M4 latent。
  → 下一步：等用户决策方向后继续。

- **2026-09-30 代码入库（5）**：`bimanual_dex_mvp/` 推送到 GitHub [ABC12345anouys/PostTraindex](https://github.com/ABC12345anouys/PostTraindex)（HTTPS + token，SSH 不通）；M0→M2 全部工作按时间顺序整理为 50 个语义 commit 推送（1 个 Initial + 50 个新增：gitignore/专家配置调优、网格评测器 + 25 份评测 JSON、6 张诊断 montage、录数脚本三阶段、录数 yaml 白名单定稿、train.sh 两阶段、eval_act.py 三阶段、ACT 三份评测结果、PLAN/PROGRESS/README 入库）；大文件（checkpoints 2.4G、data 205M）按 .gitignore 排除；同步建 5 个 issue：#1 y>0 半区失败、#2 x=+0.025 确定性坏带、#3 mujoco 3.7 物理漂移、#4 temporal ensemble 需手动补挂、#5 M3 残差 TD3 Roadmap。

- **2026-09-30 M3 手部 20-D 残差收官（7）**：
  1. 配方定型（问题 26）：残差只作用右手 20 指（`residual_dims: [34, 54]`），bound 0.02，其余沿用 v6（per-frame 转移 + pen_z shaping + L2 锚 0.5 + LayerNorm critic + 高原回滚）；win20 90–95%、wl9 两次 100%（ep20 / ep100）；
  2. **终评（确定性协议，best_actor 快照）**：wl9 9/9=**100%**（基线 8/9=89%）✅ 达成 ≥90% 目标；rand20 11/20=55%（基线 11/20=55%）泛化持平——失败 ~4–5 点落已知死区（y>0 专家死区 / y<−0.05 包络外 / x≈0.025 坏带）， rand20 上限 ~75–80%；
  3. `train_jitter: 0.015` 起点抖动续训一段 rand20 仍 55%（lr 经 4 次回滚降至 1.6e-6），收益递减，决定收官；全程 ~30k per-frame 转移、4 个 8.5 分钟前台段落；
  4. 产物：`verify_out/rl_m3_final.json`、`rl_m3_curve.png/.csv`（14 个 eval 点）、`checkpoints/rl_m3/best_actor.pt`；
  5. 登记 PLAN 偏离 #11（手部 20-D 残差 + train_jitter）。
  → M3 验收通过：任务分布 100% ≥90% ✅ + success-vs-transitions 曲线 ✅。下一步：M4（可选 Phase-B）待用户指令。

- **2026-10-01 M4 Transfer between Hands 脚本原型跑通**：
  1. 新增 `scripts/proto_transfer.py`（~560 行）+ `configs/transfer_marker.yaml`（base 复用 pick_up_marker.yaml），复用 env `step_rad`、`ArmIK`、`ScriptedExpert`、contactfilter；串行四阶段（均 20Hz）：**handoff**（右手 min-jerk carry 到会合点并静止持笔，左手先撤到笔自由端外的沿轴预抓位）→ **catch**（左手保持最终抓握朝向沿笔轴 +a 直线套入，pre_pinch→末段 pinch3 合拢，双手共持）→ **release**（右手 open 沿笔轴撤 0.10m）→ **hold**（左手 pinch3 独握，连续 10 帧判成功）；
  2. **关键根因一（问题 27）**：catch 段右臂目标从 carry 实际末点跳到独立 IK 解 `q_handoff`（6.8cm/26.5° 不可达奇异构型），位置伺服在前馈下发散（右腕瞬移 40cm、笔甩高 15cm）；通过"放慢 catch/冻结左臂/加大阻尼"三个对照实验排除 qvel 耦合与欠阻尼后定位，改右手静止/撤离基准为 `carry_q[-1]` 即稳；
  3. **关键根因二（问题 28）**：左手侧方 -y 横切时掌尺侧掌骨先撞落笔、open 指尖下铲；改沿笔轴套入（`axis_approach=0.13`、朝向锁定 R_l、pre_pinch 凹口吞入笔杆、末 40% 合拢）；
  4. 沿用前序"pick 后 `apply_arm_damping(10)`"抑制重力前馈极限环；新增 `tuck_solve()` 零空间屈肘缓解左臂中央会合点肘伸直奇异（问题 30）；
  5. **结果（确定性协议 reset seed1002 + 右手 plan_seed4，CLI 已设为默认）**：两次复现数值一致——pen_z handoff 0.751（右手独握）→ catch 0.674（左手接管、右手松开）→ release 0.678 → hold 0.677，**hold 连续 16/10 帧，TRANSFER success=True**；plan_seed 2/3 因 pick 后笔轴朝向不同而失败（问题 29，当前口径限定 seed4）；
  6. 产物：`verify_out/transfer_proto/{carry,gather,handoff,catch,release,hold}.png + result.json`；已精简逐帧诊断输出；
  7. 本轮 M4 代码（proto_transfer.py、transfer_marker.yaml、PROGRESS/PLAN）**尚未 git commit**，等用户明确指示再提交推送。
  → M4 Transfer 脚本可行性已验证（确定性）。下一步待指令：笔轴规范化做跨姿态鲁棒 → 录 transfer 示教/训练；或其它 Phase-B；或先提交代码。

## 四、关键数字速查（对齐论文）

- 动作空间：**执行器 54-D**（2×7 臂 + 2×20 手）；本体关节 **62-DoF**（Shadow 手 24 关节/手）；
  观测：3 相机 224² + 54-D actuator 状态 + 13-D 笔位姿（位置3+四元数4+速度6）
- 模型：nq=69（含笔自由关节 7），nv=68，nu=54，ncam=3；sim dt=0.002（500 Hz），控制 20 Hz，frame_skip=25
- chunk H=32，执行 C=25（论文）/ MVP 用 C=16 @20Hz；限幅 π/18 rad/帧（env 实测一致）
- 场景：桌 0.9×0.8 m（顶面 z=0.50），基座 ±0.62 m，ready 手掌 ±0.30 m/z≈0.69；笔 r=0.008、半长 0.075、0.015 kg
- RL（M3 定稿）：TD3，**per-frame 转移**（偏离 #10），手部 20-D 残差 bound 0.02（偏离 #11），γ=0.99/frame，batch 128，buffer 200k，Polyak 5e-3，actor/critic lr 5e-5（回滚降至 1.6e-6），终局奖励（0/−1）+ pen_z shaping（w=2，clip ±0.005），实际 ~30k 转移
- DAgger（若做）：α_new=0.5，每轮 50 条，30k 步重训
- 性能：物理 5.9 ms / 3 相机渲染 18.6 ms / env 单步 24.5 ms（RTX 4090 EGL，224²）


## M4 Phase-B 续：抓持态扰动与物理版本陷阱（2026-10-01）

### 关键纪律（教训）
- **proto_transfer / 一切 transfer 实验必须用 lerobot-env python（mujoco 3.7.0）+ MUJOCO_GL=egl**。
  系统 python3 是 mujoco 3.5.0，物理不同：3.5.0 下确定性成功口径（(0,0) seed1002 plan4）直接失败，
  曾据此误判"±2mm 失败=位姿效应"与"扰动零容差"，在 3.7.0 下重测才得到正确结论。
  sweep_transfer.py 内 PY 常量是正确解释器；手动跑务必 `/home/lifd/anaconda3/envs/lerobot-env/bin/python`。

### 问题 35：抓持态 teleport 扰动 → carry 途中笔 pivot 变陡
- 做法：settle 后直接改 pen qpos（xy±1-2mm、轴±1-2°）+ mj_forward + 15 帧再静置。
  静置后抓持存活（pen_z 0.62，轴偏 2-4°），但 carry 末实测轴转到近垂直
  （如 [0.866,-0.376,-0.329]→[0.426,-0.07,-0.902]，60°+），左手标称凹口无解。
- 根因：pinch3 对 8mm 半径笔是边际握持，扰动改变接触构型→旋转刚度降→carry 加速段笔 pivot。
  与问题 31（normalize 甩飞）同物理：弱握 + 腕/臂运动 = 笔在手中转。
- 结论：teleport 扰动本身可行（静置存活），**瓶颈转移到 carry 保向**。

### 问题 36：track_l 每帧实测轴重建凹口（track_orient=meas）→ 零扰动也失败
- 凹口必须是静止吸引子，笔才能"滑落入凹口自动对正"（问题 34 机制）；
  逐帧追实测轴 = 凹口跟笔一起晃，承接失效。已回退为 nominal 默认，meas 模式弃用。

### 问题 37：±2mm 桌面抖动 × plan_seed4 → pick 本身失败（3.7.0 实证）
- plan_seed4 是 (0,0) 点特化的：kle1/kle3 pick 失败（笔平躺），kle0/kle2 抓住但轴陡/滑桌。
- 录数多样性不能靠"固定位姿扰动+固定 plan_seed"；须走 plan_seed 重试池（M1 tries=5 范式）
  或修复 carry 保向后用抓持态扰动。

### 当前成功池（3.7.0，全部 (0,0) yaw0 seed1002 plan4）
- 默认参数 / left_goal_dz=0.05 / catch_close_start=0.5 三组合各 16/10。
- j13 类（轴偏 7°）推进到 ca20——参数池组合可能补齐，待扫。

### 下一步（n3→n4 决策点）
1. 温和 carry 旋钮（carry_arch=0 / handoff_seconds=3.5）能否止住扰动后 pivot → 若能，
   录数 = 标称 pick + teleport 扰动 + 温和 carry（高产出、分布可控）。
2. 若不够，做 carry 保向闭环（右手每帧按实测轴小步反旋，deadband+限幅反馈，非 normalize 开环大步）。
3. 并行兜底：桌面点抖动 × plan_seed 重试 × 浅轴过滤（|Δaxis|<10°）× 参数池，低产出但物理自然。


### 问题 38：握力加固（pinch3 沿合拢方向外推 15%/30%）→ 反效果
- 过压本身改变接触构型，ctrl 也 pivot（settle 轴都被顶偏）；grip_tighten  knob 保留默认 0，方向放弃。

### 问题 39：手腕刚性 nudge（整手带笔微旋/微移）→ 笔不随手走
- 假设：形闭包内笔-手相对不动，nudge 只改世界系位姿。实证错误：3 个 nudge seed
  （±2.7°/±2mm 各异）笔全部滑到同一次级平衡轴（≈ nominal+7.6°），carry 中再次重力
  pivot 到近垂直（[0.487,-0.153,-0.86]），catch 撞飞（pen_z 1.124）。
- 结论：该抓持有 ~7° 内部空隙，任何人工扰动都落到同一不稳定平衡。**人工扰动路线
  （teleport/nudge/tighten/温和 carry/track_orient=meas）全部证伪**。

### n3 最终结论与录数路线（决策）
- 唯一物理诚实多样性源 = **自然 pick 变体**：桌面点 × plan_seed 池 → 浅轴过滤 →
  transfer 全程。自然形成的抓持接触构型 carry 稳定（j13 轴偏 7° 推进到 ca20 实证）；
  且 replan 已按 settle 实测轴规划凹口（小轴偏可达），配参数池（默认/dz0.05/cs0.5）
  有望补齐 j13 类。
- 扫描器设计：单进程循环（point × plan_seed），pick+settle 探针 ~40s/个，浅轴
  （|Δaxis|<10°，pen_z>0.6）才跑全程；handoff 末轴漂移 >20° 提前 kill。
- 目标：40-60 条互异成功 episode（pose × 参数组合；同组合重复=逐比特相同，无效）。

## M4 n4：自然 pick 变体扫描 → 浅轴池 → 录数（执行中）

### 问题 40：扫描器 pick 前清零 dof_damping → 锚点口径崩坏（假阴性整批）
- 扫描器为"保 M1 pick 接触姿态"在 pick 前 set_damping(0)，结果 (0,0) ps=4 settle 轴从
  a_star 掉到 60.8°/pen_low_z 0.497。根因：XML 默认 joint damping=1，proto_transfer
  pick 段一直跑在默认阻尼上；"清零"才是改物理。
- 修复：env 创建时存 damp0，每组合 pick 前恢复默认、pick 后才开 arm_damping=10
  （与 proto 逐帧一致）。修复后锚点轴偏 0.0° 逐比特复现。
- 教训（与问题 32 同族）：**任何初始化侧改动必须先过锚点回归**
  （(0,0) seed1002 plan4 → settle 轴偏≈0°）再批量跑。

### 扫描结果（25 点环 × ps0-9，250 组合，3.7.0）
- pick 成功 132/250；浅轴候选（|Δaxis|<10° 且 pen_low_z>0.6）12 个，ps∈{0,2,4,9}，
  覆盖中心到 ±12mm 环 10 个点。产物：verify_out/scan_pick/scan_poses.json + scan_pick.png。

### 池 round1（12 候选 × def/dz05/cs05）→ 仅 5/36 成功
- 成功：锚点 (0,0)ps4 ×3 + (-4,+7mm)ps2 ×2（def/cs05）。
- 关键发现（推翻"settle 浅轴即充分"）：**成功条 carry 漂移也 ~33°**
  （settle [0.903,-0.224,-0.366]→handoff 末 [0.532,-0.488,-0.692]，两个成功候选落到
  同一稳定平衡轴）；失败候选中不乏 settle 轴偏 1.6° 者。
  成败判别 = 漂移是否落到同一平衡轴 + 承接凹口毫米级几何（问题 34 机制的包络）。
- 失败分类（31 条）：carry 途中掉笔(ha)3、catch 撞飞/滑落(ca/re)14、hold 不达标 14；
  其中 28 条 carry 漂移>20° 但这不是判据（成功条 33°），只是相关现象。
- 对策（round2 并行中）：参数池扩 9 组（±dz03/06、cs04/07、along±0.01）+
  plan_seed 10-19 扩展扫描补候选。

### 池 round2/3 与录数完成（n4 收官）
- 参数池扩 9 组（def/dz03/05/06/cs04/05/07/am01/ap01）后：round2 13/108（ps0-9 候选），
  round3（ps10-19 新候选 12 个）28/108——**ps19/ps11 抓握族 transfer 亲和性显著更高**。
- 合并成功池 **41 互异组合**：8 桌面点 × ps{2,4,11,19} × 9 参数（verify_out/transfer_success_pool.json）。
- 录数（record_transfer.py，STEP_HOOK 单点注入 run_phases；episode=reset+pick 不录，
  settle 后起录 handoff/catch/release/hold，只收 hold 成功条）：**41/41 一次全过**
  （池验证口径逐比特复现，train=33/val=8，每条 156 帧）。
  产物：data/bimanual_transfer_{train,val} + data/transfer_dataset_qc.png。
- 踩坑：LeRobotDataset.resume 需 HF_HUB_OFFLINE=1（否则连 hub）；val 零条空壳 meta 不全
  （缺 tasks.parquet），resume 前须检测重建（已固化在 make_datasets）。

## M4 n5：Transfer ACT 闭环 + 多模态崩塌对照实验（2026-10-04）

### 问题 41：eval/RL 环境自身成功判定提前截断 episode → 假 0%
- 现象：旧口径 eval 普遍 ~63 帧 term=True 截断，迟到的左手交接根本没机会发生，旧 0/9 结果全部含此坑。
- 根因：env 成功条件=任一手 grasped&lifted&stable 连续 hold_frames=10 帧；右手停滞持笔（handoff 等待段）
  单独满足该条件，episode 被当成成功提前结束。
- 修复：`scripts/eval_act_transfer.py` 与 `policy/residual_td3_transfer.py` 的 BimanualDexConfig 显式
  `hold_frames=10**9` 禁用该截断；掉笔 fall_z=0.35 / xy_limit=0.55 / max_steps=500 超时仍然生效。
- 口径核验：prefix pick 194 + settle 25 = 219 帧，策略段 200 帧（max_steps=500 即 25s）无截断。修复后评测统一 tag 后缀 `_v3`。

### 两臂数据集（控制变量：同 12 个 (point, plan_seed) 评测变体）
- 多模态 46 条：12 变体 × 平均 4 种参数风格（def/dz05/cs05/…），train37/val9；
  同一初始 obs 最多 8 种不同动作标签，ACT 推理 latent=0 输出模态均值——多模态崩塌假设来源。
- 单模态 12 条：每 (point,ps) 变体只保留 def 一种风格（`verify_out/transfer_pool_unimodal.json`），
  录数 12/12 全过、各 156 帧 hold=16，train9/val3。

### 对照结果（各 40k 步、同评测 12 变体、_v3 口径；曲线 verify_out/act_m4{,_uni}_curves.png）

| ckpt | 多模态 46 条 | 单模态 12 条 |
|---|---|---|
| 5k | 0/12 = 0% | — |
| 10k | 2/12 = 17% | **5/12 = 42%** |
| 20k | 0/12 = 0% | 0/12 = 0% |
| 30k | 1/12 = 8% | 0/12 = 0% |
| 40k | **5/12 = 42%**（loss 0.055，55 epoch） | 0/12 = 0%（loss 0.054，171 epoch） |

### 问题 42：「多模态崩塌」假设证伪——真问题是小数据过拟合 + 长程策略脆弱
- 假设：同一变体多种风格动作标签导致 ACT 学模态均值 → 0%；预期单模态显著且稳定更好。
- 结果：单模态仅 10k 早期达 42%，继续训练反崩到 0%；多模态晚期（40k）达到同样的 42% 且仍在上升通道。
  两者 loss 同为 0.054/0.055，闭环天差地别——**单模态 12 条 / 171 epoch 是典型小数据闭环过拟合**，
  多风格数据反而起到正则作用。多模态崩塌不成立。
- 非单调性（两臂 20k 同为 0%）：12 变体小样本评测 × 长程 chunk 策略，少量轨迹分叉即造成 0↔5/12 跳变，
  后续以「整条曲线 + 多 ckpt」而非单点成功率判读。
- 失败形态（40k 多模态）：成功 5、carry/catch 途中掉笔 6、悬空未交接 1；
  40k 单模态：掉笔 9 + 桌面拖行 3（pen_z_min~0.50 拖满 200 帧），无 near-miss。
- 决策：n6 冻结基线选多模态 40k（5/12=42%，过 M2 同款 ≥40% 闸门；论文经验 <20% 时残差 RL 无效，42% 可用）。

### 本轮工程产物
- 新增：`scripts/eval_sweep_m4.py`（逐 ckpt 评测 + 成功率/loss 双面板曲线，参数化支持两臂）、
  `policy/residual_td3_transfer.py` + `configs/residual_td3_transfer.yaml`（A_DIM=40 双切片，冒烟通过）、
  `configs/record_transfer_uni.yaml` + `policy/act_lerobot/train_transfer_uni.sh`。
- 修改：`scripts/eval_act_transfer.py`（hold_frames 修复）、`scripts/scan_pick_poses.py`（round5 补扫）。
- 数据/图：data/bimanual_transfer{,_uni}_{train,val}（不入库）、verify_out/act_m4{,_uni}_curves.png、
  verify_out/eval_transfer_act_m4{,_uni}_*_v3/（results.json + success_rate.png）、
  verify_out/transfer_pool_unimodal.json。

## M4 n6：raw 残差 TD3 50k —— 无显著增益的负结果（2026-10-05）

### 实验设置（与 M3 同构）
- 冻结多模态 ACT 40k（040000，ensemble 0.01）+ 40-D 双手指残差（[7,27]+[34,54]，bound 0.02 rad），
  TD3 超参全部对齐 M3（expl0.5/greedy0.5、100 grad/episode、warmup 500）。
- **50019 transitions / 496 episodes / 49600 updates / 49 个 pool 评测点**，16 次 best-actor 回滚、
  actor_lr 衰减到地板 1e-7。产物 checkpoints/rl_m4/（不入库）、verify_out/rl_m4_{curve,final,eval_noise,eval_stats}.*。

### 问题 43：单轮 12 变体评测是高噪声指标——训练曲线的 67% 峰值是噪声不是学习
- 训练曲线在 25%–67% 间无趋势振荡（前 10 点均值 48.3%，后 10 点 45.8%）；训练 rollout 成功率
  43%→51%（含 50% 探索噪声条），贪心评测不支持提升结论。
- 噪声校准（每策略 3 轮 ×12 变体 = 36 trial，verify_out/rl_m4_eval_noise.json）：

| 策略 | 单轮极差 | 36 trial 合并 |
|---|---|---|
| 纯 ACT（残差=0，RL harness 口径） | 42%–50% | 17/36 = 47% |
| best actor（训中峰值存档） | 50%–58% | 19/36 = **53%** |
| 末段 actor（50k） | 25%–58% | 16/36 = 44% |

  best vs 基线双侧 Fisher p=**0.81**，差异不显著；同一 actor 单轮可在 25%↔58% 间跳动
  （ACT 推理逐轮采样 style latent，长程 chunk 放大分岔）。

### 结论与归因
- **n6 判决：raw 关节空间残差 TD3 在 transfer 42% 基线上、50k 预算内无可信增益**，未达 90% 目标。
  best actor 名义 +6pp 在噪声范围内；这是本项目第一个 RL 负结果。
- 与 M3（89%→100%）对比的解释：M3 是近天花板任务的小幅修补；transfer 基线 42% 正落在论文
  「<20% RL 无效、≥50% 才稳定」的灰区。失败形态是毫米级沿轴接笔几何，40-D 手指 0.02 rad 残差
  更可能不足以表达修正（M3 中手指残差只需微调抓握）。
- 方法论收获（已固化进绘图口径 scripts/plot_rl_m4.py）：**12 变体单轮评测不能用于策略比较**，
  今后 RL 判据 = 多轮复评合并 + Fisher 显著性 + 滑动均值，曲线单峰不作数。
- 另修复一个流程细节：脚本到 50k 的自动终评用末段 actor（42%），best actor 终评须显式
  `--finalize`（曾被 done 标志短路）；两种终评均留档（rl_m4_final.json / rl_m4_final_current_actor.json）。

## M4 n7 启动：评测噪声根因定位 → ensemble/扩数据路线（2026-10-05）

### 问题 44：闭环评测随机性的真正来源 = EGL 渲染 1px 边缘噪声 × 接触混沌（非 ACT latent）
- 排查链（同变体 (-12,0) ps4 双 rollout 逐帧比对）：
  ① 推断 ACT 推理采 latent → 查 lerobot ACT 源码，`use_vae and training` 才走 VAE 分支，
     推理 latent 恒为 0；同 obs 连推 5 次动作逐比特一致（maxdiff=0）。
  ② MuJoCo reset(seed=1002)、expert plan_seed 全部确定性；双 rollout settle 末 qpos/qvel 逐比特一致。
  ③ 逐帧比对定位到 frame1：qpos/qvel/state/object 全同，**right_wrist 相机图像出现 1/255 像素差**；
     frame2 扩散到 head/left_wrist（28–71），qvel 差 4.8e-2→0.99 rad/s，episode 成败翻转。
  ④ 同状态连渲两次偶发 0、偶发 1px 差 → EGL GPU 边缘着色竞争（共享 GPU 环境），与策略无关。
- 含义：任何单轮 12 变体成功率都带 ±2~4/12 噪声；n6 的 67% 峰值与 42% 谷值同因。
- 对策（PLAN n8–n10）：多 ckpt ensemble 平均掉像素扰动的动作响应 + 多轮复评口径；
  扩示教提基线到 ≥55% 再 RL。

## M4 n8：多 ckpt ensemble 一次过闸门——58% 且评测确定性（2026-10-05）

### 方法
- 新增 `policy/act_ensemble.py`：K 个 ACT ckpt 各带 temporal ensemble 队列，逐帧动作等权平均；
  与 ACTBase 同接口，eval/RL 零侵入。`scripts/eval_ensemble_probe.py`：K 策略 × R 轮 ×12 变体复评。
- K=3 选同一次训练的 030k/035k/040k（相邻 ckpt 对像素噪声响应不相关，权重天然可用）。

### 结果（verify_out/n8_ensemble_probe.json）

| 策略 | 3 轮复评 | 合并 36 trial | 单轮极差 |
|---|---|---|---|
| K=1 040k | 42 / 58 / 42% | 17/36 = 47% | 16.7pp |
| **K=3 30/35/40k** | **58 / 58 / 58%** | 21/36 = **58%** | **0pp** |

K=3 三轮 per-variant 成功掩码完全相同（成功的 7 个变体逐轮一致）——ensemble 同时做了两件事：
① 均值 +11pp 越过 ≥55% 闸门；② 把含 1px 渲染噪声的随机评测变成**确定性指标**
（n6 中 best-actor 回滚信号失真问题随之消失，n10 曲线可直接判读）。

### 决策
- n8b（多种子训练）与 n9（扩示教）的目的（过 55% + 降方差）已由 n8 单独达成，暂缓；
  保留为 n10 卡壳时的后备（PLAN n9a/n9c 方案已备好，扫描器已加 --rings/--angles 参数）。
- 直接进入 n10：configs/residual_td3_transfer_k3.yaml（act_ckpts 三键，
  out_dir=checkpoints/rl_m4_k3），残差切片/TD3 超参与 n6 完全一致，已冒烟通过。

## M4 n10：确定性 K=3 基线上重跑残差 TD3 —— 局部真实、整体不显著（2026-10-05）

### 设置
K=3 ensemble（030/035/040k）冻结，40-D 双手指残差 + TD3 全部超参与 n6 一致；
50025 transitions / 445 episodes / 44 eval 点 / 19 次回滚。确定性评测使训练曲线首次可直接判读：
58% 起步，6.8k 起 8/12（67%）与 6/12、7/12 交替振荡，末段在 7↔8/12 徘徊；训练 rollout 63%→60% 无趋势。

### 跨进程 A/B 复评（verify_out/rl_m4_k3_ab_probe.json）
零残差基线 5 轮（n8 的 3 轮 + 独立进程 2 轮）vs best actor 6 轮（3+finalize 1+独立 2 轮）：

| 口径 | 零残差 | best 残差 actor |
|---|---|---|
| 总成功 trial | 35/60 = 58.3% | 47/72 = 65.3% |
| 整体 Fisher | — | p=0.47（不显著） |
| **变体6 (+4,+7)ps11** | **0/5** | **5/6（p=0.015）** |
| 其余 11 变体掩码 | — | 与基线**逐变体完全相同** |

### 结论（问题 45）
- 比 n6 进一步但不达标：残差 RL 在确定性基线上产生了**可证认的真实增益**——把基线 5 轮恒败的
  变体6（catch 接近段毫米级误差）在 6 轮中修复 5 次（p=0.015）；但增益只此 1 个变体，
  整体 35/60→47/72 不显著（p=0.47），**4 个恒败变体（两个 ps4 锚点、ps2、另一个 ps19）
  掩码纹丝不动，90% 目标未达**。
- 机理判断：4 个恒败变体在 n5/n6 里都败在 carry/catch 途中掉笔，属**臂接近几何**问题，
  纯手指残差（M3 pick 任务够用）够不到这个误差方向。
- 流程坑（已修）：终评输出文件名曾硬编码 rl_m4_final.json，n10 终评覆盖过 n6 留档
  （已从 git 恢复）；修复为按 out_dir 命名 {run}_final.json。
- 另观察：残差闭环 actor 对 1px 渲染噪声比零残差基线敏感（best actor 变体6 偶发 1/6 回落），
  多轮跨进程复评仍是必要口径。
