# ThoughtLink - From Brain to Robot 代码实现设计文档（严格遵守 `Requirement_doc.md`）

VC Track supported by Kernel & Dimensional

## 1. Goals and Motivation（对应 `Requirement_doc.md` 第 1 节）

随着 AI 系统能力提升，难点从“看见与生成”转向“实时执行与协调”。在机器人场景中，关键瓶颈是当环境出现歧义或异常时，人类判断如何以低延迟、可扩展的方式注入到机器人执行中。

本项目遵循赛题要求：
- 目标不是控制电机或关节
- 目标是构建“意图与编排层”(intent + orchestration layer)
- 用非侵入式脑信号解码高层离散指令，并在仿真中闭环执行

## 2. Hero Features（对应 `Requirement_doc.md` 第 2 节）

核心能力：Brain-to-Instruction Mapping
- 将非侵入式脑信号解码为离散、高层机器人指令
- 把神经活动当作多变量时间序列的“意图信号”，而非连续控制量
- 映射到机器人已经会执行的动作（本仓库的离散动作 -> 速度指令 -> MuJoCo 仿真）

编排层结果：
- 意图级控制，而不是 teleoperation
- 通过显式的置信度与稳定性机制，实现快速且安全的人工介入
- 结构上可扩展为“一人监督多机器人”的 fan-out 控制

## 3. Example Directions（本项目选定方向，对应 `Requirement_doc.md` 第 3 节）

赛题要求“选一条方向深入”。本设计主方向选择 **Hierarchical Intent Models**（先 `move/rest`，再分类方向以降低误触发），并在实现中采用 **Temporal Intent Decoding** 常用的 smoothing/hysteresis/debouncing 机制来保证时序稳定性。

落地目标（按赛题示例方向明确动作集合）：
- 解码 `left` / `right` / `forward` / `backward` 的意图
- 使用 smoothing / hysteresis / debouncing 抑制振荡
- 在 humanoid simulation 中验证意图到动作的闭环

成功标准严格按赛题：意图的清晰、稳定、时序优先于运动复杂度。

## 4. How to Think About This（对应 `Requirement_doc.md` 第 4 节）

AI 角色是 intent interpreter，不是 controller：
- 机器人已经会走（本仓库通过 velocity-conditioned ONNX policy 驱动 MuJoCo）
- 人类决定“应该发生什么”
- AI 决定“意图何时成立”“映射到哪个离散指令”“是否执行”

工程核心是在实时约束下平衡：
- Accuracy
- Latency
- Stability
- False triggers

## 5. Getting Started & Resources（对应 `Requirement_doc.md` 第 5 节）

遵循赛题建议：Start Simple, Then Scale

建议实现顺序：
1. Stage 1 二分类：`move` vs `rest`
2. Stage 2 多分类：`left/right/forward/backward`
3. 加入显式的时间稳定器（阈值、滞后、去抖、EWMA）
4. 在仿真中闭环测试与日志量化

推荐 baseline（与赛题一致）：
- Logistic regression（线性分类器）
- Linear/Kernel SVM（可选）
- Random forest（可选）
- 简单 1D CNN（可选）

本仓库可用基础设施（与赛题描述对齐）：
- 预录数据：`robot_control_data/data/*.npz`
- 仿真：`src/bri/backends/sim.py`（MuJoCo + ONNX policy）
- 控制入口：`src/bri/controller.py`（对外暴露 `Controller.set_action(Action)`）
- 现有示例：`examples/minimal_policy.py`

## 6. System Design（实现设计）

### 6.0 基于当前仓库代码的增量改造原则（不是从零重写）

本仓库 `bri` 已经提供了“高层离散动作 -> 速度指令 -> 仿真/硬件执行”的完整执行链路，因此 ThoughtLink 的实现应当以最小侵入方式接入：
- 保持 `bri.Controller` 的用法不变：外部只需要周期性调用 `Controller.set_action(Action)`。
- 不改 `src/bri/backends/sim.py` 的仿真策略，只通过 `CmdVel(vx, yaw_rate)` 输入驱动现有 ONNX velocity policy。
- 仅对动作空间做**向后兼容的增量扩展**：增加 `Action.BACKWARD`，并在动作到速度映射里实现 `vx` 为负。
- 新增的“意图解码层”代码放在 `src/thoughtlink/*`，避免和 `bri` 的机器人接口职责耦合。

这样可以确保你在现有代码基础上迭代，且 `examples/minimal_policy.py` 等现有用例仍可保持可运行（不受影响）。

### 6.1 数据契约（来自 `robot_control_data/README.md`）

每个 `.npz` chunk 为 15 秒：
- EEG：`feature_eeg`，形状 `(7499, 6)`，约 500Hz，单位 microvolts
- TD-NIRS：`feature_moments`，形状 `(72, 40, 3, 2, 3)`，约 4.76Hz
- 标签：`label` 字典，包含 `label`、`subject_id`、`session_id`、`duration`

时间轴（赛题与数据 README 一致）：
- t=0..3s rest
- t=3s cue 出现
- cue 持续 `duration` 秒

EEG 通道（数据 README 给出）：
- 0 AFF6
- 1 AFp2
- 2 AFp1
- 3 AFF5
- 4 FCz
- 5 CPz

### 6.2 机器人指令空间（严格包含 backward）

赛题示例方向明确包含 `left/right/forward/backward`。

本仓库当前动作集合在 `src/bri/command/actions.py` 只有：`FORWARD/LEFT/RIGHT/STOP`。

为严格满足赛题方向，本设计要求在 `bri.Action` 中新增：
- `Action.BACKWARD`

并在 `src/bri/command/action_controller.py` 增加动作到速度映射：
- BACKWARD：`vx = -forward_speed`

STOP 作为 `rest` 的执行动作保留（用于 Stage 1 的 rest 输出，以及超时安全兜底）。

### 6.3 Label 规范化与“意图到动作”映射

数据 label 字符串存在拼写差异（README 提到 `Left First`、`Both Firsts`）。必须做规范化，内部 canonical labels：
- `LEFT_FIST`
- `RIGHT_FIST`
- `BOTH_FISTS`
- `TONGUE_TAPPING`
- `RELAX`

将数据 cue 映射到赛题要求的 4 个方向指令：
- LEFT_FIST -> LEFT
- RIGHT_FIST -> RIGHT
- BOTH_FISTS -> FORWARD
- TONGUE_TAPPING -> BACKWARD
- RELAX -> REST（最终执行 STOP）

说明：数据没有原生 backward cue，本设计用 `Tongue Tapping` 作为 backward 代理，以便在不改变数据集的前提下覆盖 4 向指令集合。

### 6.4 从 chunk 到窗口（time-series intent）

固定长度滑窗用于训练和实时推理。

窗口参数以低延迟为目标，并允许 ablation：
- 默认 window：0.5s
- 默认 hop：0.1s
- 备选 window：1.0s（更稳但更慢）

窗口标注规则（严格利用 cue timing + label）：
- 对 `RELAX` chunk：全程标注为 `rest`
- 对其它 cue chunk：t=3..(3+duration) 标注为 `move`，其余时间标注为 `rest`

为降低过渡期噪声，增加 guard band：
- 设 `g`（例如 0.2s）
- 排除靠近 cue on/off 的窗口：`[3-g, 3+g]` 与 `[3+duration-g, 3+duration+g]`

### 6.4.1 训练/评估划分（Subject vs Session）

赛题关注的是“实时、稳定、低误触发”的闭环演示，并未要求跨参与者泛化必须成立。结合非侵入式 BCI 的常见现实约束（通常需要对单一用户做短暂校准），本项目同时支持两类划分方式：
- `split=subject`：按 `subject_id` 划分训练/验证/测试（更严格，用于评估跨人泛化，避免泄漏）。
- `split=session`：按 `session_id` 划分（通常配合 `--subject-id <id>` 进行单用户校准训练；同时避免同一录制 session 内泄漏）。
- `split=chunk`：仅用于快速 smoke（不作为最终严谨评估）。

### 6.5 模型（fast, simple baseline）

Stage 1：Move vs Rest
- 输入：窗口特征
- 输出：`p(move)`
- 模型：线性分类器（Logistic Regression 级别）

Stage 2：Direction（Left/Right/Forward/Backward）
- 触发条件：Stage 1 判定 move
- 输出：`p(left/right/forward/backward)`
- 模型：线性分类器（softmax）

特征（强调推理速度，不引入重依赖）：
- EEG 时域能量特征：每通道去 DC 后的 `log-variance`（对跨 session/subject 更稳）
- EEG 频域近似：rFFT bins 估计 theta/alpha(beta) bandpower，并取 `log-bandpower` + `relative bandpower`（不依赖 SciPy）

Baseline correction（BCI 常用做法，提升稳定性）：
- 每个 chunk 使用已知 rest 段 `t=0..(cue_start-guard)` 计算一次 baseline 特征向量
- 每个滑窗特征做 `feat_delta = feat_window - feat_baseline`，减少跨 session 的幅度尺度漂移

部署与推理：
- baseline 训练/推理优先使用 Numpy 线性模型（矩阵乘法，低延迟，依赖最小）
- 可选：如需统一部署形式，可将线性模型导出 ONNX 并用 `onnxruntime` 推理（与 `src/bri/backends/sim.py` 的 onnx 使用方式一致）

### 6.6 Temporal Stability（显式稳定器，赛题强要求）

稳定器实现为独立模块，输入为模型概率输出，输出为最终离散 Action。

代码实现：`src/thoughtlink/stability.py`（`IntentStabilizer` + `StabilityConfig`）。

机制组合（赛题点名的阈值/去抖/滞后/平滑）：
- EWMA 概率平滑
- Move 的滞后阈值：`p_move_on` 与 `p_move_off`
- 连续窗口去抖：进入/退出 move 需要连续 K 次满足
- 方向切换去抖：方向 top-1 需要连续 K 次且 `>= p_dir` 才切换
- 方向置信度“峰值差值”约束：`top1 - top2 >= dir_margin` 才允许执行非 STOP（降低 rest 期间误触发）
- 方向 release：当方向置信度持续低于阈值（`p_dir_off` 或 `dir_margin`）达到 `dir_off_k` 次，输出 STOP 直到重新稳定

推荐 demo 参数（优先降低 `false_rate`，再考虑压低 `onset_latency`）：
- `dir_k=5`, `dir_off_k=2`, `dir_margin=0.06`

状态机状态：
- REST
- MOVE_LEFT
- MOVE_RIGHT
- MOVE_FORWARD
- MOVE_BACKWARD

与 `bri.Controller` 的安全机制配合：
- `hold_s` 必须大于 hop（建议 `hold_s >= hop * 3`），否则动作更新间隔会触发自动 STOP
- `smooth_alpha` 仅用于速度命令平滑，不替代意图层稳定器

### 6.7 Execution & Testing Script（赛题提供的“连接仿真测试脚本”在本仓库的实现）

赛题提到“ready-made script connects predictions to simulation”。本仓库已有 `examples/minimal_policy.py`（键盘 -> Action -> 仿真）。

本设计要求新增两类脚本，以满足赛题的闭环测试：
- `examples/train_intent.py`
- `examples/intent_policy.py`

`examples/intent_policy.py` 的职责：
1. 启动 `Controller(backend="sim")`
2. Step 2（完美解码器 / oracle）：先直接用 `.npz` 的 label + cue timing 产生 action 序列，验证闭环与日志指标计算
3. Step 3+（真实解码器）：再替换为滑窗推理 + 稳定器 -> `Action`
4. 调用 `ctrl.set_action(action)` 驱动仿真
5. 记录日志以量化 latency/stability/false triggers

实现接口（便于评测与复现）：
- `--mode oracle|model`
- `--model artifacts/intent_*.npz`（`examples/train_intent.py` 输出）

说明：`examples/minimal_policy.py` 使用 `termios/tty`，在 Windows 环境不可用；`examples/intent_policy.py` 设计为不依赖这些交互库。

### 6.8 Evaluation Criteria（严格对齐 `Requirement_doc.md` 第 6 节）

Intent decoding performance
- Stage 1 accuracy
- Stage 2 accuracy
- 混淆矩阵

Inference speed & latency
- 单次推理耗时均值与 p95
- 端到端延迟分解：window + debounce(K*hop) + 推理耗时

Temporal stability
- 每分钟 action 切换次数
- 非 STOP 状态平均驻留时间

False triggers
- rest 期间被判 move 的比例
- rest 期间非 STOP 的累计时长

指标落地：`examples/intent_policy.py` 在 `--mode model` 下输出：
- `inference_ms mean/p95`
- `onset_latency/release_latency`（端到端 timing）
- `switches_per_min/dir_switches/stop_toggles`（稳定性）
- `false_nonstop_rest_s/false_rate`（误触发）

Bonus（赛题加分项）
- latency-accuracy tradeoff：对 window/hop/K/阈值做系统 ablation
- failure modes：误触发类别、边界抖动、跨 subject 泛化

### 6.9 Scalability（对齐赛题“监督 100 机器人”的可扩展性要求）

本仓库的 `Controller` 是单机器人控制循环。

为了结构上支持 one-to-many supervision：
- 意图解码器与稳定器输出“高层 action 事件流”
- 通过一个轻量 router 将 action fan-out 到多个 `Controller` 实例
- router 支持将 action 广播给所有机器人，或按 robot_id 选择性下发

该设计保持“意图层”与“执行层”解耦，便于未来扩展到多机器人。

### 6.10 可选扩展（对应赛题其它方向）  

在 baseline 达标后，可按赛题建议做增量扩展，并用实测证明收益大于成本：  
- Phase-Aware Modeling：利用已知 cue 时序（t=3 与 `duration`）把 move 段划分为 initiation/sustain/release，用于训练更稳定的过渡检测。  
- Interpretability & Visualization：对窗口特征或模型中间表示做降维可视化（PCA/UMAP），展示意图随时间演化。  
- Longer temporal models：仅在确有收益时引入 RNN/state-space/transformer，并将推理延迟作为硬约束进行对比评测。  

## 7. Why It Matters（对应 `Requirement_doc.md` 第 7 节）

赛题关注的是当大规模机器人系统遇到歧义时，人类判断如何以机器速度、安全注入。ThoughtLink 的价值在于提供一个非侵入式、可部署的意图通道，使一人可监督多机器人并在关键时刻快速介入。

## 8. Repo 级改动清单（确保设计可在本仓库落地）

为满足 `Requirement_doc.md` 明确要求的 `left/right/forward/backward` + 稳定性 + 仿真闭环，本设计在“现有代码基础上”做以下增量修改（尽量不破坏现有接口）：  
- 修改 `src/bri/command/actions.py`：新增 `BACKWARD`（保持枚举兼容，原有动作不变）  
- 修改 `src/bri/command/action_controller.py`：在 `get_cmd_vel()` 增加 BACKWARD 分支，输出 `CmdVel(vx=-forward_speed, ...)`  
- 新增 `src/thoughtlink/*`：数据读取/滑窗/特征/模型/稳定器（只负责“脑信号 -> Action”）  
- 新增 `examples/train_intent.py`：离线训练与评估（输出模型与配置）  
- 新增 `examples/intent_policy.py`：execution & testing script（模型预测 -> 稳定器 -> `Controller.set_action` -> MuJoCo 仿真）  

建议的实现里程碑（便于在现有仓库上逐步验证，而不是一次性大改）：  
1. 先只实现 `Action.BACKWARD` 并在仿真里验证负 `vx` 的行为（例如新增一个非交互脚本 `examples/action_sequence.py` 顺序执行 FORWARD/BACKWARD/LEFT/RIGHT/STOP）。  
2. 写一个“完美解码器”版本的 `intent_policy.py`：直接用数据标签在 t=3..3+duration 输出对应动作，验证闭环与日志指标计算。  
3. 再替换为真实模型推理（Stage 1/Stage 2 + 稳定器），并以日志量化 latency/stability/false triggers 的 tradeoff。  

