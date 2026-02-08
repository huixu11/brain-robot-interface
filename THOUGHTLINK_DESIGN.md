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
- 输入：EEG 窗口特征（默认使用 **raw 特征**）
- 输出：`p(move)`
- 模型：线性分类器（Logistic Regression 级别）

Stage 2：Direction（Left/Right/Forward/Backward）
- 触发条件：Stage 1 判定 move
- 输出：`p(left/right/forward/backward)`（默认使用 **baseline-delta 特征**）
- 模型：线性分类器（softmax）

特征（强调推理速度，不引入重依赖）：
- EEG 时域能量特征：每通道去 DC 后的 `log-variance`（对跨 session/subject 更稳）
- EEG 频域近似：rFFT bins 估计 theta/alpha(beta) bandpower，并取 `log-bandpower` + `relative bandpower`（不依赖 SciPy）

Baseline correction（BCI 常用做法，提升稳定性）：
- 每个 chunk 使用已知 rest 段 `t=0..(cue_start-guard)` 计算一次 baseline 特征向量
- 为了兼顾 move 检测与方向分类，本仓库实现采用 **dual-feature**（对应 `examples/train_intent.py --feature-layout dual`，默认开启）：
  - Stage 1（move/rest）使用 `feat_move = feat_window`（raw，保留绝对能量尺度，提升触发率）
  - Stage 2（direction）使用 `feat_dir = feat_window - feat_baseline`（delta，减少跨 session 漂移，提升方向可分性）
  - Stage 1/Stage 2 使用 **两个独立 scaler**（`IntentModel.scaler` 与 `IntentModel.scaler_dir`）避免不同特征分布互相干扰

False trigger 训练控制（与赛题第 6 节对齐）：
- Stage 1 支持类别加权以抑制误触发（`examples/train_intent.py --stage1-balance` 或手动 `--stage1-w-rest`/`--stage1-w-move`）

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
- 方向 release（可选）：当方向置信度持续低于阈值（`p_dir_off` 或 `dir_margin`）达到 `dir_off_k` 次，输出 STOP 直到重新稳定
- `stop_on_dir_uncertain` 开关（默认开启）：若关闭，则在 MOVE 状态下方向不确定时不强制 STOP，而是保持上一方向（降低 STOP 抖动、提高 `move_coverage`，但可能抬高 `false_rate`）

推荐调参流程（优先降低 `false_rate`，再考虑压低 `onset_latency`）：
- 用 `examples/eval_closed_loop.py` 在 **test split** 上批量评测 `false_rate_global/move_coverage_global/switches_per_min`，避免只看单个 chunk。
- 优先调 Stage 1（move/rest）的滞后参数：`p_move_on/p_move_off/move_on_k/move_off_k/ewma_alpha`，先把误触发压住。
- 再调方向稳定参数：`dir_k/dir_off_k/dir_margin/p_dir/p_dir_off`，在不显著牺牲 `move_coverage` 的前提下减少抖动。
- 一个可复现实验起点（`update_hz=50`）：`ewma_alpha=0.15 p_move_on=0.68 p_move_off=0.50 move_on_k=5 move_off_k=3 dir_k=5 dir_off_k=2 dir_margin=0.06`

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

批量评测（得到更可信的结论，且便于调参）
- 单个 `.npz` chunk（15s）的指标只能代表一个样本，不能代表整体。应当在 **test split** 上汇总，并查看分布（p50/p95）而不是只看均值。
- 本仓库提供 `examples/eval_closed_loop.py`：不启动 MuJoCo，但复现 `intent_policy.py` 的闭环判决逻辑（滑窗 -> 模型 -> 稳定器 -> Action）并对多 chunk 汇总指标。
- 汇总推荐按时长加权（避免不同 `duration` 的 rest/move 时长差异带来偏差）：
  - `false_rate_global = sum(false_nonstop_rest_s) / sum(rest_total_s)`
  - `move_coverage_global = sum(nonstop_move_s) / sum(move_total_s)`
- `onset_latency` 的定义是 `t_pred_onset - cue_start`（数据 cue_start 固定为 3s），理想值接近 0s。

#### 6.8.1 `false_rate_global` 目标量级（项目 KPI，不是赛题硬门槛）

说明（严格对齐赛题表述）：`Requirement_doc.md` 在 “False Trigger Rate & Confidence Handling” 中要求你显式使用阈值/去抖/滞后等机制抑制误触发，但**没有给出数值型门槛**。因此下面的阈值是本项目为了工程落地与可扩展性讨论而设定的 KPI，用于调参选点与报告 tradeoff，并不声称是赛题强制要求。

建议的两档目标（按 rest 时间加权）：
- Demo 及格线：`false_rate_global <= 0.05`
- 更接近“可扩到 100 机器人”的安全量级：`false_rate_global <= 0.01`

为什么需要到 `0.01` 量级：在 one-to-many supervision 设定里，误触发会随机器人数量叠加。若单机器人在 rest 段误动比例为 `f`，则 N 台中“至少一台在误动”的概率上界约为 `1 - (1 - f)^N`。当 `f=0.05, N=100` 时几乎必然发生；当 `f=0.01` 时依然不低，但显著改善。因此最终展示建议至少达到 `<=0.05`，并在讨论可扩展性时展示向 `<=0.01` 收敛的策略与代价。

#### 6.8.2 如何达到上述目标（训练 + 稳定器的闭环调参流程）

关键原则（与赛题一致）：先把误触发压住，再在误触发可控的前提下压延迟/提升覆盖率，并明确展示 tradeoff。

1) 先建立“可信评估集”
- 不要只看单个 `.npz` chunk（15s）。使用 `examples/eval_closed_loop.py` 在一个 split 的多个 chunks 上汇总 `false_rate_global`，并查看 per-chunk 的 p50/p95。
- 建议使用 session split 做单用户校准（更贴近真实 BCI 部署），并留出 non-empty val 用于调参：

```powershell
# 例：单用户 a5136953，按 session 切分，留 1 个 session 做 val，1 个做 test
python examples\train_intent.py --split session --subject-id a5136953 --val-sessions 1 --test-sessions 1 --out artifacts\intent_user.npz

# 先在 val 上调稳定器参数（下面的 --subset val）
python examples\eval_closed_loop.py `
  --split session --subject-id a5136953 --val-sessions 1 --test-sessions 1 --subset val `
  --mode model --model artifacts\intent_user.npz --update-hz 50

# 选定参数后，只跑一次 test 报最终结果
python examples\eval_closed_loop.py `
  --split session --subject-id a5136953 --val-sessions 1 --test-sessions 1 --subset test `
  --mode model --model artifacts\intent_user.npz --update-hz 50
```

2) 训练侧先把 Stage 1（move/rest）偏向安全
- 如果 `false_rate_global` 过高，优先在 Stage 1 加大 REST 权重（减少误触发），再去调稳定器；因为这属于“模型层面减少误触发”，比纯靠阈值更稳健。
- 本仓库支持：
  - `--stage1-balance`（按训练集比例自动设权重）
  - 或手动：`--stage1-w-rest <wR> --stage1-w-move <wM>`（例如 `wR>wM`）

```powershell
# 例：更重惩罚误触发（REST 侧权重大）
python examples\train_intent.py --split session --subject-id a5136953 --val-sessions 1 --test-sessions 1 `
  --stage1-w-rest 2 --stage1-w-move 1 --out artifacts\intent_user_safe.npz
```

3) 稳定器调参，把 `false_rate_global` 作为硬约束
- 调参顺序（推荐）：先调 move/rest 再调 direction。
- Move/rest（主要影响 `false_rate_global` 与 `trigger_rate`）：
  - 增大 `p_move_on`、增大 `move_on_k`、减小 `ewma_alpha`：更保守，误触发更低，但触发更慢/覆盖更低
  - 减小 `p_move_on`、减小 `move_on_k`、增大 `ewma_alpha`：更敏感，触发更快/覆盖更高，但误触发更高
- Direction（主要影响 `switches_per_min`、MOVE 内 STOP 抖动、release 行为）：
  - 增大 `dir_margin`、增大 `dir_k`：更稳，但可能出现 MOVE 内 STOP 或延迟方向切换
  - `stop_on_dir_uncertain=True`（默认）：更安全，方向不确定时允许 STOP，通常降低误触发，但可能降低 `move_coverage`
  - `stop_on_dir_uncertain=False`：更连续，但更容易在 rest 段“拖尾”导致 false triggers（需要配合更强的 move-off 条件）

4) 用批量评估挑一个“满足目标的 Pareto 点”
- 在 val 上，以 `false_rate_global <= 0.05` 作为硬约束，选择 `move_coverage_global` 更高、`switches_per_min` 更低、`onset_latency_p95` 更小的参数组合。
- 如果目标是 `<= 0.01`，通常需要更保守的 `p_move_on/move_on_k`，并接受更大的 `onset_latency` 或更低的 `move_coverage`，并在报告中明确 tradeoff。

5) 最后用 `examples/intent_policy.py` 做仿真闭环 demo
- 先 `--mode oracle` 验证闭环与指标计算
- 再 `--mode model` 展示真实解码 + 稳定器，并输出 `inference_ms` 与闭环指标

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
- 新增 `examples/eval_closed_loop.py`：批量闭环指标评测（不启动 MuJoCo，用于在 test split 上汇总 `false_rate/onset_latency` 等）  

建议的实现里程碑（便于在现有仓库上逐步验证，而不是一次性大改）：  
1. 先只实现 `Action.BACKWARD` 并在仿真里验证负 `vx` 的行为（例如新增一个非交互脚本 `examples/action_sequence.py` 顺序执行 FORWARD/BACKWARD/LEFT/RIGHT/STOP）。  
2. 写一个“完美解码器”版本的 `intent_policy.py`：直接用数据标签在 t=3..3+duration 输出对应动作，验证闭环与日志指标计算。  
3. 再替换为真实模型推理（Stage 1/Stage 2 + 稳定器），并以日志量化 latency/stability/false triggers 的 tradeoff。  

