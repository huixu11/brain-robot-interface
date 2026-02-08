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

可选：自动找稳定器参数（推荐用于快速迭代）
- 本仓库提供 `examples/tune_stability.py`：在 `val` 上自动搜索一批 `StabilityConfig` 候选（默认随机 200 组），以 `false_rate_global <= target` 为硬约束，优先最大化 `move_coverage_global`，并打印可直接复制到 `eval_closed_loop.py / intent_policy.py` 的参数串。
- 示例（单用户 session split，先在 val 上找参数，再查看它在 test 的表现）：

```powershell
python examples\tune_stability.py `
  --split session --subject-id a5136953 --val-sessions 1 --test-sessions 1 `
  --model artifacts\intent_user.npz --update-hz 50 `
  --target-false-rate 0.05 --max-evals 200
```
- `--objective robust` 会用 `val+test` 的最坏情况打分（更鲁棒，但等价于“用 test 参与调参”，不适合作为严格的最终评测流程）。

如何在 session 数有限时做“训练/调参/评测”切分（非常重要）
- 本数据集是按 `subject_id/session_id` 组织的，**跨 session 漂移**往往是 `false_rate_global` 爆炸的主要原因（尤其是 Stage 1 把 REST 误判为 MOVE）。因此调参时必须确保训练集包含足够多的 session，避免“train 只有 1 个 session”导致的偶然性。
- 如果你想严格做 `train/val/test = 2/1/1`（推荐，用于自动调参 + 最终评测分离），需要该 subject 至少有 4 个 session。实践中建议挑“session 最多的 subject”来做稳定器搜索与展示，避免 session 不足导致的调参偏差。
- 如果某个 subject 只有 3 个 session（例如 `a5136953`），无法同时做到 `train=2,val=1,test=1`。此时有两种务实做法：
  - 做法 A（更严谨）：轮换 test session 做 3 折 cross-session（每次 `train=2,test=1`），把 `false_rate_global/move_coverage` 在 3 个 test 上取均值/分位数，避免挑选有利 session。
  - 做法 B（更省事）：固定 `train=2,test=1`，用 `examples/tune_stability.py` 在 train 上找参数，然后只在 test 上复核一次（注意：没有 val，调参结论不如做法 A 稳健）。

示例：只有 3 个 session 的 subject（无 val）

```powershell
# 训练：train=2 sessions, test=1 session
python examples\train_intent.py --split session --subject-id a5136953 --val-sessions 0 --test-sessions 1 `
  --stage1-balance --out artifacts\intent_a513_2sess.npz

# 自动找参数：在 train(2 sess) 上搜索，并打印 best_cfg 在 test(1 sess) 的复核结果
python examples\tune_stability.py --split session --subject-id a5136953 --val-sessions 0 --test-sessions 1 `
  --model artifacts\intent_a513_2sess.npz --update-hz 50 --target-false-rate 0.05 --max-evals 200
```

示例：session >= 4 的 subject（推荐，有 val）

```powershell
# 训练：train=2 sessions, val=1 session, test=1 session
python examples\train_intent.py --split session --subject-id d696086d --val-sessions 1 --test-sessions 1 `
  --stage1-balance --out artifacts\intent_d696.npz

# 自动找参数：在 val 上搜索，再在 test 上复核
python examples\tune_stability.py --split session --subject-id d696086d --val-sessions 1 --test-sessions 1 `
  --model artifacts\intent_d696.npz --update-hz 50 --target-false-rate 0.05 --max-evals 200
```

常见问题：`false_rate_global` 偏高如何处理（按优先级排查）
- 先看 Stage 1（move/rest）混淆矩阵：如果 `REST -> MOVE` 大量发生，这是 rest 段误动的直接来源。此时不要优先微调 direction 参数，而应先让 Stage 1 更偏安全。
- 优先手段 1（训练侧）：对 Stage 1 加权抑制误触发。

```powershell
# 自动按类频率平衡（常用起点）
python examples\train_intent.py --split session --subject-id a5136953 --val-sessions 1 --test-sessions 1 `
  --stage1-balance --out artifacts\intent_user_bal.npz

# 或更强的 REST 权重（更安全，但会牺牲触发率/覆盖率）
python examples\train_intent.py --split session --subject-id a5136953 --val-sessions 1 --test-sessions 1 `
  --stage1-w-rest 2 --stage1-w-move 1 --out artifacts\intent_user_safe.npz
```

- 优先手段 2（稳定器侧）：在不改变模型的前提下，提高 move-on 门槛（降低误触发），并加快 move-off（减少 rest 段拖尾）。典型方向是：提高 `p_move_on`/`move_on_k`，同时提高 `p_move_off` 或降低 `move_off_k`。
- 如果你为了提高 `move_coverage` 关闭了 `stop_on_dir_uncertain`（`--no-stop-on-dir-uncertain`），要特别关注 `release_rate/no_release`。MOVE 状态下“保持上一方向”容易把动作拖到 rest 段，导致 `false_rate_global` 上升；这时 move-off 需要更严格（更快回到 REST）。

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

#### 6.10.1 充分利用多用户数据：Multi-Task / Lifelong Learning（ELLA-style）

赛题本身不要求“跨用户零样本泛化”，但在实际非侵入式 BCI 中，“单用户数据少 + 跨 session 漂移”常常是误触发（`REST -> MOVE`）和不稳定的根因之一。为了在**不牺牲实时推理速度**（仍保持线性模型点积级别）的前提下更充分利用多个人的数据，可采用 multi-task learning，并进一步用 ELLA（Efficient Lifelong Learning Algorithm, Ruvolo & Eaton）式的“共享基 + 每用户稀疏系数”来做快速校准与持续学习。

核心思想（把每个用户当作一个 task）：
- 每个用户都有自己的解码器参数 `w_u`（因为脑信号个体差异大）
- 但这些 `w_u` 并不是彼此独立，而是可以表示为少量共享“基向量” `L` 的线性组合：`w_u = L s_u`
- `L` 由多用户数据学到，捕获跨人的共性；`s_u` 是用户特定的稀疏系数，可用少量校准数据快速求解（few-shot / per-user calibration）

在本项目两阶段解码器上的具体参数化（保持 fast + simple）：
- Stage 1（move/rest，二分类）：
  - 对每个用户 `u`：`w_u_move = L_move s_u_move`
  - 预测：`p(move|x) = sigmoid(w_u_move^T x)`
  - Stage 2（direction，4 分类）：
    - 对每个用户 `u`：`W_u_dir = L_dir S_u_dir`，其中 `W_u_dir` 是 `(d,4)` 权重矩阵，`L_dir` 是 `(d,k)` 字典，`S_u_dir` 是 `(k,4)` 系数矩阵
    - 预测：`p(dir|x) = softmax(W_u_dir^T x)`
  - 推理成本：与当前 baseline 相同数量级（仍是矩阵乘法/点积），满足赛题的 “fast inference matters more than model size”。

  常见疑问（实现选择的理由）：
  - 我们到底在训练什么模型？
    - Stage 1 是二分类 Logistic Regression（`sigmoid`），输出 `p(move)`，用于抑制误触发的第一道关口。
    - Stage 2 是多分类 Softmax Regression（`softmax`），只在稳定器判定 MOVE 时才输出方向 `LEFT/RIGHT/FORWARD/BACKWARD`。
  - 为什么“全量数据”训练也会很快？
    - 数据量本质上不算大（本仓库大约 `900` 个 15s chunk，滑窗后是十几万级窗口），而线性模型每步更新是 `O(N*d)`（`d` 约几十维），Numpy 在 CPU 上很容易跑到秒级或几十秒级。
    - 这也是刻意遵循 `Requirement_doc.md` 的建议: 先用简单强基线（logistic/softmax/SVM 等），优先保证推理速度与闭环稳定性，再考虑更复杂模型。
  - 用更多人的数据会更准吗？
    - 往往更“稳”（尤其是跨 session 漂移更不容易把 REST 判成 MOVE），但前提是严格 split 避免泄漏，并且最终仍需要对目标用户做少量校准（非侵入式脑信号个体差异很大）。
    - 因此推荐的利用方式不是“直接混在一起当 i.i.d.”，而是:
      - 用多用户数据学共享结构（全局模型或共享基 `L`）
      - 用目标用户少量 session 解出个体系数（`s_u`）或做轻量微调

  稳定器参数能否只用“两段 session”自动找？
  - 可以，但需要明确“调参数据”和“最终评测数据”的边界，避免只在同一段数据上自嗨。
  - 当目标 subject 只有 3 个 session（例如 `a5136953`）时，一个实用且不泄漏的流程是：
    - Step A: 用 `train_ella_intent.py` 随机挑 1 个 `eval` session 做最终评测，其余 2 个 session 作为 `calib`（用于训练/调参）。
    - Step B: 在这 2 个 `calib` session 内，用 `tune_stability.py --split session --val-sessions 1 --test-sessions 1` 自动搜索稳定器参数（val 约束 false_rate，test 作为 sanity check）。
    - Step C: 把找到的参数固定住，只在 `eval` session 上用 `eval_closed_loop.py --session-id <EVAL>` 汇报最终 `false_rate_global/move_coverage/onset_latency`。
  - 如果目标 subject 有 4 个 session，则更推荐 `train/val/test = 2/1/1`（val 专职调参，test 专职汇报），结论更稳健。

  ELLA 闭环指标没达标时，下一步怎么做（Runbook）
  - 典型现象：在 held-out `eval` session 上，`false_rate_global > 0.05`（误触发偏高），或 `move_coverage_global` 过低（不触发/不持续）。
  - 原则（对齐 `Requirement_doc.md`）：先把误触发压住（安全优先），再在不破坏 false_rate 的前提下尽量提高覆盖与降低 latency。

  1) 先只用 calib sessions 自动找稳定器参数（不碰 eval）
  ```powershell
  # calib sessions 内部做 val/test，再把最优参数带去 eval session 复核
  python examples\tune_stability.py `
    --split session --subject-id <SUBJECT_ID> `
    --session-id <CALIB_SESSION_1> --session-id <CALIB_SESSION_2> `
    --val-sessions 1 --test-sessions 1 `
    --model artifacts\intent_ella_<SUBJECT_ID>.npz `
    --update-hz 50 `
    --target-false-rate 0.05 `
    --min-move-coverage 0.15 --min-trigger-rate 0.6 `
    --max-evals 500 --objective robust
  ```
  - 关键点：`--min-move-coverage/--min-trigger-rate` 用来防止“全程 STOP”这种退化解（false 低但不可用）。
  - `--objective robust` 会对候选参数在 (val+test) 上更保守，减少只在某一段 session 上碰巧表现好的概率。

  2) 用上一步的 `best_cfg` 去 eval session 做一次最终复核
  ```powershell
  python examples\eval_closed_loop.py `
    --split all --subset all `
    --subject-id <SUBJECT_ID> --session-id <EVAL_SESSION_ID> `
    --mode model --model artifacts\intent_ella_<SUBJECT_ID>.npz `
    --update-hz 50 `
    <把 tune_stability 打印的 best_cfg 整段粘到这里>
  ```
  - 如果 `false_rate_global <= 0.05` 且 `move_coverage_global` 也满足 demo 需要: 这个参数点可作为展示点，并在报告里对比 “更激进的低延迟配置 vs 更保守的低误触发配置”。

  3) 如果 calib 上能达标，但 eval 上“0 触发/覆盖很低”
  - 这通常说明 “跨 session 漂移” 让 Stage 1 的 `p(move)` 分布整体变低，保守门槛推不过去。
  - 不建议为了提高覆盖直接在 eval 上调参（会泄漏）。推荐两种不泄漏的解决方案：
    - 方案 A（推荐，结论更稳）：做 cross-session 轮换（3 折）。每次把 1 个 session 作为 eval，其余 2 个 session 调参，然后汇报 3 个 fold 的 `false_rate_global/move_coverage` 分位数或均值，避免“挑最有利的 session”。
    - 方案 B（演示优先）：换一个 session 数更多的 subject（>=4 session），严格做 `train/val/test = 2/1/1`，让稳定器搜索更稳健。

  4) 如果怎么调稳定器都压不住误触发（`REST -> MOVE` 太多）
  - 这是“模型层面”的问题，优先修 Stage 1:
    - 增加 REST 的训练权重/采样权重（让 Stage 1 偏安全）
    - 选择更抗漂移的特征与基线策略（例如继续使用 `baseline=pre_cue`，并避免窗口跨 cue 边界的泄漏）
    - 在 ELLA 的 basis 学习里优先用 `--basis-task session` 来覆盖漂移形态
    - 对 ELLA：优先让 Stage 1 也使用 delta 特征（`--feature-mode-move delta`，默认 `auto` 在 `baseline=pre_cue` 时会选 delta），减少跨 session 的整体偏移导致的 “eval session 完全不触发/触发很少”。
  - 只有当 Stage 1 输出靠谱，稳定器的阈值/滞后才不会被迫调到极端保守。

  这个阶段建议 commit 什么（让你每一步都可复现）
  - 需要 commit：
    - 代码改动（脚本/模型/稳定器）与 `THOUGHTLINK_DESIGN.md` 的流程与结论更新
    - 推荐额外加一个小的“实验记录文件”（例如 `reports/ella_<subject>_<eval_session>.md`），记录:
      - session 划分（eval/calib）
      - 训练命令（`train_ella_intent.py`）
      - 调参命令（`tune_stability.py`）与最终采用的 `best_cfg`
      - eval session 的关键指标（false_rate_global/move_coverage/onset_latency/switches_per_min）
  - 不建议 commit：
    - `robot_control_data/`（数据）
    - `artifacts/*.npz`（模型权重通常较大，建议作为运行产物保存或在提交说明里给出生成命令；如 hackathon 需要交付权重文件，可打包为 release/附件而非 git 版本控制）

  #### 6.10.2 本次迭代做了哪些更新？问题如何被定位与解决？

  这部分用于记录“从现象到修复”的工程链路，方便复现与向评委解释 tradeoff（同样对齐 `Requirement_doc.md` 的评价点：false trigger、stability、latency、scalability）。

  本次新增/更新（与本仓库代码一致）：
  - `examples/train_ella_intent.py`：
    - 新增 `--basis-task {session,subject}`。推荐 `session`，让共享基覆盖更多 session 漂移形态。
    - 新增 `--feature-mode-move {auto,raw,delta}`。用于控制 Stage 1（move/rest）是否使用 `pre_cue` 基线差分特征；默认 `auto` 会在 `baseline=pre_cue` 时选 `delta`。
  - `THOUGHTLINK_DESIGN.md`：
    - 补充 ELLA 的可运行命令、数据切分防泄漏流程、以及遇到 no-trigger/全 STOP 退化时的处理路径（Runbook）。

  问题现象（闭环评测）：
  - 在 calib sessions 上调参可以达到较低 `false_rate_global`，但迁移到 held-out eval session 后出现:
    - `false_rate_global = 0` 且 `move_coverage_global = 0`
    - `trigger_rate = 0`、`no_trigger = 全部`（等价于全程输出 `STOP`）
  - 这不是“模型变好了”，而是稳定器过于保守导致的退化解。赛题需要的是“低误触发 + 仍然能及时执行意图”。

  根因定位（为什么会全 STOP）：
  - 稳定器是 gated 的：只有当 `p(move)` 连续超过阈值并通过滞后计数，才会从 REST 切换到 MOVE。
  - 跨 session 漂移会导致 Stage 1 的 `p(move)` 分布整体偏移（例如整体变低），从而让“在 calib 上刚好能触发的阈值/滞后”在 eval 上完全触发不了。
  - 因此，单纯追求更低 `false_rate` 的参数搜索，会倾向产生 “把门槛提到 eval 永远过不去” 的候选，造成全 STOP。

  解决方案（分两层，先评测流程再模型改进）：
  - 评测/调参层（不改模型也能避免退化解）：
    - `tune_stability.py` 搜索时必须加“可用性”约束，避免全 STOP：
      - `--min-move-coverage`（例如 `>= 0.15`）
      - `--min-trigger-rate`（例如 `>= 0.6`）
    - 使用 `--objective robust`，让候选同时在 calib 的 val+test 上更保守，减少偶然性。
    - 严格不在 eval session 上调参，只在 eval 上做一次最终复核。
  - 模型层（提升跨 session 稳定性，减少 `p(move)` 整体漂移）：
    - 对 ELLA Stage 1 使用 delta 特征（`--feature-mode-move delta`），与 `baseline=pre_cue` 配合，用 pre-cue 基线抵消 session 级别的整体偏移。
    - 仍可配合 Stage 1 加权（让模型更偏安全或更偏敏感），再在稳定器层做最终权衡。

  下一步（推荐执行顺序）
  - Step 1: 用 ELLA 重训模型并开启 Stage 1 delta：
  ```powershell
  python examples\train_ella_intent.py `
    --target-subject a5136953 `
    --eval-sessions 1 --calib-sessions 0 `
    --basis-task session `
    --baseline pre_cue `
    --feature-mode-move delta
  ```
  - Step 2: 只在 calib sessions 上调参（加可用性约束），并固定参数：
  ```powershell
  python examples\tune_stability.py `
    --split session --subject-id a5136953 `
    --session-id <CALIB_SESSION_1> --session-id <CALIB_SESSION_2> `
    --val-sessions 1 --test-sessions 1 `
    --model artifacts\intent_ella_a5136953.npz `
    --target-false-rate 0.05 `
    --min-move-coverage 0.15 --min-trigger-rate 0.6 `
    --max-evals 800 --objective robust
  ```
  - Step 3: 只在 eval session 上做最终汇报（避免泄漏）：
  ```powershell
  python examples\eval_closed_loop.py `
    --split all --subset all `
    --subject-id a5136953 --session-id <EVAL_SESSION_ID> `
    --mode model --model artifacts\intent_ella_a5136953.npz `
    --update-hz 50 `
    <粘贴 best_cfg>
  ```
  - Step 4: 如果仍出现 eval 触发极低，做 cross-session 轮换（3 折）汇总 p50/p95，避免单 session 偶然性；或选 session>=4 的 subject 做更严谨的 `2/1/1` split。

  #### 6.10.3 基于当前跑出来的结果，下一步具体做什么？

  你现在看到的核心矛盾是：为了把 `false_rate_global` 压到 `<=0.05`，稳定器很容易被调到“保守到 eval session 上完全不触发”的区域（`trigger_rate=0 / move_coverage=0`），这在赛题语境下属于不可用解。

  因此下一步的优先级应该是：
  1) 先保证 eval session 上“能触发且有覆盖”（避免全 STOP）
  2) 在 1) 的前提下，把 `false_rate_global` 压到 `<=0.05`
  3) 最后再看 `onset_latency`（更接近 3s 的 cue_start，即 latency 趋近 0）

  推荐执行（严格不泄漏的闭环流程）：
  1) 训练 ELLA 模型时，Stage 1 也用 delta 特征（对跨 session 更稳）：
  ```powershell
  python examples\train_ella_intent.py `
    --target-subject a5136953 `
    --eval-sessions 1 --calib-sessions 0 `
    --basis-task session `
    --baseline pre_cue `
    --feature-mode-move delta `
    --out artifacts\intent_ella_a5136953.npz
  ```
  2) 在 calib sessions 上跑 `tune_stability.py` 时加“可用性约束”，避免搜到全 STOP：
  ```powershell
  python examples\tune_stability.py `
    --split session --subject-id a5136953 `
    --session-id <CALIB_SESSION_1> --session-id <CALIB_SESSION_2> `
    --val-sessions 1 --test-sessions 1 `
    --model artifacts\intent_ella_a5136953.npz `
    --update-hz 50 `
    --target-false-rate 0.05 `
    --min-move-coverage 0.15 --min-trigger-rate 0.6 `
    --max-evals 800 --objective robust
  ```
  3) 把 `best_cfg` 固定住，只在 eval session 上复核一次（关键是 `trigger_rate > 0` 且 `move_coverage_global` 不为 0）：
  ```powershell
  python examples\eval_closed_loop.py `
    --split all --subset all `
    --subject-id a5136953 --session-id <EVAL_SESSION_ID> `
    --mode model --model artifacts\intent_ella_a5136953.npz `
    --update-hz 50 `
    <粘贴 best_cfg>
  ```

  如果仍然出现 “eval session no-trigger”：
  - 不要在 eval 上调参（会泄漏），按下面顺序处理：
    - 提高 `tune_stability.py` 的 `--min-move-coverage/--min-trigger-rate` 约束，让搜索拒绝“低覆盖但 false 很低”的候选。
    - 回到模型侧提升 Stage 1 的 recall（让 `p(move)` 在 eval session 上能跨过阈值）：
      - 继续使用 `--feature-mode-move delta`（优先）
      - 调整 Stage 1 校准的类别权重（`--stage1-w-move/--stage1-w-rest`）与正则（`--stage1-l2`），把输出分布推到更可触发的区间，再用稳定器把 false 压回去
    - 做 cross-session 轮换（3 折）：每次换一个 session 当 eval，用另两个做 calib 调参，最后汇总 3 个 fold 的 p50/p95（避免单 session 偶然性）。

  训练与更新流程（从简单到复杂，逐步加码）：
  1) 先做一个强 baseline：全数据（多用户）训练一个“全局模型”作为初始化，然后对目标用户用少量数据 fine-tune（最简单、最容易实现）。
  2) 再做 low-rank / shared-basis（ELLA 风格）：
     - 离线阶段：用多用户的监督数据学习共享字典 `L_move/L_dir`。
   - 校准阶段：对新用户只优化 `s_u_move/S_u_dir`（或再加一个轻量 bias），用很少的 chunk/窗口即可把 `REST -> MOVE` 的误触发压住，同时保持 `move_coverage`。
   - Lifelong（可选）：当同一用户的新 session 到来时，只更新 `s_u`（保持字典固定更安全）；或在离线再训练时批量更新字典。

为什么 ELLA 对本赛题有价值（与 `Requirement_doc.md` 的关注点对齐）：
- False trigger suppression：共享字典从多用户学习到“更通用的 rest/move 分界”，在目标用户数据不足或漂移时更不容易把 REST 判成 MOVE。
- Temporal stability：更可靠的 Stage1 概率输出可让稳定器阈值更“合理”，减少依赖极端保守的 `p_move_on/move_on_k`。
- Scalability：用户侧只需要存 `s_u`（小向量/小矩阵），且可在新 session 到来时快速更新，符合“可扩展系统工程”的范式。

评估方式（必须避免数据泄漏）：
- Cross-subject（`split=subject`）：用训练 subjects 学 `L`，对 test subject 用少量校准数据解 `s_u`，再在该 subject 的 held-out sessions 上评估 `false_rate_global/move_coverage/onset_latency`。
- Cross-session（`split=session`，单用户）：当目标用户 session 少时，用其它用户的数据学 `L`，再用目标用户的 1 个 session 解 `s_u`，在其余 session 上评估（模拟“快速校准 + 适配漂移”）。

  落地实现建议（仍保持仓库风格的最小依赖）：
  - 不引入重依赖（PyTorch/TF 不是必须）。字典学习与系数求解可以用 Numpy + 坐标下降/ISTA 的 L1（或先用 L2 作为简化版），并将 `L` 与 `s_u` 打包保存为 `.npz`。
  - 与现有闭环系统无缝对接：ELLA/MTL 只替换 `IntentModel.predict_*` 的参数来源；稳定器 `IntentStabilizer`、评测 `eval_closed_loop.py`、仿真脚本 `intent_policy.py` 的接口不变。

  本仓库的最小可运行实现（ELLA-style shared basis）：
  - 代码位置：
    - `src/thoughtlink/ella.py`：从多任务权重中提取共享基 `L`（SVD），并在低维系数空间训练/校准，再重构回全维权重（保持推理为点积/矩阵乘）。
    - `examples/train_ella_intent.py`：用“其它用户”学习 `L_move/L_dir`，再用目标用户的少量 session 做校准，输出标准 `IntentModel .npz`，可直接喂给 `intent_policy.py / eval_closed_loop.py`。
  - 重要实现细节（针对本数据集的 session 漂移）：
    - `--basis-task session`（推荐）：把“每个 recording session”当作一个 task 来学基，而不是把整个 subject 当作 1 个 task。原因是本赛题数据的主要难点之一是跨 session 漂移，用 session 作为 task 能显式让 `L` 覆盖更多漂移形态。
    - `--basis-task subject`：更简单，但 task 数更少，`basis_k` 可能受限，通常不如 session 版本稳健。

  用法示例（先训练 ELLA 模型，再用闭环评测脚本验证指标）：
  ```powershell
  # 1) 训练 ELLA-style 模型（输出 artifacts\intent_ella_<subject>.npz）
  python examples\train_ella_intent.py `
    --target-subject a5136953 `
    --eval-sessions 1 --calib-sessions 0 `
    --basis-task session `
    --basis-k-move 8 --basis-k-dir 8 `
    --baseline pre_cue
  ```
  `train_ella_intent.py` 会打印 `eval=[...] calib=[...]` 的 session_id 列表。为了避免数据泄漏，闭环评测时只在 `eval` session 上汇总指标：
  ```powershell
  # 2) 只评测 held-out 的 eval session（用 session-id 精确筛选）
  python examples\eval_closed_loop.py `
    --split all --subset all `
    --subject-id a5136953 --session-id <EVAL_SESSION_ID> `
    --mode model --model artifacts\intent_ella_a5136953.npz `
    --update-hz 50
  ```
  之后再按 `6.8.1` 的流程，用 `tune_stability.py` 在非 eval 的数据上自动找稳定器参数，并在 eval session 上复核 `false_rate_global` 是否达标（`<=0.05` 或向 `<=0.01` 收敛）。

## 7. Why It Matters（对应 `Requirement_doc.md` 第 7 节）

赛题关注的是当大规模机器人系统遇到歧义时，人类判断如何以机器速度、安全注入。ThoughtLink 的价值在于提供一个非侵入式、可部署的意图通道，使一人可监督多机器人并在关键时刻快速介入。

## 8. Repo 级改动清单（确保设计可在本仓库落地）

  为满足 `Requirement_doc.md` 明确要求的 `left/right/forward/backward` + 稳定性 + 仿真闭环，本设计在“现有代码基础上”做以下增量修改（尽量不破坏现有接口）：  
  - 修改 `src/bri/command/actions.py`：新增 `BACKWARD`（保持枚举兼容，原有动作不变）  
  - 修改 `src/bri/command/action_controller.py`：在 `get_cmd_vel()` 增加 BACKWARD 分支，输出 `CmdVel(vx=-forward_speed, ...)`  
  - 新增 `src/thoughtlink/*`：数据读取/滑窗/特征/模型/稳定器（只负责“脑信号 -> Action”）  
  - 新增 `examples/train_intent.py`：离线训练与评估（输出模型与配置）  
  - 新增 `src/thoughtlink/ella.py` + `examples/train_ella_intent.py`：可选的多用户共享基（ELLA-style）训练与快速校准（不改变闭环接口）  
  - 新增 `examples/intent_policy.py`：execution & testing script（模型预测 -> 稳定器 -> `Controller.set_action` -> MuJoCo 仿真）  
  - 新增 `examples/eval_closed_loop.py`：批量闭环指标评测（不启动 MuJoCo，用于在 test split 上汇总 `false_rate/onset_latency` 等）  

建议的实现里程碑（便于在现有仓库上逐步验证，而不是一次性大改）：  
1. 先只实现 `Action.BACKWARD` 并在仿真里验证负 `vx` 的行为（例如新增一个非交互脚本 `examples/action_sequence.py` 顺序执行 FORWARD/BACKWARD/LEFT/RIGHT/STOP）。  
2. 写一个“完美解码器”版本的 `intent_policy.py`：直接用数据标签在 t=3..3+duration 输出对应动作，验证闭环与日志指标计算。  
3. 再替换为真实模型推理（Stage 1/Stage 2 + 稳定器），并以日志量化 latency/stability/false triggers 的 tradeoff。  

## 9. Demo 与多机器人扩展（时间紧时的最小可交付）

Hackathon 时间有限时，建议用“可复现实验 + 现场闭环演示”的组合交付，而不是无限追求离线分类分数。本节给出 stop criteria、demo 流程与多机器人 fan-out 论证方式（对齐 `Requirement_doc.md` 的评价维度：Latency/Stability/False triggers/Scalability/Demo clarity）。

### 9.1 何时可以直接 Demo（Stop Criteria）

可以认为“够 demo 了”的最小条件（建议都在 held-out `eval` session 上验证）：
- `false_rate_global <= 0.05`（rest 段误动不超过 5%，满足 demo 及格线 KPI）
- `trigger_rate > 0` 且 `move_coverage_global > 0`（避免 `false_rate=0` 但全程 STOP 的退化解）
- `inference_ms p95` 明显小于 `1000/update_hz`（例如 `update_hz=50` 时，小于 20ms；本仓库线性模型通常远低于该值）

建议条件（更像“可用系统”而不是仅仅能跑）：
- `switches_per_min` 不要过高（避免抖动），且方向切换与 STOP toggles 可解释
- `onset_latency` 的 p50/p95 不要过大（能在 cue_start=3s 后较快触发）

如果短时间内做不到“低误触发 + 高覆盖 + 低 latency”同时达成：
- demo 时优先选择低误触发配置（安全优先），并在讲解中明确 tradeoff：更保守的阈值会提高 latency/降低覆盖。
- 同时保留一个“更激进”的参数点（更快触发但 false 稍高），用来展示 latency-false 的可调节性（赛题 bonus 点）。

### 9.2 Demo 流程（3-5 分钟可完成）

推荐用同一个 `.npz` chunk 演示，顺序如下：
1. Oracle 闭环（证明仿真链路无误）：
```powershell
python examples\intent_policy.py --mode oracle --backend sim --speed 5 --update-hz 50
```
2. Model 闭环（展示真实解码 + 稳定器）：
```powershell
python examples\intent_policy.py --mode model --model artifacts\intent_ella_<SUBJECT_ID>.npz --backend sim --speed 5 --update-hz 50 <稳定器参数>
```
3. 批量闭环指标（展示不是只挑一个 chunk）：
```powershell
python examples\eval_closed_loop.py --split all --subset all --subject-id <SUBJECT_ID> --session-id <EVAL_SESSION_ID> --mode model --model artifacts\intent_ella_<SUBJECT_ID>.npz --update-hz 50 <稳定器参数>
```
4. 一句话解释指标与赛题点的映射：
- `false_rate_global` 对应 False Trigger Rate & Confidence Handling
- `switches_per_min`/STOP toggles 对应 Temporal Stability
- `inference_ms` 对应 Inference Speed & Latency

### 9.3 多机器人扩展（Fan-Out）怎么讲才可信

赛题强调“one-to-many supervision”。在工程上，brain decoder 的扩展方式通常是 **decode once, fan-out many**：
- 同一时刻人的脑信号只有一条流，意图解码器只需要跑一次
- 解码输出是离散指令（`Action`），可以通过路由层分发给一个或多个机器人
- 因此推理成本不随机器人数量线性增长，主要瓶颈变成安全与路由策略

最小 fan-out 架构（无需改变解码器）：
- `IntentDecoder`：`EEG -> (p_move, p_dir) -> IntentStabilizer -> Action`
- `ActionRouter`（可先用传统 UI/键盘做选择，不影响赛题核心）：决定把 `Action` 发给哪一台/哪一组机器人
- `RobotAgent[i]`：各自的执行器与本地 autonomy；收到 `Action` 作为高层 override 或任务指令

多机器人安全性要点（解释为什么我们强调 `false_rate`）：
- 如果一个错误 Action 被 fan-out 给很多机器人，影响会被放大，因此需要更严格的误触发控制（本设计给出 `<=0.05` demo 及格线与 `<=0.01` 可扩展量级的 KPI）。
- 推荐把“低误触发”作为硬约束，用稳定器与（可选）两阶段/多阶段 gating（例如 move/rest 之外再加 intervene/arm 状态）减少误动。

### 9.4 时间不够时的交付策略（务实版）

若截止前无法稳定达到 `<=0.05` 且仍保持可用覆盖：
- 用 oracle + 模型并排 demo（oracle 证明闭环链路，模型展示现实噪声与改进方向）
- 用 `eval_closed_loop.py` 报告你的 best-effort 指标，并明确 failure mode（例如某个 eval session no-trigger）与下一步（`feature-mode-move delta`、Stage1 权重、cross-session 轮换）
- 强调推理速度与可扩展的系统结构：解码器与稳定器开销极小，fan-out 本质是消息分发与安全策略问题
