# P2 Revision V2 Design Report

## 结论与边界

- 协议：`P2_REVISION_V2_DESIGN_ANALYSIS_V1`。
- 当前状态保持：`REVISION_DESIGN_NOT_FROZEN`、`OFFLINE_METHOD_REQUIRES_REVISION`。
- 本任务只读取既有 convergence/root-cause 产物；没有执行 P2、formal personalization、人体或机器人实验。
- active reference、ROM、`theta_shank = q_hip - q_knee`、五参数模型、mechanical objective、generator bounds、0.005 equivalence tolerance 与 90% support gate 均未修改。
- 历史 truth 只标注反事实结果，不是 policy、拟合、正式 calibration 或 stopping feature 输入。

## Part A — `LOCAL_DECISION_VALIDATION_PROTOCOL_V1`

### Local decision pair

一个 local decision pair 必须由“当前 formal candidate 与一个候选”组成：两点都在既有 generator admissible parameter space；只允许 hip、knee、phase 中恰好一个坐标变化；有符号变化量必须等于该轮既有 trust-region step（1、1/2 或 1/4 层级）；禁止 clipping。定义完全处于 generator 参数空间，不使用欧氏物理距离，也没有发明任何物理距离阈值。

### Global 与 local 样本不是同一尺度

| pair class | n | mean | P95 | P99 | max | Pearson(pred,actual) | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| current global identification pair | 61 | 0.00850543052 | 0.00753215325 | 0.193060872 | 0.216052983 | 0.963163 | 0.949735 |
| retrospective local decision pair | 341 | 2.22800744e-05 | 2.41244828e-05 | 0.000863147482 | 0.00170839436 | 0.998766 | 0.998445 |

global 的 61 个实例是 identification excitation pair；local 的 341 个实例是历史 decision opportunities。后者更接近待决策尺度，但它是 post-hoc retrospective 样本，不是 designated validation，因此不能直接冻结阈值。

局部样本中 supported=274、unsupported=67；观察到的坐标步长只有 `[('hip', 1.0), ('knee', 1.0), ('phase', 0.01)]`，即当前历史只覆盖 initial trust step，没有 1/2 与 1/4 层级。pooled 候选保守地保留两种 support status，但未来 designated protocol 必须按 support status、坐标和全部既有 trust levels 预声明分层；不能从这 341 个样本推断跨层级不变性。

仅生成三个数值候选：local max `0.00170839435891`、local P95 `2.412448282e-05`、local P99 `0.000863147481512`。三者均为 `threshold_frozen=false`。

### Historical counterfactual（不修改真实 policy）

| guard | exploit candidates | missed improvements | false improvements | conservative-stop rounds |
|---|---:|---:|---:|---:|
| G0 current global replay | 20 | 7 | 0 | 7 |
| G1 local max candidate | 0 | 27 | 0 | 27 |
| G2 local P95 candidate | 24 | 3 | 0 | 3 |
| G2 local P99 candidate | 8 | 19 | 0 | 19 |

为了降低直接复用同一 case 的偏差，G1/G2 反事实对每个 case 使用 leave-one-case-out 候选 bound。P95 相对 G0 少 4 个 missed、false 不变为 0、conservative stop 少 4 轮；但这仍是同一历史研究体系内的回放，不能把“未观察到 false”解释为安全证明。local uncertainty 架构值得进入下一版设计，具体 max/P95/P99 均不应现在冻结；下一步需要预声明且与拟合、adaptation outcome、final held-out test 分离的 designated local validation。

## Part B — `EXPLORATION_VALUE_AWARE_STOPPING_CANDIDATE_V1`

三个价值严格分开：`MODEL_VALUE` 是参数、prediction map 或 validation uncertainty 的可观察变化；`SUPPORT_VALUE` 是新支持点；`DECISION_VALUE` 是新增 exploit eligibility 或在既有 0.005 规则下 best trajectory 改变。support 不是 decision value。

32 次 explore 中：MODEL_VALUE=0，SUPPORT_VALUE=32，DECISION_VALUE=3，纯 support 且 exact-zero decision/model value=29。

- `knee_stiff` 的 8 次 explore 都增加 support，但参数、prediction map、validation uncertainty、exploit eligibility 和 best trajectory 均没有改变，所以 8 次都没有 decision value。
- baseline / hip_stiff / heavy_leg 的 Trial 7–13 共 21 次：仍然只是扩大 support；新增 exploit=0、best change=0、MODEL_VALUE=0。
- 禁用候选 S1 在历史上会少执行 25 次，之后原历史 exploit=0、accepted best change=0。
- 禁用候选 S2 会少执行 21 次，之后原历史 exploit=0。

因此 decision-value-aware exploration 值得作为下一版 research candidate；S1/S2 只是结构候选，`candidate_enabled=false`、`threshold_frozen=false`，本任务不选择 stopping rule。

## Part C — Subject specificity

`subject_specificity_gap.csv` 分开记录 A objective、B generator direction、C search policy、D guard：

- baseline、hip_stiff、heavy_leg 的 P2 alpha 与 truth optimum 不同，但 truth regret 分别为 0.003600、0.000173、0.004584，都在不变的 0.005 equivalence 内。
- knee_stiff regret=0.025226，是唯一超出 0.005 的 meaningful gap；在 P2 selected 点，历史一步最优 truth ΔJ=-0.004467188，单步改善幅度仍小于 0.005。主要成因是 local stepwise acceptance 无法累积多个 sub-threshold knee moves；不是 local uncertainty bound 单独造成。
- 当前 generator 已包含所有四个 observed truth optima，因此现有证据不支持扩大 search direction。全部 optimum 触及 knee=-5 边界只说明边界外未知，需要另行科学审查，不能据此扩界。
- objective 仍产生 subject-dependent complete optima；本任务禁止且不建议修改 objective。

## Part D — Revision recommendation

1. local uncertainty 值得进入下一版本吗？**值得进入架构设计，但不能冻结任何候选阈值。**
2. decision-value-aware exploration 值得进入下一版本吗？**值得，以 disabled research candidate 先实现并复核。**
3. 是否需要修改 objective？**不需要。**
4. 是否需要扩大 generator direction？**当前证据不支持。**
5. 最小修改集合：增加可审计的 local uncertainty provider（正式冻结前 G0 保持默认）；分开记录 MODEL/SUPPORT/DECISION；增加默认关闭的 decision-value stopping candidate；保留 reference、ROM、模型、objective、bounds、0.005 与 90% gate。

是否值得实现 P2 V2：`P2_V2_RESEARCH_PROTOTYPE_WORTH_IMPLEMENTING_AFTER_DESIGN_FREEZE`。实现必须是下一项单独授权的 research-prototype 任务，不能把本报告直接当作 formal policy freeze。
