# ALGORITHM_ARCHITECTURE_REVIEW_V1

审查日期：2026-09-18。任务：`FULL_REHAB_ROBOT_ALGORITHM_ARCHITECTURE_REVIEW_V1`。

本报告以本地当前工作树（包括未跟踪的新模块）为对象，依据源码调用关系、冻结结果和本轮 existing tests；不把 README 或历史报告视为当前实现的唯一依据。本轮只新增本报告，未修改算法、V3、endpoint、控制/安全或依赖文件，未运行机器人、重生成完整 landscape、训练 PINN 或自动提交。

## 1. Executive summary

**目前是多个可运行的离线研究组件，加上独立的机器人采集/执行基础设施；尚不是“证据 gate → 五参数 refit → E2 residual GP → EI → 真机 full trial”的贯通系统。**

```text
CURRENT_MAIN_ALGORITHM =
  RESEARCH: evidence-gated fixed-ROM V3 probing;
            coordination personalization currently inactive
  RUNNABLE_OFFLINE_STAGE2: subject-ROM V3 + five-effective-parameter
            scalar-RMS gray-box + residual Matern-5/2 GP + LCB

CURRENT_PRIMARY_CANDIDATE_COUNT = 625
LEGACY_CANDIDATE_DOMAINS = [
  21025-point 3D alpha proposal/geometry lattice,
  16675-point MyoLeg-V2 admissible subset
]
ROM_ARCHITECTURE = PARTIALLY_CONSISTENT
RESIDUAL_GP_IMPLEMENTATION = CONSISTENT
MODEL_INFORMED_BO_IMPLEMENTATION = PARTIAL
EVIDENCE_GATED_ARCHITECTURE = PARTIAL
CURRENT_BUDGET_SEMANTICS = K_total includes reference; default 4 = 1 + 3
SIMULATED_MECHANICAL_PERSONALIZATION_NECESSITY = NOT_SUPPORTED
READY_FOR_RESUME_DESCRIPTION = true
READY_FOR_REAL_MEASUREMENT_STAGE = false
```

最后两个状态需精确定义：可以描述已实现的**离线方法、轨迹族和测量分析软件**，不能描述为完成真实个性化验证；真实测量的**离线日志分析入口可用**，但上述 `false` 指尚不能声称真实 V3 probing、合格测量证据和 downstream personalization 已准备完毕或已贯通。它不否定下一研究阶段应是测量验证，也不是本轮对机器人现场状态的评估。

最重要的发现：

1. 当前二维 V3 BO 使用 **LCB**。EI 确实存在，但在历史三维 alpha / 21,025 点路径，不能移花接木地描述成当前主算法。
2. E2 的定义、同腿参考归一化和 necessity 分析已实现；当前 BO 的 full-dynamics adapter 仍预测旧的合力矩 RMS（E0 类），并没有 E2 adapter。
3. `SubjectROMProfile`、subject-specific reference/domain、ROM 改变后重启均已实现；旧统一绝对参考路径仍可绕过 ROM wrapper。
4. 当前 BO 的五参数 refit 使用每 trial **一个 endpoint 标量**。完整时序力/力矩辨识是另一条已实现路径，不能写成 BO 自动调用它。K=4 时只有至多 4 个标量数据约束，五参数解依赖先验正则，不能据此声称五参数可独立辨识。
5. 结果对象同时返回 measured-best 和 model-recommended；基准的 `final_regret` 默认评价后者，不符合所提保守最终选择规则。
6. 测量分析入口强制三级状态顺序且不自动输出 personalization supported，但未连接 BO 启动 gate；判据完整性和 task-direction correlation eligibility 还有实现缺口。
7. 三组定向 existing tests **186 passed**。这是软件/离线验证，不是全仓测试结果或物理验证。

## 2. Current actual architecture：模块、数据结构与调用图

### 2.1 Repository module map

| 模块 | 当前实际责任 | 与主线关系 |
|---|---|---|
| `personalization/rom_gated_v2/` | Stage 0 状态机、冻结 ROM、subject reference、625 domain、episode wrapper | 当前 ROM/V3 离线架构主体 |
| `external_simulation/myoleg_v3_trajectory_parameterization_design_v1/parameterization.py` | 无目标函数依赖的二维 V3 生成器 | 当前共用 trajectory operator |
| `personalization/{observations,ledger,environment,sequential}.py` | observation contract、试验账本、离线环境边界、固定预算循环 | 当前 BO/greedy 运行主体 |
| `personalization/models/`、`selectors/` | gray-box adapter、residual/outcome GP、LCB/greedy/random/reference | 当前离线方法组件 |
| `personalization/benchmarks/` | 合成先验质量、ROM smoke、鲁棒性比较 | 离线方法验证，不是人群/临床证据 |
| `lower_limb_sim/{kinematics,jacobian,full_dynamics,force_mapping,parameter_estimator}.py` | 二连杆运动学/动力学、力映射、时序五参数辨识 | 复用基础科学组件 |
| `lower_limb_sim/five_leg_mujoco_v1/` | 五个冻结机械腿、精确状态逆动力学 replay、E0/E2 landscape、necessity | 当前仿真表征主线；不执行 BO |
| `external_simulation/myoleg_*` | MyoLeg 导入、ROM 兼容、V2/V3 replay、truth boundary、历史 cohort/necessity | 保留的外部仿真证据与工具 |
| `measurement_validation/analysis.py` | normalized recorded data → validity/repeatability/sensitivity | 当前下一阶段分析组件 |
| `hardware/`、`collection/`、`control/`、`safety/`、`scripts/` | SDK 适配、采集/日志、参考轨迹执行、preflight、离线识别与分析 CLI | 独立实机基础设施；不是 BO runtime |
| `config/`、`reference_release/` | 机器人协议/已冻结参考与配置 | 不能与 subject-specific V3 domain 混为一谈 |
| `lower_limb_sim/*p2*`、`*sequential*`、`*finite*`、`*equal_budget*` | 旧 alpha 搜索、有限 shortlist、EI 等多个研究阶段 | 历史，不是当前二维 V3 主线 |
| 各 `formal_artifacts/`、`external_simulation_audits/`、根目录研究报告 | 报告/冻结输出 | 证据及历史描述，不构成运行时连接 |

扫描覆盖这些源码树、根目录架构文档、主要 runner 和相关测试；重点逐函数追踪下列主路径。未逐行审计所有第三方 SDK 示例，未把每个历史 artifact 都重新生成。

### 2.2 实际主数据结构

| 数据结构 | 定义 | 关键语义 |
|---|---|---|
| `ROMBoundaryObservation` / `ROMCalibrationLedger` | `personalization/rom_gated_v2/rom.py` | Stage 0 配置、门状态与独立校准记录 |
| `SubjectROMProfile` | 同上 `:60` | frozen dataclass、ROM bounds、profile/version、来源、fingerprint |
| `SubjectSpecificReference` / `SubjectSpecificCandidate` | `personalization/rom_gated_v2/reference.py:100,227` | subject q/dq/ddq、固定时间与 candidate identity |
| `Candidate` / `V3CandidateDomain` | `personalization/candidates.py:35,56` | beta、ID、index；基类没有 subject ROM 契约 |
| `EpisodeObservation` | `personalization/observations.py` | endpoint scalar、unit、uncertainty、valid/reason、metadata；不包含必需的原始时序 |
| `LedgerEntry` / `ExecutedCandidateLedger` | `personalization/ledger.py` | 已执行点、观测、模型摘要、下一点与 acquisition |
| `Prediction` / `Selection` | `personalization/models/base.py` / `selectors/base.py` | mean/std/valid；candidate/acquisition |
| `SequentialRunResult` | `personalization/sequential.py:36` | 两种最终候选分开保存 |
| `TrajectoryReplayResult` | `lower_limb_sim/five_leg_mujoco_v1/replay.py` | 模拟关节力矩时序、RMS、valid/reason |
| normalized V1 input/result mapping | `measurement_validation/analysis.py` | static records、episodes、外部 criteria、三级结果；不是 `EpisodeObservation` |

### 2.3 当前可运行的 subject-ROM 离线调用链

```text
OfflineROMDevelopmentCase
  [personalization/rom_gated_v2/development.py: make_offline_rom_development_cases]
→ ROMBoundaryObservation → ROMDeterminationController.observe / freeze_profile
  [personalization/rom_gated_v2/rom.py: SyntheticThresholdGate; controller :402]
→ SubjectROMProfile(frozen=True)
  [rom.py:60]
→ SubjectSpecificReferenceAdapter.adapt
  [personalization/rom_gated_v2/reference.py:157]
→ SubjectSpecificV3CandidateDomain.from_frozen_beta_grid
  [reference.py:354; canonical source: personalization/candidates.py]
→ generate_v3_trajectory(beta_flex, beta_extend) → validated q/dq/ddq
  [external_simulation/myoleg_v3_trajectory_parameterization_design_v1/parameterization.py]
→ ROMGatedPersonalizationEpisode.run
  [personalization/rom_gated_v2/episode.py:41]
→ run_sequential_personalization: reference first
  [personalization/sequential.py:115]
→ environment.evaluate(candidate, trial_index)
  [development.py: SubjectSpecificOfflineMechanicalEnvironment.evaluate]
→ EpisodeObservation(scalar RMS + declared synthetic residual)
  [personalization/observations.py]
→ PhysicsInformedResidualModel.fit(history)
  [personalization/models/residual_gp.py:102]
→ PhysicsSubjectModel.fit → FullDynamicsGrayBoxEndpointAdapter.fit
  [personalization/models/physics_graybox.py:294,46]
→ valid scalar targets + candidate trajectories → bounded regularized least_squares
→ candidate_subject_from_parameters → inverse_dynamics → scalar RMS prediction
  [lower_limb_sim/parameter_estimator.py:128; full_dynamics.py:218;
   personalization/rom_gated_v2/physics.py:18]
→ recompute each past residual using current theta → ResidualGaussianProcess.fit_arrays
  [personalization/models/residual_gp.py]
→ physics_mean + residual_mean, residual_std
→ LowerConfidenceBoundSelector.select_next → unexecuted candidate
  [personalization/selectors/bo.py:9; selectors/base.py]
→ next evaluate → append/refit → final measured-best AND model-recommended
  [personalization/sequential.py:95,105,115; ledger.py]
```

这里没有经过 measurement gate，也没有调用 EI 或 E2。`run_rom_gated_v2_smoke.run` 是这条链的现成端到端离线入口。

### 2.4 另外两条独立数据流

```text
Simulation characterization:
FrozenBenchmarkDefinition + MechanicalLegDefinition
  [five_leg_mujoco_v1/model.py: load_frozen_benchmark_definition]
→ leg.make_rom_profile() → build_leg_domain → subject V3 trajectories
  [five_leg_mujoco_v1/benchmark.py:27]
→ make_mujoco_model → replay_trajectory → tau_hip/tau_knee
  [model.py; replay.py:73]
→ generate_leg_landscape / compare_gray_box(reference only)
  [benchmark.py:35,160]
→ discriminability_audit.run_discriminability_audit
→ endpoint_design.build_feature_representation / _endpoint_values
→ e2_necessity.run_e2_necessity_study
→ NOT_SUPPORTED; no algorithm activation

Real recorded measurement:
RokaeRobotAdapter → RealRobotAcquisition → EpisodeLogger
  [hardware/rokae_adapter.py; collection/real_robot_acquisition.py:56;
   collection/episode_logger.py:178]
→ raw state/wrench/aligned CSV + metadata
→ optional StaticValidationLabelLogger PRE/LOAD/POST sidecar
  [scripts/static_validation_logging.py:339]
→ externally prepared normalized V1 JSON / static_csv_paths / episode_json_paths
  [measurement_validation/analysis.py: load_analysis_input :1192]
→ run_validation_analysis
  [analysis.py:1103; scripts/run_real_measurement_validation_analysis.py:main]
→ analyze_static_load → extract_episode_features
→ analyze_same_trajectory_repeatability → analyze_trajectory_sensitivity
→ FUTURE_SUBJECT_TRAJECTORY_EVALUATION_READY (not personalization approval)
→ [cross-subject decision-relevant interaction and BO activation: not implemented]
```

另有 `scripts/identify_real_episode.py:identify_real_episode`：审核过的 episode/几何/映射 → IK 重建 q → 导数与时序对齐 → `estimate_subject_parameters`。它不调用 current BO；测量分析也不自动调用它。

## 3. Expected vs implemented mapping

| Expected | 实际核对 | 判断 |
|---|---|---|
| Stage 0 subject-specific ROM → freeze | 专用 V2 wrapper、合成状态机可运行；real interface 关闭 | 已实现离线契约，真实 ROM determination 未完成 |
| V3 fixed-ROM 625 | 当前 canonical 与每腿 domain 均 625 | 一致 |
| validity → repeatability → sensitivity | analysis orchestration 强制状态依赖 | 主入口一致；低层 API/criteria 有限制 |
| decision-relevant subject×trajectory gate | 仿真有 necessity study；真实框架仅给 future-evaluation ready | 尚无通用执行路由 |
| NO → common/reference | 冻结研究结论；`ReferenceSelector` 存在 | 没有从 gate 到 reference 的统一 orchestrator |
| YES → gray-box + residual GP + EI | gray-box + residual GP + LCB 可运行 | PARTIAL |
| E2 simulated primary | 独立 E2 后处理/necessity 链 | 正确，但未接入 BO adapter |
| 每 valid trial refit | 循环每 trial 调 fit，内部过滤 invalid | 因果性正确；invalid 后也会重拟合已有数据 |
| best executed valid 为 primary final | 有该字段；benchmark primary final 使用 predicted recommendation | 不统一 |
| reference + K_add 0–3 | current API 是含 reference 的 K_total；旧 runner 用额外 validation budget | 不统一 |

命名上的 Stage 1 也不一致：`ROMGatedPersonalizationEpisode` 将全部 fixed-ROM optimization 称 Stage 1；本任务把 V3 probing 称 Stage 1、BO 称 Stage 2。这是术语迁移问题，不应据此重写 trajectory。

## 4. ROM architecture

**`ROM_ARCHITECTURE = PARTIALLY_CONSISTENT`；专用 subject-ROM 正常调用路径一致，仓库所有入口并不一致。**

- `SubjectROMProfile` 为 frozen dataclass，嵌套 metadata/evidence 被 `_freeze` 处理；bounds 有有限值和大小关系检查。它表示当前任务允许域，不是完整生理 ROM。
- `ROMDeterminationController.observe` 在 invalid/external stop/nonmonotonic expansion 等情况下停止；`freeze_profile` 基于 first/last SAFE 配置。合成阈值门明确不是人类安全模型；`UnavailableValidatedSafetyGate` 与 `RealROMDeterminationInterface` 保持关闭。
- `SubjectSpecificReferenceAdapter.adapt` 将既有非对称参考关节范围适配到该 profile，保持 progression/timing；`SubjectSpecificV3CandidateDomain.from_frozen_beta_grid` 在同一 profile 内仅改变 beta。
- `_subject_candidate_id` 包含 profile ID、version、fingerprint、beta、canonical ID、V3 operator version 和 subject reference version。相同 beta、不同 ROM 不同身份。
- `ROMGatedPersonalizationEpisode` 要求 frozen profile 且 domain fingerprint 匹配；`restart_for_changed_rom` 关闭原 episode，要求新 frozen profile 和增加版本；正常完成后也不允许再次 `run`。
- `assert_same_frozen_rom_comparison` 检查方法结果 fingerprint 一致；五腿 `MechanicalLegDefinition.make_rom_profile` 每腿生成自身 profile。
- BO input 只有 beta；ROM 不是优化变量。改变 ROM 需新 episode，不能在原候选历史里继续。

限制：`run_sequential_personalization` 接受普通 `V3CandidateDomain`；旧 `FullDynamicsGrayBoxEndpointAdapter` 从统一 `NATIVE_ROM_REFERENCE_CANDIDATE.csv` 生成轨迹；`run_equal_budget.py` 不经过 subject ROM。它们须明确归类为 legacy/development 路径。真实 robot protocol 固定的 ROM 是另一套执行契约，并非证明所有患者共享 ROM。

此外 wrapper 不校验传入 environment/physics adapter 的 ROM fingerprint、endpoint、episode identity；`domain` 属性也不是只读接口。现有正确 runner 会配对它们，但不可声称任意调用方式都有端到端防混用保证。profile 自身不可变与整个运行上下文不可替换是两回事。

## 5. V3 trajectory family

```text
beta = [beta_flex, beta_extend]
each axis: -0.03 .. +0.03, step 0.0025, 25 values
CURRENT_PRIMARY_CANDIDATE_COUNT = 25 * 25 = 625
canonical reference = MYOLEG_V3_K0312, beta=(0,0)
```

来源：`personalization/candidates.py:V3CandidateDomain.from_frozen_artifact` 校验 frozen table、顺序、included/kinematic flags；`reference.py:from_frozen_beta_grid` 另要求 625 点。

实际 operator 在 `external_simulation/myoleg_v3_trajectory_parameterization_design_v1/parameterization.py`：

```text
b(s) = 64*s^3*(1-s)^3
w(s; beta_branch) = s + beta_branch*b(s)
q_hip = subject reference q_hip
q_knee = knee reference spline at warped branch phase
```

`np.where(phases == "flexion", beta_flex, beta_extend)` 将两参数分别作用于屈曲、伸展内部。正 beta 提前沿原膝分支推进，负 beta 延后；不是独立改变髋 ROM、膝 ROM 或周期。

`generate_v3_trajectory` 用 chain rule 计算 dq/ddq；零参数直接复制 reference arrays。分支端点 b、b′、b″=0；复制 anchor samples 消除插值舍入误差。`_validate_candidate` 核验 start/end/分支 anchor、extrema、closure、finite arrays、无分支折返、正 warp derivative；失败拒绝，不 clip。duration 来自同一 reference time vector。subject extrema 容差 `2e-5 rad`，不是对任意参数无条件精确的承诺。

文档 family ID `P4_BRANCH_AWARE_COORDINATION_FUNCTION_V3` 与代码 operator ID `V3_PARAMETERIZATION_SEMANTICS_V1` 是不同层级名称，candidate identity 使用后者；应解释映射，不能误认成不同算法版本。当前实现与 branch-aware V3 描述一致。

历史 domain：`lower_limb_sim` 的 alpha=(hip_delta,knee_delta,phase_delta) 29×29×25=21,025；`external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py` 对这些 proposal 筛出 16,675 个 admissible MyoLeg-V2 点（当前 manifest 核实）。它们会改变任务范围/协调定义，不能并入二维 fixed-ROM 625。

## 6. Mechanical endpoint

**E2 实现正确；其角色是 `PRIMARY_SIMULATED_MECHANICAL_ENDPOINT / MODEL_DERIVED / NOT_REAL_MEASUREMENT_VALIDATED`。**

`lower_limb_sim/five_leg_mujoco_v1/endpoint_design.py`：

- `_response_metrics_from_torque` / source discriminability metrics：分别按 joint 和 branch 计算 time-weighted RMS、peak。
- `build_feature_representation:194` 先按 `leg_id` 分组，在每组找 beta=(0,0)。
- `_relative_feature_mapping:122` 各 component 除以**同腿、同分量** reference。
- `_endpoint_values:166`：

```text
E2 = max(
  RMS_hip_flex / RMS_reference_same_leg_hip_flex,
  RMS_hip_extend / RMS_reference_same_leg_hip_extend,
  RMS_knee_flex / RMS_reference_same_leg_knee_flex,
  RMS_knee_extend / RMS_reference_same_leg_knee_extend
)
```

当前冻结五腿中未跨腿混用。该函数是冻结数据集分析器，分组只用 leg_id；没有为未来“同 leg 多 ROM/version 混表”设计通用防护，也没有零分母的一般化策略，不能直接视为 real endpoint service。

| Endpoint | 当前科学角色 | 代码状况 |
|---|---|---|
| E0：full torque-vector RMS | Historical baseline | `replay.py`、原 benchmark JSON、gray-box adapter 仍使用；历史主入口名未同步迁移 |
| E1：max(two joint full-cycle normalized RMS) | Diagnostic | endpoint study 保留 |
| E2：max(four branch/joint normalized RMS) | Primary simulated | 独立 endpoint/necessity 链 |
| E3/E4：normalized peak maxima | Secondary diagnostics | 未被最终冻结文档选为 primary |
| real Fx/Fy/Fz、分支/局部特征 | Measurement validation features | 没有直接冒充 E2；尚无 validated real scalar J |

关键断点：`SubjectSpecificFullDynamicsGrayBoxEndpointAdapter._predict_with_theta` 返回 time RMS of `[tau_hip,tau_knee]`，不是 E2。`FrozenOfflineReplayEnvironment` 虽能接收任意 scalar table，也不能使 gray-box 自动获得同单位/同定义的 E2 mean。当前适配器不强制检查 observation endpoint name/unit，误接 E2 数值会仍然执行拟合。

本轮查阅的 current mainline/endpoint/measurement 文档均保留 model-derived 和 real-validation 边界，未发现它们把 E2 宣称为真实机器人已验证指标。旧 benchmark 的 “primary endpoint unchanged” 是其 E0 阶段历史语义，应增加历史作用域说明。

## 7. Measurement gate

### 7.1 已实现部分

`measurement_validation/analysis.py:run_validation_analysis` 顺序传入 Level 1、2 的 `criterion_status`，上游不支持则下游公开结论为 `INSUFFICIENT`；描述统计仍可计算。非 `REAL_MEASUREMENT` 输入的三个公开状态统一为 `INSUFFICIENT`。

缺测/invalid 保留 `None`/blank 与原因和计数，不填 0。static PRE/LOAD/POST raw force/torque 被保留；trajectory 原始日志不改写，features 保留 Fx/Fy/Fz 分量，不能把 feature export 误当作完整原始时序副本。

`_task_direction` 在未给方向时不假定 X/Y/Z；显式但 geometry 未验证的投影可作描述，其 `decision_eligible=false`。`_condition_key` 用 entity、beta、ROM ID/version、setup、source 分组；`_context_key` 去掉 beta，分离同轨迹重复波动与异轨迹效应。full-cycle / flexion / extension RMS、peak、phase-local curves 均存在；branch 依赖显式标签，不自行推断。

输出只有 `FUTURE_SUBJECT_TRAJECTORY_EVALUATION_READY`，且 `personalization_conclusion_emitted=false`。没有自动 `PERSONALIZATION_SUPPORTED`，没有运行 BO。cross-subject decision relevance 仍是未来阶段。

本轮运行现有 CLI 的 synthetic demo，内部三级计算均 `SUPPORTED`，对外为：

```text
MEASUREMENT_VALIDITY = INSUFFICIENT
SAME_TRAJECTORY_REPEATABILITY = INSUFFICIENT
TRAJECTORY_SENSITIVITY = INSUFFICIENT
FUTURE_SUBJECT_TRAJECTORY_EVALUATION_READY = false
```

原因是输入不是合格真实 evidence，不能归因为计算失败。仓库有历史硬件诊断日志，不能笼统说“完全没有任何真实日志”；本次未找到这套 normalized V1 的真实三级验证输入/完成结果。真实 geometry、predeclared thresholds 和配对重复试验仍待建立。

### 7.2 不能忽略的边界问题

1. **Gate 未接到 BO**：`personalization/` 无 measurement analyzer 或 future gate 引用。BO 可作为离线方法直接运行；科学上的 evidence-gated policy 并未实现成跨模块强制执行。
2. **Criteria 完整性未强制**：`_criterion_status` 只检查调用者提供的 recognized checks。将 demo 复制、标记 `REAL_MEASUREMENT`，仅提供 L1 `minimum_valid_fraction=0.1`、L2 `minimum_repeats_per_group=2`、L3 `minimum_beta_conditions=2`，本轮实际得到三个 `SUPPORTED` 和 future gate `true`。这证明省略物理有效性/重复性幅度/SNR 判据不会自动阻止 readiness；它不是证明真实数据已通过。evidence 分类也是调用者声明，不是日志真实性认证。
3. **未验证 task direction 的 correlation eligibility 漏检**：L2 的 `feature_cvs` 调用 `_decision_feature_is_eligible`，但 `correlations` 列表没有同样检查。用 demo + direction=[1,0,0]、`geometry_validated=false`，L2 仅给 `minimum_time_series_correlation=0` 与 `decision_feature_ids=["task_direction_projection.full_cycle_rms"]`，实际返回 L2 `SUPPORTED`（correlation≈0.98918）。所以“所有未验证 projection 均无法进入任何判据”目前不成立。
4. 低层 `analyze_same_trajectory_repeatability` / `analyze_trajectory_sensitivity` 的 upstream 参数默认 `SUPPORTED`；严格顺序是 orchestration 的性质，不是任意直接调用都强制。
5. measured episode RMS 目前是有效样本的均方根；sim E2 是 time-weighted RMS。非均匀采样、较大缺口、实际周期完整性/同步资格不能由“至少两个有效样本”保证；当前 `analysis_valid` 是解析可分析性，不是 full-cycle physical validity。

这些问题不改变当前 demo 的 `INSUFFICIENT` 结论，但应在将来把 gate 当作真实实验准入规则前做最小修复。本轮保留源码，仅记录。

## 8. Gray-box model 与辨识数据流

### 8.1 正式五参数

`lower_limb_sim/config.py:82` 和 `candidate_subject_from_parameters:128`：

```text
theta = [mass_scale,
         k_hip_nm_per_rad, k_knee_nm_per_rad,
         b_hip_nm_s_per_rad, b_knee_nm_s_per_rad]
```

mass_scale 同时缩放模板的大腿/小腿质量与转动惯量；k、b 为等效关节刚度和阻尼，q0/COM 等来自模板。不是生理组织参数，不是五个完整人体可辨识属性。

### 8.2 当前 BO 使用的是 scalar fitting 路径

```text
EpisodeObservation(valid, endpoint_value, endpoint_uncertainty, beta, candidate_id)
→ valid_observations(history)
→ target = endpoint_value                 # 无 raw force 特征提取
→ uncertainty = max(reported or 0, 0.05)
→ candidate trajectory q/dq/ddq from frozen subject domain
→ candidate_subject_from_parameters(template, theta)
→ inverse_dynamics → sqrt(time integral(tau_hip²+tau_knee²)/duration)
→ minimize concat[(prediction-target)/uncertainty,
                  sqrt(lambda)*(theta-prior)/parameter_scales]
→ theta_hat → predict every candidate on demand → candidate landscape
```

来源：`personalization/models/physics_graybox.py:FullDynamicsGrayBoxEndpointAdapter.fit`；subject trajectory lookup 由 `rom_gated_v2/physics.py` override。拟合采用有界 `scipy.optimize.least_squares`，linear loss，默认 lambda=1，`max_nfev=40`，以上次 theta warm start，regularization prior 始终为初始配置。没有调用 `estimate_subject_parameters`，也没有从 EpisodeObservation 重建完整测量时序。

每 valid new trial 后拟合包含它与全部过去 valid observations；invalid 不进 targets，却消耗预算，循环仍可能重新拟合旧数据。模型预测与 observed endpoint 分别保存在 Prediction metadata / EpisodeObservation 中。625-landscape 在 selector 遍历时逐点计算，当前循环不是每轮必须持久化一张 625 行表。

限制：未把 optimizer success、残差秩/条件数或 parameter covariance 纳入 adapter validity；返回 result.x 即更新。K_total≤4 的 endpoint scalar 不足以从数据独立辨识五参数，正则解不等于 identification evidence。`PhysicsSubjectModel.predict` 的 std=0，后续不传播 theta uncertainty。

### 8.3 完整时序辨识是独立路径

`lower_limb_sim/parameter_estimator.py:estimate_subject_parameters` 输入 q/dq/ddq、observed Fx/Fz、sample_valid 等 DataFrame；拒绝 `true_* / ground_truth* / tau_total*` 列，过滤 invalid/stale。通过 `measured_joint_torque` 做 JᵀF，再拟合两个 joint torque 的完整时序，默认 soft_l1、max_nfev=500，输出 optimizer status、Jacobian singular values、covariance 等。

`scripts/identify_real_episode.py:330` 调用这一路径，并先做 reviewed mapping、IK、离线导数/对齐、时间切分 train/test。它与 current BO 只是共享五参数语义和 inverse dynamics，不共享完整 fitting algorithm。

### 8.4 Truth leakage 核对

- 正常 sequential selectors 只接收 history/domain/model，不接环境或全 truth table；`evaluate` 返回当前点，选择下一点发生在下一 observation 产生前。
- `FrozenOfflineReplayEnvironment` 仅 reveal 请求行；truth/oracle metrics 在 run 完成后计算；现有因果/无 oracle 测试通过。
- 五腿 `compare_gray_box` 只把 reference 一条 observation 传给 adapter；全 landscape 用于事后相关性与 regret，不是拟合输入。
- **合成先验例外须披露**：`run_equal_budget._physics` 把 `case.optimum_beta` 和 `case.landscape` 传给 `AnalyticDevelopmentPhysicsAdapter`。P0–P3 是以已知真函数设计的先验质量实验，其 `fit` 仅计数，不辨识 theta。虽然无运行中 future-data 查询，仍有 intentional truth-informed prior construction；不能宣传为完全不依赖真值构造的 subject identification benchmark。

## 9. Residual GP

**`RESIDUAL_GP_IMPLEMENTATION = CONSISTENT`，限于当前 scalar endpoint / fixed GP 设计。**

| 核对项 | 源码实际行为 |
|---|---|
| Input dimension | 2：beta_flex、beta_extend |
| Normalization | 各除 0.03，冻结域映射到 [-1,1]² |
| Kernel | isotropic Matérn 5/2，length_scale=0.7，signal_std=0.6 |
| Mean / target | residual prior mean=0；`r_i=observed_i-current_physics_prediction_i` |
| Refit | 先 refit physics，再重算所有过去 valid residual；未混用旧 theta 的 residual |
| Noise | reported uncertainty 的平方进 covariance，最低 std=1e-6，加 jitter=1e-9；unknown uncertainty 被调用处当 0 |
| Posterior | Cholesky solve；mean=kᵀK⁻¹r；latent variance=signal²−vᵀv，截断到非负 |
| Total prediction | physics_mean+residual_mean；std=residual_std |
| Duplicate | GP 不自行 deduplicate；selector/ledger 对当前 canonical/domain ID 去重 |
| Invalid | fit 仅用 valid observations；invalid candidate 仍视为已执行 |
| Acquisition uncertainty | LCB 的 `mean−1.5*std` 真正使用 posterior uncertainty |

`StandardGaussianProcess` 作为 Pure BO：对 J 减训练样本均值后拟合同一个 GP，预测再加均值；它没有访问 physics。

限制：未知 measurement noise 不是已证明的零噪声；signal/length-scale 固定，不同 endpoint 单位改变会影响尺度；只建模 residual uncertainty，未传播 theta 和真实几何/同步 uncertainty。普通自定义 domain 只检查 ID 唯一，不检查 beta 唯一，所以“不重复 beta”对冻结 625 domain 成立，不是任意手工 domain 的全局保证。

## 10. Bayesian Optimization：三条实际循环

`personalization/sequential.py:_components` 的对应关系：

```text
Pure BO = "Standard BO"
reference → observe J → StandardGaussianProcess.fit(past valid J)
→ predict mean/std → LCB(unexecuted domain) → next evaluate → repeat

Model-Informed BO = "Physics-Informed BO"
reference → observe J → physics.fit(past valid scalar observations)
→ physics prediction + refitted residual GP → mean/std
→ LCB(unexecuted domain) → next evaluate → refit both → repeat

Adaptive Greedy = "Model-Only Greedy" when using refitting full-dynamics adapter
reference → observe J → physics.fit(past valid scalar observations)
→ argmin model mean(unexecuted domain) → next evaluate → refit → repeat
```

Greedy 用 analytic adapter 时 physics 形状固定、fit 仅计数，不能把该 runner 的结果描述成真正五参数自适应 greedy。

选择只在传入 domain 的 unexecuted IDs 上发生，invalid 已执行点也不能偷偷 retry。canonical domain 已过 kinematic gate；但 `LowerConfidenceBoundSelector` 与 `ModelOnlyGreedySelector` 都**没有**过滤 `Prediction.valid` 或独立 subject safety mask。`SubjectROMProfile.configuration_validity_mask` 是预留描述字段，未变成 acquisition mask。不能把 kinematic allowed 等同 physical safe。

当前 normal flow 没有 future data；refit 在收到当前 observation 后、挑选下一点前执行。账本条目保存的是当前 trial 后用于下一点的 acquisition，解读时不能当成当前点的 acquisition。

**EI 的真实位置**：`lower_limb_sim/equal_budget_model_informed_bo_baseline.py:expected_improvement/acquisition_table/run_bo_sequence`。该路径是三维 alpha normalized input，sklearn ConstantKernel×ARD Matérn + WhiteKernel，固定的 J_pred landscape + residual GP + EI；第一查询是冻结 C1 而非 reference，循环不 refit 五参数 landscape。不能用“仓库里有 EI”证明当前 V3 方法符合预期。

## 11. Trial budget semantics

建议语义 `reference + K_add` 可以用于报告换算，但本轮没有修改任何 API。

| 路径/runner | 现行预算 | Reference / calibration | 与建议的关系 |
|---|---|---|---|
| `personalization/sequential.py:run_sequential_personalization` | `budget>=1`，默认4，range(1,K+1) | reference 为第1次并计入 | K_total=1+K_add；K_add=0/1/2/3 对应传1/2/3/4 |
| `benchmarks/run_equal_budget.py` | PRIMARY_BUDGET=4，sensitivity=(3,5) | 每方法各自 reference 第1次 | 主实验1+3；敏感性1+2、1+4，不仅0–3 |
| `rom_gated_v2/episode.py`、`run_rom_gated_v2_smoke.py` | adaptation_budget / PRIMARY_K=4 | Stage 0 ledger 不计入；V3 reference 计入 | “adaptation budget”不是 additional budget |
| `adaptive_trust_v1/sequential.py` + 对应 runner | 强制 K=4 | reference 第1次 | 1+3 |
| `predictive_failover_v1/sequential.py` + 对应 runner | 强制 K=4 | reference 第1次 | 1+3 |
| `active_diagnostic_v1/sequential.py` + 对应 runner | 强制 K=4 | reference + diagnostic + 2 trials | diagnostic 同样占 full-trial budget |
| `repeated_active_diagnostic_k5_v1/`、`run_k4_vs_k5_information_budget_study_v1.py` | K4/K5 对比，repeated 强制5 | K5=reference+2 diagnostic+2后续 | 1+4；不是 equal K4 主比较 |
| `lower_limb_sim/run_final_model_screened_finite_sequential_validation.py` | validation budget 0/1/2/3 | 已有初始 fitting state；另 reference baseline | 这里 K 是追加验证次数，不是当前 K_total |
| `lower_limb_sim/equal_budget_model_informed_bo_baseline.py` / runner | (1,2,3,5) 次查询 | reference fallback=1，初始模型在 loop 外；第1 query=C1 | K 是 reference 之外 validation queries |
| `lower_limb_sim/sequential_personalization.py` 等 P2 历史链 | 如 MAX_EXECUTED_TRIALS=6，阶段各自 stop/budget | 依赖已有识别数据和阶段契约 | 不可仅替换 K 标签并与 V3 主线合并 |
| 五腿 E2/necessity runner | 每腿625 landscape，5腿 | 同腿reference用于归一化 | 不是低预算 personalization trial 实验 |

特别注意：旧有限验证路径的成本不一定只是“一个 reference calibration”；`_evaluate_case` 从 `prepared.state.fitting_data` 和 parameters 开始，初始化信息量在 loop 外。公平比较必须单列该前置数据成本。

Reference baseline 在当前统一循环中会执行 K 次相同 reference（`ReferenceSelector.allows_reference_repeat=True`），不是一次 reference + 不执行其余预算。invalid trial 消耗 K 且不补试。需要区分 `K_total`、`K_additional`、`N_valid`、`N_unique`、Stage 0 observations，不能一概写“4条不同轨迹”。

## 12. Baselines 与保留算法角色

| 期望角色 | 当前实现 | 审查分类 |
|---|---|---|
| Primary proposed：Model-Informed BO | `Physics-Informed BO`：gray-box + residual GP + LCB | CURRENT offline method；EI 版本未接上；真实激活条件未满足 |
| Reference | `selectors/reference.py` | PRIMARY BASELINE；当前是显式 repeat-reference |
| Random-K | `selectors/random.py` | PRIMARY BASELINE；reference 后 K−1 个不重复随机点 |
| Adaptive Greedy | `selectors/greedy.py:ModelOnlyGreedySelector` | PRIMARY BASELINE interface；需 full-dynamics refit adapter 才符合名称 |
| Pure BO | `Standard BO` + `StandardGaussianProcess` | PRIMARY BASELINE；当前同样 LCB |
| Frozen Top-3 | `final_model_screened_finite_sequential_validation.py:FrozenShortlist` 等 | LEGACY_BUT_VALID；是旧 alpha 的 shortlist/有限验证，不是当前 V3 方法名单 |
| Space Filling | `selectors/space_filling.py` | OPTIONAL development comparator；现成 ROM smoke 使用它 |
| Adaptive Trust V1 | `adaptive_trust_v1/model.py:AdaptivePhysicsMixtureModel` | SECONDARY ROBUSTNESS / EXPERIMENTAL；standard/physics 两专家 mixture moments，不是单个精确 GP posterior |
| Predictive Failover V1 | `predictive_failover_v1/sequential.py:run_predictive_evidence_failover` | SECONDARY ROBUSTNESS；预测证据驱动专家切换 |
| Active Diagnostic Arbitration V1 | `active_diagnostic_v1/sequential.py:run_active_prior_diagnostic_arbitration` | SECONDARY ROBUSTNESS；reference 后主动 diagnostic，再选专家 |
| K5 repeated diagnostic | `repeated_active_diagnostic_k5_v1/sequential.py:run_repeated_active_diagnostic_arbitration_k5` | SECONDARY ROBUSTNESS；两次 diagnostic、五次总 trial；“repeated”指重复诊断机会，不代表必须重复同 beta |
| PINN | 旧清理报告/负向 scope 描述 | CURRENTLY NOT JUSTIFIED；当前主运行链不调用、不训练 |

最终 mainline 文档把这些方法保留为离线方法实现，符合停止继续调 heuristics 的科学结论；某些版本文档/runner 仍称自己的方法 `PRIMARY`，只能解释为该历史子实验内部 primary，不能覆盖最终主线。保留源码，不删除 robustness algorithms。

## 13. Final selection rule

`personalization/sequential.py:_best_observed` 正确取 valid history 的最小 observed endpoint，tie 取较早 trial；无 valid observations 返回 None。

`_model_recommendation` 对**整个 domain**求模型 mean 最小点：允许未执行点、曾 invalid 的点；不检查 prediction.valid。无 model 时返回 measured-best。

因此结果契约区分了两者，但**没有单一保守 beta_final 的公共字段或统一政策**。尤其 `personalization/benchmarks/metrics.py:evaluate_run`：

```text
final = result.model_recommended_final_candidate
final_regret = oracle(final) - oracle_optimum
best_seen_regret = min(oracle(executed candidates)) - oracle_optimum
```

`best_seen_regret` 也不是“按 noisy measured-best 选点后的 true regret”：它对所有 executed 点直接取 true minimum，并且包含 observation invalid 的 executed 点。故论文若写 final measured-best，需要单独计算其结果，不能直接复用上述两个字段。

ROM smoke 的 `best_observed_endpoint` 则确实取 valid executed min。历史 EI runner 的 `_selected`/post-run结果保留 reference fallback，并按已 queried truth 作 primary，posterior recommendation 标为 secondary；两个实验体系的 `final` 语义不相同。

## 14. Fair comparison integrity

以下是当前 unified sequential API 可提供的同预算比较，不是宣称已有 E2/EI 正式比较结果：

| 项目 | Adaptive Greedy | Pure BO | Model-Informed BO |
|---|---|---|---|
| 当前 method 字符串 | Model-Only Greedy | Standard BO | Physics-Informed BO |
| Initialization | reference；需要 physics adapter | reference；zero-history GP | reference；physics+residual GP |
| 总 full-trial calls | K，默认4 | K，默认4 | K，默认4 |
| Reference | 各 method 第1次 evaluate | 同左 | 同左 |
| 每步信息 | past valid scalar J + prior | past valid scalar J | past valid scalar J + prior |
| Candidate domain | 调用者传入同一625 subject domain才一致 | 同左 | 同左 |
| Endpoint | environment 决定；physics须同定义 | environment 决定 | environment 与 physics必须同定义 |
| Allowed mask | domain成员−executed IDs | 同左 | 同左 |
| Acquisition | 最小 model mean | LCB | LCB |
| Result fields | measured-best与model recommendation | 同左 | 同左 |
| benchmark primary final | model recommendation | model recommendation | model recommendation |

Reference 是同一初始化条件，不是同一 `EpisodeObservation` 对象被复用：V1 每方法重建 environment，相同 seed/candidate/trial 产生同样 reference noise；V2 smoke deterministic。未来真实执行若每方法各采 reference，应计入其预算和变异；若共享校准，也必须明确共享。

当前比较的主要局限：

- `run_equal_budget.py` 三方法同预算，但 physics 是由 truth case 构造的 analytic prior、不是 refitted five-parameter model。
- `run_rom_gated_v2_smoke.py` 有 full-dynamics adapter，但仅运行 Reference / Space Filling / Standard BO / Physics-Informed BO，未包含 Random 和 Greedy。不能声称该现成 smoke 已完成所列四个 baselines 的完整公平比较。
- 五腿 benchmark/endpoint study 根本不运行这三条 sequential pipelines；没有 E2 equal-budget algorithm comparison，且 E2 necessity 输出明确 `READY_FOR_E2_FROZEN_ALGORITHM_COMPARISON=false`。
- 所有方法共享“未校验 predicted validity”的限制，不等于拥有 real safe mask；observation endpoint/unit/ROM 的 cross-component compatibility 也未强制。
- 当前报告的 predicted-final 与期望 measured-final 不同；K4/K5、alpha/beta、E0/E2、不同初始化数据成本之间不能直接汇总胜率。

## 15. Simulator role

五腿是固定 pelvis、两关节的五个机械定义，不是五个人体实测样本，也不是 MyoSuite MyoLeg 肌肉模型本身。两条 simulator 路径需分开：

- 五腿：`model.py:make_mujoco_model` → `replay.py:replay_trajectory`。代码逐样本直接赋 qpos/qvel/qacc，再 `mj_inverse`，保留 `theta_shank=q_hip−q_knee`（MuJoCo knee 用负号）和 cuff-equivalent point。
- MyoLeg：`external_simulation/myoleg_v3_development_truth_landscape_generation_v1/replay_api.py:replay_v3_subject_candidate`；`truth_access.query` 限定 purpose 并委托具体 replay。已有 development-only landscape/cohort 边界，但不是当前 BO environment 的自动连接，也不是新的个体化必要性证据。

五腿 `tracking_error` 是写入同一 q 后再读回计算；**不是前向动力学闭环跟踪性能，也不是机器人轨迹执行验证**。可以说验证轨迹输入、坐标变换和逆动力学 replay 一致性。

当前冻结结果 `results_e2_necessity_v1/study_summary.json` 实际为：

```text
E2_PERSONALIZATION_NECESSITY = NOT_SUPPORTED
READY_FOR_E2_FROZEN_ALGORITHM_COMPARISON = false
```

该 artifact 给出的 cohort common relative regret=0.000141368074…（0.0141368%），Leg 4=0.000706840370…（0.0706840%）；这些是冻结离线数据读取，本轮未重跑全625×5 landscape。`FINAL_PERSONALIZATION_RESEARCH_MAINLINE_V1.md` 的结论与之吻合。

可支持 physics prior 开发、方法测试、轨迹/坐标一致性、mechanical landscape 表征；不能支持 real patient personalization、comfort、临床安全/疗效。仿真 NOT_SUPPORTED 也不等于真实患者无需个性化。

## 16. Real-measurement role

真实执行链仍是 frozen slow reference → FK equivalent pull point → start-anchored TCP → reviewed preflight → executor；见 `control/start_anchored_relative_trajectory.py:build_start_anchored_relative_trajectory`、`control/robot_trajectory_executor.py`、`scripts/run_rehab_experiment.py`。这条入口并非任意 `SubjectSpecificCandidate` 的自动执行器。

采集层的 state/wrench/snapshot 和日志流存在；与 V3 的 episode metadata、beta/branch/ROM 及 normalized measurement-analysis schema 仍需明确转换/实验标注。目前 `load_analysis_input` 读取 normalized JSON / CSV / episode JSON，不自动将全部原始五文件 episode 转为带完整 V3 标签的分析输入。

`RealRobotEnvironment.evaluate` 直接抛 `REAL_ROBOT_ENVIRONMENT_DISABLED: NOT_ROBOT_APPROVED`；Stage 0 real ROM interface 也关闭。测量分析 readiness 不解除 robot execution gate；本轮没有检查或操作机器人。

因此：下一阶段可以使用现成离线分析器处理按协议整理的真实日志；仍需真实测量与 geometry/threshold 依据、输入整合和前述 analysis 边界修复。不能把软件 demo、旧硬件 timing diagnostics 或 synthetic repeated episodes 当作已完成的真实三级验证。

## 17. Legacy / dead architecture 与不一致项

### 17.1 分类

| 文件/家族 | 分类 | 给未来开发者的解释 |
|---|---|---|
| `FINAL_PERSONALIZATION_RESEARCH_MAINLINE_V1.md`、current V3、ROM、measurement analysis | CURRENT | 当前研究方向与可复用组件；注意本报告列出的集成缺口 |
| `personalization/sequential.py` 及 models/selectors | CURRENT offline | 可运行方法基础；实际LCB，未启用真实personalization |
| 五腿 `endpoint_design.py` / `e2_necessity.py` | CURRENT | E2 主仿真表征与停止结论 |
| 五腿原 `run_benchmark.py` / frozen parameter JSON / `results/` | LEGACY_BUT_VALID within current package | E0 原始benchmark；不能冒称当前E2主结果 |
| `FullDynamicsGrayBoxEndpointAdapter` 的统一 reference路径、V1 analytic benchmark | LEGACY_BUT_VALID / development | 保留共用机械模型与先验质量实验；不是逐subject ROM的正式端到端验证 |
| `lower_limb_sim/equal_budget_model_informed_bo_baseline.py` | LEGACY_BUT_VALID | 旧alpha/EI，非当前V3/LCB |
| `final_model_screened_finite_sequential_validation.py` 与 runner | LEGACY_BUT_VALID | frozen Top-3/additional validation，不是最终V3主算法 |
| `sequential_personalization.py`、P2 revision/trust-region/prospective 各链及其动画 | LEGACY_BUT_VALID / EXPERIMENTAL | 历史alpha任务、阈值/预算与失败结论都保留，勿拼入现主线 |
| `external_simulation/myoleg_v2_candidate_domain_design_v1/` 与 V2 truth | LEGACY_BUT_VALID | 21025 proposals→16675 admissible，非625域 |
| MyoLeg cohort expansion / structural heterogeneity builders | EXPERIMENTAL, frozen historical | 当前停止扩展；函数存在不代表获准继续制造oracle多样性 |
| Adaptive Trust / Failover / Active Diagnostic / K5 | EXPERIMENTAL / SECONDARY ROBUSTNESS | 保留，不作为主算法“升级版” |
| `README.md`、`CURRENT_ARCHITECTURE.md` | LEGACY_BUT_VALID for robot stack; MISLEADING as whole-repo algorithm map | 主要描述8月机器人收口，缺少现有personalization/measurement主线 |
| `PROJECT_AUDIT.md`、`CODE_CLEANUP_REPORT.md` | LEGACY_BUT_VALID | 历史清理记录，旧PINN/MPC路径被列出不代表仍可执行 |
| 历史文档中无版本前缀的“final method”“primary K5”“primary endpoint unchanged” | OBSOLETE / MISLEADING if treated as current | 须限定至对应实验阶段，不能覆盖最终freeze |

未把“旧”直接等同 dead code：许多历史模块仍有 runner/test/import 使用，删除会破坏可复现性；本轮不作无依据的不可达宣判。存在两套 GP、两套五参数 fitting 是可确认的重复职责/不同契约，不能只按同名合并。

### 17.2 优先级明确的不一致清单

| ID | 严重性/影响 | 代码位置与实际问题 |
|---|---|---|
| A01 | 主要方法不符合预期 | `sequential._components` / `selectors/bo.py`：LCB，非EI |
| A02 | Endpoint未集成 | `rom_gated_v2/physics.py` 返回E0类RMS；E2仅在五腿后处理中 |
| A03 | Evidence-gate未贯通 | `ROMGatedPersonalizationEpisode.run` 不要求measurement/interaction gate |
| A04 | 最终决策口径不同 | `benchmarks/metrics.evaluate_run` final是model recommendation；best_seen也不是measured-best regret |
| A05 | Gray-box辨识陈述易夸大 | `FullDynamicsGrayBoxEndpointAdapter.fit` 为1 scalar/trial的正则拟合；不接完整时序estimator、不检查优化成功 |
| A06 | Measurement判据缺口 | `analysis.py:183,395,829,1028` 不强制必需criterion组合；只给计数条件也可ready |
| A07 | 明确eligibility遗漏 | `analysis.py:analyze_same_trajectory_repeatability` correlations未排除geometry未验证projection |
| A08 | Allow/safe语义缺口 | `selectors/bo.py` / `greedy.py` 不检查Prediction.valid；无额外dynamic safe mask |
| A09 | ROM/endpoint上下文防混用不足 | episode、model.fit和ledger未强制environment/model/observation身份与单位一致 |
| A10 | Budget迁移未完成 | 当前含reference的K4 vs 历史reference外validation K0–3/K1–5 |
| A11 | Synthetic prior优势应披露 | analytic `_physics` 从case optimum构造先验，不能等同无truth先验辨识 |
| A12 | 文档与命名版本滞后 | README全仓描述、Stage1/2、Standard/Pure、Physics/Model-Informed、旧primary标签 |
| A13 | 物理验证表述边界 | 五腿exact-state replay tracking=0不构成闭环控制/实机跟踪结果 |
| A14 | 集成/复现限制 | 多个current模块与文档本来未跟踪；当前工作树通过不代表干净checkout已具备全部代码/产物 |

初始工作树已包含 `.DS_Store`、`requirements.txt` 修改，以及 ROM V2、五腿、measurement、robustness 等大量未跟踪文件。本报告未覆盖它们、未纳入提交；不能把工作树状态错误归因于本轮。

## 18. Recommended minimal cleanup、测试与可用表述

### 18.1 最小后续整理建议（本轮未执行）

1. 给 README / CURRENT_ARCHITECTURE 增加 current research map 链接与版本作用域；在旧 alpha/E0/EI/Top-3 文档前注明历史角色。保留历史 artifact，不追改其冻结结果。
2. 用文档/报告字段先明确 `K_total=1+K_additional`、Stage 0成本、invalid计数、reference是否复用；不要不经审阅就改变 runner 默认预算。
3. 保留 measured-best 与 predicted recommendation 两字段，后续将所需 primary report 明确指向 measured-best，并修正对应 regret 口径；不把未执行recommendation当成执行结果。
4. 最小修复 L2 correlation eligibility；将对真实gate必要的criterion集合绑定外部预声明协议，并明确低层函数不是独立资格认证。这比继续开发新selector优先。
5. 后续若恢复集成，增加 observation↔candidate↔ROM↔endpoint单位的一致性校验及 prediction.valid handling；不能伪造物理安全mask。
6. 如研究后来有证据支持比较，再明确决定是否为当前V3增加EI和相容的endpoint adapter，以及用哪一种五参数fitting。当前E2 necessity不支持继续比较，因此**不建议现在自动实现这些算法变更**。
7. 将已有工作树中需要保留的新文件作为单独版本管理整理任务；本轮不自动stage/commit。

### 18.2 本轮 tests

运行现有测试，未增加/修改测试、算法、阈值或expected结果：

| 分组 | Existing test files | 实际结果 |
|---|---|---|
| Current core | `tests/test_physics_informed_sequential_personalization_v1.py`; `tests/test_subject_specific_rom_gated_sequential_personalization_v2.py`; `tests/test_real_measurement_validation_analysis_v1.py`; `external_simulation/test_myoleg_v3_trajectory_parameterization_design.py` | **58 passed in 2.45s** |
| Estimator / historical EI / robustness | `lower_limb_sim/test_parameter_estimator.py`; `lower_limb_sim/test_equal_budget_model_informed_bo_baseline.py`; `tests/test_identify_real_episode.py`; `tests/test_adaptive_physics_prior_trust_personalization_v1.py`; `tests/test_predictive_evidence_physics_prior_failover_v1.py`; `tests/test_active_prior_diagnostic_arbitration_v1.py`; `tests/test_k4_vs_k5_prior_identification_information_budget_study_v1.py` | **96 passed in 13.77s** |
| Five-leg / E2 | `tests/test_five_leg_mujoco_mechanical_benchmark_v1.py`; `tests/test_mechanically_interpretable_endpoint_v1.py`; `tests/test_e2_personalization_necessity_v1.py`; `tests/test_v3_trajectory_objective_discriminability_audit_v1.py` | **32 passed in 10.63s** |

每组命令为相应解释器 `-m pytest -q <上表各组文件>`。前两组用 `python3`；第三组用 `/private/tmp/rehab_architecture_review_venv/bin/python`。仓库 `.venv` 缺 pytest，系统 Python 有 pytest 但最初缺 MuJoCo，最初合并运行出现4个 collection dependency errors，不能算测试断言失败。临时 system-site-packages venv 安装 MuJoCo 3.6.0 后，四个受阻文件全部通过；TLS始终验证，未改系统/仓库依赖。

本轮没有运行 full repository suite：主线已覆盖；全仓含大量历史冻结artifact、平台/SDK与数据前置条件，不能拿历史全仓数字代替本轮结果。测试通过不抵消本报告根据未覆盖边界发现的问题。

可查看本机临时日志：`/private/tmp/rehab_architecture_core_tests.log`、`/private/tmp/rehab_architecture_extended_tests.log`、`/private/tmp/rehab_architecture_mujoco_tests.log`。它们是临时测试输出，不是正式研究数据归档。

另运行现有 measurement `--demo` CLI 至 `/private/tmp/rehab_architecture_measurement_demo/`；三级公开 `INSUFFICIENT`，无robot action。第7节两个边界复现仅使用内存里的demo副本，没有把synthetic数据保存/归档成真实测量。

### 18.3 简历和论文现在可以如何写

可直接据代码支持的简历表述：

> 实现面向仰卧位髋膝康复研究的离线算法框架，包含 subject-specific ROM 冻结与版本化、625种分支协调轨迹、等效五参数动力学模型、残差高斯过程及低预算顺序搜索；建立测量有效性、同轨迹重复性和异轨迹敏感性的分级离线分析流程。

若要写 acquisition，应明确“当前V3实现使用LCB；另有历史alpha域EI基线”，不能简写成“当前完整实现gray-box+residual GP+EI”。若写五参数辨识，应区分时序estimator与BO scalar refit，避免声称4次trial已充分辨识全部参数。

适合论文的研究边界：

> 冻结五腿被动MuJoCo研究中，V3产生可辨识的模型运动学与机械响应差异；同腿参考归一化的分支平衡E2用于机械表征，但当前证据不支持决策相关的协调个性化必要性。已有gray-box/GP/BO保留为离线方法，后续是否启用取决于真实测量的有效性、重复性、敏感性及跨主体决策相关交互证据。

不支持的陈述：已完成真实患者在线个性化；已实机验证E2；证明舒适性/安全性/临床有效性；五参数为生理组织参数；五腿逆动力学零tracking error证明闭环控制效果；current V3 E2 EI equal-budget comparison已完成；整个repository所有测试通过。

最终判断：**当前成果足以准确描述“离线方法与验证基础设施”，不足以描述“已贯通、经真实测量支持的个性化康复控制系统”。**
