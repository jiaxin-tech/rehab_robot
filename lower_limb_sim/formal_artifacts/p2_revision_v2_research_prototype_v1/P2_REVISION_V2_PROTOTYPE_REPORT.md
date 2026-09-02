# P2 Revision V2 Research Prototype Report

## 状态与实现边界

- Prototype：`P2_REVISION_V2_RESEARCH_PROTOTYPE_V1`，状态 `DEFAULT_OFF_RESEARCH_SHADOW_ONLY`。
- 当前 P2 仍是默认且未修改；prototype 没有 policy override、自动停止、累计决策执行或 truth-policy 输入。
- 最终状态保持：`OFFLINE_METHOD_REQUIRES_REVISION`、`NOT_HUMAN_READY`、`NOT_ROBOT_MOTION_APPROVED`。
- active reference、ROM、`theta_shank = q_hip - q_knee`、五参数模型、mechanical objective、generator definition、0.005 tolerance 和 90% support gate 均未改变。

## Part A — `LOCAL_DECISION_VALIDATION_PROTOCOL_V1` prototype

local neighborhood builder 直接复用既有 `SearchAlpha`、`TrustRegionSteps` 和不裁剪的 coordinate neighborhood；只接受 existing generator lattice 内、恰好一个坐标变化且等于当前 trust step 的相邻点。alpha distance 是 formal grid-normalized L1 distance，不是物理距离。protocol examples 覆盖 initial、half、minimum 三个既有层级；历史误差证据仍只有 initial 层级。

341 个历史 local pairs 均保存 pair id、current/candidate alpha、formal alpha distance、predicted ΔJ、post-hoc truth ΔJ 与 error。它们是 retrospective virtual research pairs，不是 formal designated calibration。

- local max = `0.00170839435891`
- local P95 = `2.412448282e-05`
- local P99 = `0.000863147481512`

所有 metric 均为 `threshold_frozen=false`，没有进入当前 policy。反事实逐 case 使用 leave-one-case-out metric：

| guard | would exploit | missed | false | conservative stop |
|---|---:|---:|---:|---:|
| G0 current replay | 20 | 7 | 0 | 7 |
| G1 local max | 0 | 27 | 0 | 27 |
| G2 local P95 | 24 | 3 | 0 | 3 |

回答 1：local P95 在历史 shadow 中把 missed 从 7 降到 3，减少 4 个；local max 反而把 missed 增至 27。因此“local uncertainty architecture 有价值”不等于“max/P95 已可冻结”。

回答 2：G2 的 observed false improvement 仍为 0，没有增加；这是 retrospective virtual evidence，不是安全或正式 policy 证明。

## Part B — `DECISION_VALUE_AWARE_EXPLORATION_SHADOW_V1`

`exploration_value_history.csv` 对每次 explore 分开记录：

- SUPPORT_VALUE：new supported points；
- MODEL_VALUE：parameter delta、prediction-map delta、validation uncertainty change；
- DECISION_VALUE：best trajectory change、predicted local/global rank-1 change、exploit eligibility change。

共 32 次 explore：support=32、model=0、decision=3，其中 29 次只有 support、没有 model/decision value。

回答 3：本 prototype **没有实际减少 exploration**，actual avoided=0，因为它严格 default-off 且不自动停止；它只识别出 29 个低 decision-value 历史轮次，为下一项经审核的 stopping rule 提供观测层。

## Part C — knee_stiff cumulative improvement

沿现有 generator 的 knee 方向从 0 到 -5，每次使用既有 1° trust step；没有扩界或修改 objective。5 个单步 improvement 为约 0.00435–0.00447，全部低于不变的 0.005；两个连续步骤累计 improvement=0.008905285，已超过 0.005；五步累计=0.022042232。

回答 4：若要解决已观察到的 knee_stiff multi-step gap，**需要把 cumulative decision rule 作为下一版 research design candidate**；但本 prototype 没有启用该规则，truth 也没有反馈给 policy。assessment：`CUMULATIVE_RULE_RESEARCH_CANDIDATE_REQUIRED_TO_ADDRESS_OBSERVED_KNEE_GAP_NOT_FORMAL_POLICY_APPROVAL`。

## Part D — 下一步判断

回答 5：值得继续 P2 V2 的研究实现，但当前仍不值得替换/冻结为正式 P2 V2。最低前提是：独立 designated local validation 覆盖全部 trust levels；人工选择 uncertainty statistic；为 cumulative rule 预声明窗口与防误接受规则；对 decision-value stopping 做独立 shadow/prospective offline 复核。完成前保持 `OFFLINE_METHOD_REQUIRES_REVISION`、`NOT_HUMAN_READY`、`NOT_ROBOT_MOTION_APPROVED`。
