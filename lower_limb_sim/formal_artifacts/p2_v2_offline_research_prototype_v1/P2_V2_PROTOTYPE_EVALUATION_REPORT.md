# P2 V2 Prototype Evaluation Report

## 边界

- Prototype：`P2_V2_OFFLINE_RESEARCH_PROTOTYPE_IMPLEMENTATION_V1`，状态为 `DEFAULT_OFF_SHADOW_EVALUATION_NOT_FORMAL_POLICY`。
- P2 V1 仍是默认策略；本 runner 没有调用 P2、执行 personalization、连接机器人或修改安全配置。
- active reference、ROM_PROTOCOL_V2、`theta_shank=q_hip-q_knee`、五参数模型、机械目标、generator bounds、0.005 tolerance 和 90% support gate 均保持不变。
- 最终状态保持 `OFFLINE_METHOD_REQUIRES_REVISION`、`NOT_HUMAN_READY`、`NOT_ROBOT_MOTION_APPROVED`。

## Part A — designated local validation

严格读取 SHA-256 为 `ffaf01c65f9097bae35d165c25c2dddf5a617fd97835fd3fa5d50604c4beeb55` 的冻结 pair plan。324 对 pair 未重新选择；仅使用 pair ID 和既有 case ID 的 SHA 排序做均衡分配，9 个既有 case 各 36 对。prediction 先计算，随后才附加 fresh offline virtual truth。

Local error 候选：max=0.00168273790494，P95=0.000430956758924，P99=0.00127694201359。这些仍是 research metrics，不是正式 threshold。

## Part B — local uncertainty guard shadow

| guard | exploit | missed improvement | false improvement | conservative rejection |
|---|---:|---:|---:|---:|
| G0 current replay | 20 | 7 | 0 | 314 |
| G1 local max | 0 | 27 | 0 | 314 |
| G2 local P95 | 24 | 3 | 0 | 314 |
| G3 local P99 | 4 | 23 | 0 | 314 |

回答 1：在这批历史 decision opportunities 上，local P95 将 missed improvement 从 7 降到 3，减少 4 个。

回答 2：false improvement 从 0 变为 0，本次 shadow 中没有增加。但单次 offline shadow 不能自动冻结 P95，也不能证明未来 case 的错误率。

## Part C — knee_stiff cumulative shadow

| rule | transitions | predicted cumulative ΔJ | truth cumulative ΔJ | recovered improvement | false acceptance |
|---|---:|---:|---:|---:|---|
| A single | 1 | -0.004467188 | -0.004467188 | 0.000000000 | False |
| B two-step | 2 | -0.008905285 | -0.008905285 | 0.008905285 | False |
| C three-step | 3 | -0.013313785 | -0.013313785 | 0.013313785 | False |
| D five-step | 5 | -0.022042232 | -0.022042232 | 0.022042232 | False |

回答 3：2/3/5-step candidate 在结构上恢复了 knee_stiff 被单步 0.005 rule 阻断的累计改善，且本次 post-hoc truth 中 false acceptance 为 0。但 bundle uncertainty 尚未冻结，因此它解决了已观察到的 stepwise mechanism，尚未构成可启用的正式规则。

## Part D — decision-value stopping shadow

| strategy | explores | reduction | missed opportunity | support increase |
|---|---:|---:|---:|---:|
| current | 32 | 0 | 0 | 33200 |
| K=1 | 7 | 25 | 0 | 19450 |
| K=2 | 11 | 21 | 0 | 21650 |
| K=3 | 15 | 17 | 0 | 23850 |

回答 4：K=1/2/3 在 frozen history 中分别减少 25/21/17 次 exploration，final best trajectory 不变且 missed opportunity=0，因此确实识别出无 decision/model/prediction value 的探索。不过 support increase 同时下降，仍需要 prospective offline shadow 验证，不能自动启用。

## Part E — 最小 P2 V2 修改集合

1. `review_and_freeze_an_independent_designated_local_uncertainty_statistic`
2. `add_a_default_off_local_uncertainty_provider_while_P2_V1_remains_default`
3. `prospectively_validate_same_direction_cumulative_bundles_and_uncertainty_aggregation`
4. `add_default_off_cumulative_bundle_evaluation_for_knee_stepwise_candidates`
5. `record_support_model_prediction_and_decision_value_separately`
6. `prospectively_validate_default_off_K_stopping_before_any_policy_enablement`

回答 5：上述集合是最小研究实现路径，但 designated statistic、bundle uncertainty 和 K 都需要独立复核后才能冻结。当前 prototype 不足以替换 P2 V1。
