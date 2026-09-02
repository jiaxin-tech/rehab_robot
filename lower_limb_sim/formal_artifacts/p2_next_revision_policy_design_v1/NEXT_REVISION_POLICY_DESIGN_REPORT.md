# P2_NEXT_REVISION_POLICY_DESIGN_V1

Candidate manifest SHA-256: `1102654b003ca3899021dc2e43c3d682053b7e49082e46b3a722b0495db06166`

## 通俗结论

下一版候选没有把旧 P95 简单替换为新 P95。直接一步分支仍是原 P2 V1 的 G0；新增的是 bundle-supported one-step commitment：当一步本身不足以跨过 0.005 时，只查看同轴、同方向、全部节点合规且有独立 2/3/5-step residual calibration 的累计 endpoint。即使 endpoint 证据通过，本轮也只执行第一个 formal-grid 邻点。随后立即 refit 一次五参数模型、重算整张 prediction map，并使旧授权失效。

这样保留了 trial-by-trial 更新；不存在一次执行整个 bundle、预排后续轨迹或 analytic uncertainty scaling。

## R0--R4 development shadow

- R0_P2_V1_G0_NO_BUNDLE_S0: trials=87, EXPLORE=56, EXPLOIT=31, missed=13, false=0, final J=0.987946424, regret=0.022990289.
- R1_G0_BUNDLE_SCALE_P95_S0: trials=87, EXPLORE=56, EXPLOIT=31, missed=13, false=0, final J=0.987946424, regret=0.022990289.
- R2_G0_BUNDLE_SCALE_P95_S2: trials=51, EXPLORE=20, EXPLOIT=31, missed=13, false=0, final J=0.987946424, regret=0.022990289.
- R3_G0_BUNDLE_SCALE_P99_S2: trials=51, EXPLORE=20, EXPLOIT=31, missed=13, false=0, final J=0.987946424, regret=0.022990289.
- R4_G0_BUNDLE_SCALE_AXIS_P95_S2: trials=51, EXPLORE=20, EXPLOIT=31, missed=13, false=0, final J=0.987946424, regret=0.022990289.

## Small-step repair

- P95 scale bundle recovered 0/9 historical direction-consistent paths.
- P99 sensitivity recovered 0/9.
- Old G2 remains a direct one-step comparator; no path-specific rule or percentile was added after truth.

## Scale-only versus scale-by-axis

R2 is scale-only P95 and R4 is scale-by-axis P95. Their outcomes are reported without selecting a winner: R2 final J `0.987946424`, R4 `0.987946424`; R2 missed `13`, R4 `13`.

## Matched/mismatch and stopping

- R2 mismatch executed false improvements: 0; selected bundle endpoint false improvements: 0.
- Mismatch label: `NO_CLEAR_MISMATCH_FALSE_IMPROVEMENT_INCREASE`.
- S2 K=2 changed total trials from 87 to 51, mean final J by 0.000000000, and regret by 0.000000000. K was not tuned.

## New failure modes

Observed case-policy failure-mode rows: 21. They are preserved in `new_failure_mode_audit.csv`; none caused an in-task policy change.

## Data and evidence boundary

- Policy outcomes use only 9 original development and 6 rejected-prospective-now-development cases.
- The 12 independent calibration cases supply residual distributions only and never enter policy outcome selection.
- No new prospective cohort was generated; held-out final test was not read.
- P2 V1 remains unchanged and the new candidate is default-off.
- Active reference, ROM, `theta_shank = q_hip - q_knee`, five-parameter model, objective, generator, 0.005 tolerance, and 90% support gate are unchanged.
- Hardware/control/collection/safety are unchanged; no robot was connected.

## Formal status

`POLICY_DESIGN_REQUIRES_REVISION`

This is synthetic offline development evidence, not prospective validation, human readiness, robot-motion approval, safety validation, or clinical evidence. Status remains `NOT_HUMAN_READY` and `NOT_ROBOT_MOTION_APPROVED`.
