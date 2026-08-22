# 下一步实现 Prompt：P2 V2 Research Prototype

前提：仅在导师/实验负责人审核并冻结 `LOCAL_DECISION_VALIDATION_PROTOCOL_V1` 的数据划分方案、候选选择原则和 `EXPLORATION_VALUE_AWARE_STOPPING_CANDIDATE_V1` 的研究开关后执行。

请实现一个 **research-only、default-off、可与当前 P2 完全对照** 的 P2 V2 prototype：

1. 不修改 active reference SHA、ROM_PROTOCOL_V2、`theta_shank = q_hip - q_knee`、五参数模型、mechanical objective、generator bounds、0.005 equivalence tolerance、90% support gate，且不触碰 hardware/control/collection/safety。
2. 保留 current G0 作为默认行为；新增 uncertainty-provider 接口，但不得把本次 retrospective max/P95/P99 直接写成正式阈值。
3. 建立预声明的 designated local-decision validation split：只包含既有 generator admissible points 与既有 trust-region 单坐标 pair；与 model fitting、adaptation executed outcomes、final held-out test 完全隔离。
4. 只在独立 local validation 上重新报告 n、distribution、P95/P99/max、correlation；候选必须带 provenance，未审核时 fail closed 回到 G0。
5. 对每次 explore 持久化 information gain、support increase、parameter change、prediction-map change、validation-uncertainty change、exploit-eligibility change、best-trajectory change，并分别输出 MODEL_VALUE、SUPPORT_VALUE、DECISION_VALUE。
6. 将 stopping candidate 实现成 disabled shadow evaluation；不允许 support alone 触发继续，也不允许 truth 成为 stopping feature。先复现 S0/S1/S2 counterfactual，再由人工审核是否启用。
7. 输出 current P2 与 prototype 的 paired offline comparison；truth 只用于最后的 post-hoc missed/false/regret 标签，不得用于 proposal、guard、fitting 或 threshold tuning。
8. 新增测试证明默认 P2 bit-for-bit/row-for-row 不变、所有冻结项哈希不变、candidate 未审核不能启用、无机器人连接、无 formal personalization 声明。

完成后状态仍应为 `RESEARCH_ONLY`、`NOT_HUMAN_READY`、`NOT_ROBOT_MOTION_APPROVED`；除非另有独立证据与人工冻结，不得宣称 P2 V2 formalized。
