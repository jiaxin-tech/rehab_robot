# MyoLeg 个体化研究执行计划 V3

更新时间：2026-09-25。目标是在 2026-10-31 前冻结能够被数据支持的实验主张，2026-11-01 开始论文集中写作。计划围绕一个可证伪主线展开：**主体差异是否改变候选轨迹/辅助机制的相对收益；如果改变，证据门控的低预算算法能否安全地识别并利用这种差异。**

## 当前结论

V1 的两个原生候选族在 24 个 development 主体上共享最优，不能证明个体化必要性。V2 的速度/协调域只有 2/24 个不同 oracle，排序仍高度一致。`CONTROLLED_ACTUATION_V3` 在 3 个主体上出现 2 种 oracle，但相对最佳公共方案的最大额外收益只有 0.003589%，低于预先设定的 0.5% 实用门槛。因此 V1 扩展和 V3 主体扩展都暂停。

这不是“算法已经失败”，而是当前模拟任务的决策相关个体差异太弱：主体主要改变质量、惯量和小范围被动张力；每个采样点都重置到规定的 q/dq/ddq，主动激活与控制清零；V3 辅助是 `tau_native - tau_assistance` 的受控代数负荷分摊，没有模拟主体受到辅助后重新招募、反射、疲劳或改变动作。

## 研究分层

| 层 | 任务 | 可支持的主张 | 当前状态 |
|---|---|---|---|
| A 原生 MyoLeg null | V1/V2/V3 native 或受控力矩屏幕 | 当前模拟任务的共同最优、约束行为和无个体化基线 | 已完成；V1/V3 暂停扩展 |
| B 机制压力测试 | 预声明的角度位置、速度敏感性、髋膝阻力分配交互域 | 该算法在明确的主体—轨迹交互下是否能识别差异 | 已完成 6-profile development pilot；仍不是患者生理证据 |
| C 算法比较 | 固定 common policy、Random、Residual Greedy、Pure EI、SAST/EG-CPI-BO | 在相同预算和安全约束下的 regret、门控假阳性、约束覆盖 | 已完成 controlled resistance development-only 比较；仍不包含 native 或患者证据 |
| D 独立确认 | 冻结协议后访问 held-out 或真实测量 | 预先声明范围内的确认结果 | 未授权、未开始 |

B 层必须和 A 层分开报告。不能用 B 层合成成功替代 A 层动力学，也不能用 A 层缺少差异证明算法无效。

## 下一段代码与实验

实现 `CONTROLLED_RESISTANCE_INTERACTION_V2`，候选坐标继续使用 V3 的 `(duration_scale, assistance_timing, hip_share)`，但响应机制改为可解释的受控阻力场：

- `angle_onset_phase`：被动阻力开始显著增加的分支相位；
- `velocity_sensitivity`：速度增加时阻力的增幅；
- `hip/knee_resistance_share`：主体在髋膝方向上的阻力分配。

阻力场只在 controlled synthetic wrapper 中作用于请求的候选响应，输出仅包含当前请求的标量/约束观测；完整 landscape、主体参数和 oracle 对 learner 不可见。配置、profile 列表、split、seed 和 manifest hash 在运行前冻结。`COMMON_RESISTANCE` 用于 false-positive gate，`DIVERGENT_RESISTANCE` 用于 positive control；两者都明确标记为软件压力测试。

第一轮只做 6 个 development profile：2 个 common null、4 个 divergent profiles（两种阻力起始位置 × 两种速度/分配组合），每个 profile 运行固定预算 8，比较固定 common policy、Random、Residual Greedy、Pure EI 和 SAST/EG-CPI-BO。不得读取 confirmatory profiles；confirmatory profile 参数必须与 development 不同。

### 进入下一阶段的硬门槛

必须同时满足：

1. null profile 的 gate active rate 为 0，或在预设容差内可解释；
2. positive profile 的 oracle 差异在至少 75% profile 中可检测；
3. SAST/EG-CPI-BO 相对 common policy 的主体级 regret 改善中位数至少 0.5%，且 95% bootstrap 区间不跨 0；
4. 约束预测覆盖率、实际违反率和回退次数均有记录，不能用低 regret 掩盖不安全；
5. 结果在 development pilot 完成后冻结算法与阈值，再决定是否扩大主体或写确认方案。

不满足任一条件就停止该机制版本，并把结论写为“当前机制或算法预算不足”，不通过追加网格或调大场强制造异质性。

## 当前 development 机制结果

旧的 `CONTROLLED_RESISTANCE_INTERACTION_V1` 输出是开发期调参后的解析压力测试，保留用于 provenance，但不能作为独立或预注册结果。当前 `CONTROLLED_RESISTANCE_INTERACTION_V2` 代码将其明确标为 analytical stress field，并将公式版本、系数、代码哈希和真实约束结果写入 manifest/协议；它仍不构成 MyoLeg、生理或患者证据。

旧的 [controlled algorithm comparison](../../outputs/myoleg_controlled_resistance_algorithm_v1/REPORT.md) 不能继续按原排名引用：其中 common policy 实际挑选了 reference+probe 中的最好观测，Residual BO 也没有使用 EI。修订后的运行器区分固定 common、Random、Residual Greedy、Pure EI 和 SAST_BO，保存完整 observation/decision sequence，并把候选约束结果与 evaluator truth 分开；修订版结果见 [pilot v6](../../outputs/myoleg_controlled_resistance_pilot_v6/REPORT.md) 与 [algorithm comparison v2](../../outputs/myoleg_controlled_resistance_algorithm_v2/REPORT.md)。这些结果仍只是软件机制证据，不替代独立 held-out 或真实测量。

首次 resistance pilot 使用的 probes 没有覆盖持续时间轴，导致 FAST profile 的门控证据不足；该错误已作为开发诊断保留在旧输出，修正版 protocol 使用短/长周期、早时机和髋分配四个 probes，并以新输出目录重新运行。正式论文只引用修正版结果，不把旧 pilot 当作主结果。

## 算法主线

每个主体固定执行：reference → 校准 probe → residual model → evidence gate → constrained acquisition。证据不足时使用冻结 `COMMON_POLICY`；证据充分时才启用 residual GP 和安全 acquisition；无安全候选时记录 `SAFETY_LOCK` 并回退已观测可行候选。

主比较固定为五个方法：`COMMON_POLICY`、Random、`RESIDUAL_GREEDY`、`PURE_EI`、SAST/EG-CPI-BO。每个方法使用相同候选域、校准 probe、最大预算、噪声实现和先验安全约束；不把 evaluator oracle 暴露给 learner。K=4 使用 reference+2 probes，K=8 使用 reference+4 probes。固定 common 或门控未激活时可以在已观测 reference 已足够作为推荐后提前停止，但必须报告实际试验次数，不能把最大预算写成已执行观测数。主指标是主体级 final selection loss、相对 common regret、gate false-positive/true-positive、实际约束违反率、覆盖率和 fallback 次数。

## 论文交付

| 时间 | 交付 |
|---|---|
| 09-25—09-30 | 已完成：resistance interaction wrapper、manifest、6-profile development pilot 和单元/隔离测试 |
| 10-01—10-07 | 已完成：common/null/positive 算法比较、主体级 regret、门控与安全覆盖；go/no-go 为机制比较可进入、确认实验保持关闭 |
| 10-08—10-14 | 若通过，冻结方法、阈值、候选域与分析脚本；若不通过，冻结 null/negative result，不再加新机制 |
| 10-15—10-21 | 在独立授权前提下运行一次确认集；没有授权就只写 development 证据和局限 |
| 10-22—10-31 | 复核表图、源码 hash、样本数和指标定义，关闭实验范围 |
| 11-01 起 | 写论文：问题与可辨识性、机制模型、证据门控算法、null/positive 对照、native 限制与真实测量计划 |

论文创新点应表述为“在安全约束和有限试验预算下，先判断个体交互是否有证据，再进行个体化优化”。如果 native 或 controlled interaction 均不能产生实际收益，负结果本身就是边界条件，不能把 synthetic stress test 写成患者疗效。
