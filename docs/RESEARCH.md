# 研究主线与证据边界

[返回文档索引](README.md) · [代码地图](CODE_MAP.md)

## 当前研究范围与入口

| 研究部分 | 当前状态 | 入口与说明 |
|---|---|---|
| 固定 ROM 的 V3 与 E2 | 冻结的二维 `beta_flex/beta_extend` 域，共 25×25=625 点；E2 为四个关节/分支 RMS 相对同腿参考比值的最大值 | [FINAL_PERSONALIZATION_RESEARCH_MAINLINE_V1.md](research/FINAL_PERSONALIZATION_RESEARCH_MAINLINE_V1.md)、`personalization/rom_gated_v2/`、`lower_limb_sim/five_leg_mujoco_v1/` |
| 时序灰箱与 BO 集成 | 已实现过去完整时序的五参数辨识、E0/E2 预测、残差 GP、当前二维 V3 的 EI；保留 LCB、Greedy 和 Pure BO | [MODEL_INFORMED_BO_ARCHITECTURE_V2.md](research/MODEL_INFORMED_BO_ARCHITECTURE_V2.md)、`personalization/integrated_v2.py:run_offline_configuration` |
| E3 三参数探索 | 独立的候选族与低预算回放实验，比较 β＋时间分配和关键姿态点样条＋时间分配 | [候选族报告](../outputs/e3_candidate_comparison/REPORT.md)、[低预算报告](../outputs/e3_low_budget/REPORT.md)、`lower_limb_sim/e3_candidate_comparison/`、`lower_limb_sim/e3_low_budget/` |
| 新 MyoLeg 开发实验 | 已完成 native 与原24个development主体、两轨迹族、七算法的比较；关键姿态族三物理方法主K=4并列改善2.97%，工程默认Physics Greedy | [实际报告](../outputs/myoleg_benchmark_v1/development_20260923_r2/REPORT.md)、[研究与论文计划](research/MYOLEG_RESEARCH_AND_PAPER_PLAN.md)；[环境与命令](GETTING_STARTED.md#myoleg-独立开发实验) |
| 真实测量分析 | 已有静态有效性、同条件重复性、轨迹敏感性分析；真实数据到当前个性化环境的适配仍未贯通 | [REAL_MEASUREMENT_VALIDATION_ANALYSIS_V1.md](research/REAL_MEASUREMENT_VALIDATION_ANALYSIS_V1.md)、`measurement_validation/analysis.py`、`scripts/run_real_measurement_validation_analysis.py` |
| 机器人诊断与执行 | 已有 Windows 真机只读诊断；采集可靠性仍有阻塞问题，运动未放行 | [机器人操作与诊断指南](ROBOT_OPERATIONS.md)；`hardware/`、`collection/`、`control/`、`safety/` |

冻结的五腿 V3/E2 结果仍是 `SIMULATED_MECHANICAL_PERSONALIZATION_NECESSITY = NOT_SUPPORTED`：四条腿的最优点为共同参考，共同参考相对个体最优的队列平均相对 regret 仅为 `0.014137%`（以个体最优值为分母）。这一结论只适用于该仿真模型、候选域和机械指标，不能推出真实患者不需要个体化。ROM 个体化与固定 ROM 内的协调个体化是两个层次；后者需要真实测量的有效性、重复性、轨迹敏感性及跨主体决策差异证据。

后续 E3 探索单独使用四项参考归一化 RMS 的等权均值，并保留 E2 观察最坏分量取舍；它没有替换冻结的 E2/V3。其三维输入为两项协调/姿态参数及 `time_share_shift`，不能当作原二维 625 点实验。最新低预算实验覆盖 6 个模型×2 个轨迹族×3 档约束，在参考＋3 次追加的预算下，Model-Informed BO EI 与 Adaptive Greedy 的最终已执行合格最佳 E3 在 **36/36** 个组合相同；当前结果未显示 BO 探索相对该 Greedy 基线的最终收益优势，也不表示两种算法完全相同。候选网格中更低的 E3、边界最优或模型间差异均不直接构成真实患者个体化收益证据。

V2 集成已经接通 `EpisodeObservation + TimeSeriesIdentificationPayload → 五参数时序辨识 → E0/E2 → 残差 GP → EI`，默认总预算 4 包含参考试验。五参数是有效灰箱参数，E0/E2 仍为模型派生指标。`personalization/environment.py:RealRobotEnvironment` 与真实 ROM 测定接口仍关闭；机器人 episode 的离线辨识入口也不会自动运行 BO。

新 MyoLeg benchmark 使用独立的版本化运行链，对算法实际请求的轨迹计算 prescribed-state 广义驱动力矩，保留不可行但有效的观测用于辨识/建模，并把最终推荐限制在已执行、观测合格候选中。七算法包括 Reference、Random、3D Space Filling、Physics Greedy、Residual GP Mean Greedy、Pure BO EI 和 Model-Informed BO EI，用于分辨物理先验、残差修正与 EI 探索各自的作用。主预算为含参考的 K=4，同时报告 K=1/2/8；噪声是合成力矩采样噪声，力矩转换的平面力是代数等效力，均不等同于真实束带测量。

这轮工作不修改冻结 XML、主体 delta 或旧 V3/E2/E3 结果。已完成的 [MyoLeg 开发报告](../outputs/myoleg_benchmark_v1/development_20260923_r2/REPORT.md)包含830个方法/种子运行、3320条预算结果，全部最终推荐真实合格，无运行失败。主K=4时，关键姿态族的Physics Greedy、残差GP Greedy和MI-EI均改善E3约2.9724%，优于本配置的Pure EI（0.9446%）与Random（1.3332%）；三种物理方法均无观测负荷超限，且24个主体推荐同一候选。工程默认可选较简单的Physics Greedy，但不能据此宣称EI探索优势或个体化必要性。BETA族的三物理方法与空间填充同为2.3729%，空间填充每轮平均有2次观测超限。

Native上1%/3%噪声各10种子已完成，仍属单模型合成噪声证据。全24主体30种子噪声、公共轨迹对照和独立确认尚未完成；sealed held-out仍关闭。[研究与论文计划](research/MYOLEG_RESEARCH_AND_PAPER_PLAN.md)给出10月底前实验收尾和后续确认步骤。

[ALGORITHM_ARCHITECTURE_REVIEW_V1.md](history/ALGORITHM_ARCHITECTURE_REVIEW_V1.md) 是 2026-09-18 的阶段审查，其中“当前 V3 未接 EI、E2 adapter 与时序辨识”的缺口已由上述 V2 集成补齐。历史标量拟合、旧 LCB 方法、alpha/EI、P2 和信任/诊断分支仍保留各自作用域；阅读旧报告时应结合 V2 集成文档及后续 E3 报告，不将历史状态当作整个仓库的最新结论。

## 按研究阶段阅读

1. [ROM 门控设计](research/MEASUREMENT_DRIVEN_PERSONALIZATION_ALGORITHM_V2_ROM_GATED.md)说明先确定并冻结 ROM，再比较域内协调；其中早期标量拟合和 LCB 需按原阶段理解。
2. [V2 时序集成](research/MODEL_INFORMED_BO_ARCHITECTURE_V2.md)说明实际数据契约和 EI/E2 实现。
3. [冻结五腿算法比较](research/FROZEN_MULTI_LEG_ALGORITHM_BENCHMARK_V1.md)是集成之后的 E2 主结果、E0/LCB 敏感性实验。四条腿没有 E2 优化余量；第五条腿中 Model-Informed EI 在第一次追加达到 oracle 等值平台，Pure EI 在第三次达到。这个低余量软件实验不构成患者收益证明。
4. [最终 V3/E2 主线](research/FINAL_PERSONALIZATION_RESEARCH_MAINLINE_V1.md)给出冻结结论及转向真实测量的依据。
5. [E3 候选族](../outputs/e3_candidate_comparison/REPORT.md)与[低预算回放](../outputs/e3_low_budget/REPORT.md)是独立后续探索，原始报告与产物留在同一实验目录。
6. [真实测量分析](research/REAL_MEASUREMENT_VALIDATION_ANALYSIS_V1.md)给出静态有效性、重复性和轨迹敏感性的离线接口；实机采集现状见[机器人指南](ROBOT_OPERATIONS.md)。
7. [MyoLeg 研究与论文计划](research/MYOLEG_RESEARCH_AND_PAPER_PLAN.md)区分本轮开发比较、后续噪声与约束实验、最终方法冻结及确认性评估，避免用单模型调试或未完成结果替代跨主体证据。

阶段报告保留当时的完整结论。例如 V2 集成报告中的“本次未做 E2 科学比较”描述该次集成的范围，后续冻结算法报告已经进行了比较；历史架构审查列出的缺口也不能当作最新状态。

## 指标命名不能跨报告混用

| 名称与作用域 | 定义 | 对应入口 |
|---|---|---|
| 当前 E0 | 全周期双关节 RMS | `lower_limb_sim/mechanical_endpoints.py` |
| 冻结 V3 的 E2 | 四个关节/屈伸分支 RMS 相对同腿参考比值的最大值 | V3/E2、V2 时序集成、冻结五腿算法比较 |
| 当前三参数 E3 | 同样四个归一化 RMS 比值的等权均值；同时观察 E2 最坏分量 | `lower_limb_sim/e3_candidate_comparison/`、`e3_low_budget/` |
| 历史端点设计中的 E3 | 髋/膝峰值相对参考比值的最大值 | [历史端点设计](history/DESIGN_MECHANICALLY_INTERPRETABLE_ENDPOINT_V1.md) |

历史 E3 与当前 E3 的名字相同，公式和实验作用域不同。引用时应同时给出报告或 endpoint 标识。原始五腿 E0 景观、后续 E2 必要性分析、V2 算法比较和 E3 探索也各有独立问题与预算，不能合并为一张未经区分的“最新结果”。
