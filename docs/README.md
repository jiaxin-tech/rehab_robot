# 文档导航

[返回项目概览](../README.md) · [本次整理记录](REPOSITORY_ORGANIZATION.md)

初次阅读按 [开始使用](GETTING_STARTED.md) → [当前研究](RESEARCH.md) → [代码地图](CODE_MAP.md) 的顺序。要准备实机测量时再读 [机器人操作](ROBOT_OPERATIONS.md)；数据、模型及生成产物的位置与重建边界见 [产物指南](ARTIFACTS.md)。

| 文档区域 | 作用 |
|---|---|
| 本目录的指南 | 汇总当前入口、研究状态与操作方法 |
| [research](research/) | 当前研究仍需直接查阅的设计与实验记录；保留各报告的原始作用域 |
| [history](history/) | 早期设计、诊断、否定结果与迁移记录；日期或“当前/下一步”指原报告阶段 |
| [visualization](visualization/README.md) | 三组离线视频的统一入口与原始展示报告 |

## 当前研究的详细依据

1. [冻结 V3/E2 主线决策](research/FINAL_PERSONALIZATION_RESEARCH_MAINLINE_V1.md)：固定 ROM 内协调个体化的研究问题、仿真否定结论与真实测量证据顺序。
2. [ROM 分层与冻结机制](research/MEASUREMENT_DRIVEN_PERSONALIZATION_ALGORITHM_V2_ROM_GATED.md)：`SubjectROMProfile`、subject reference 与 V3 候选域；其中 Stage 1 的标量拟合/LCB 描述属于当时版本。
3. [时序灰箱与 EI 集成 V2](research/MODEL_INFORMED_BO_ARCHITECTURE_V2.md)：已接通的完整时序辨识、E0/E2、残差 GP、EI/LCB 和离线配置。
4. [冻结五腿算法比较](research/FROZEN_MULTI_LEG_ALGORITHM_BENCHMARK_V1.md)：V2 集成之后的实际低预算实验；E2 为该比较主指标，E0/LCB 为敏感性分析。
5. [真实测量分析接口](research/REAL_MEASUREMENT_VALIDATION_ANALYSIS_V1.md)：静态有效性、同轨迹重复性和轨迹敏感性的输入与输出。
6. [E3 候选族比较](../outputs/e3_candidate_comparison/REPORT.md)与[E3 低预算实验](../outputs/e3_low_budget/REPORT.md)：后续独立三参数探索，报告保留在生成器原输出目录。
7. [MyoLeg 开发实测报告](../outputs/myoleg_benchmark_v1/development_20260923_r2/REPORT.md)与[研究及论文计划](research/MYOLEG_RESEARCH_AND_PAPER_PLAN.md)：已完成native＋24 development、七算法比较与native噪声；关键姿态族三种物理方法在K=4并列，E3改善约2.97%，后续仍须检验公共轨迹、全队列噪声与独立确认。运行入口见 [MyoLeg 配置与运行](GETTING_STARTED.md#myoleg-独立开发实验)及 [新 benchmark 代码](../lower_limb_sim/myoleg_benchmark/)。
8. [当前个性化执行计划 V2](research/MYOLEG_PERSONALIZATION_EXECUTION_PLAN_V2.md)与[候选机制设计 V1](research/MYOLEG_MECHANISM_CANDIDATE_DESIGN_V1.md)：后续全域审计已确认 V1 两个候选族各自为 24/24 主体共享最优；[候选空间诊断](../outputs/myoleg_candidate_space_diagnostic_v1/REPORT.md)显示参数网格满秩但主体排序几乎一致；[速度/协调机制筛查](../outputs/myoleg_mechanism_screen_development_v1/REPORT.md)虽把 evaluator-only oracle 增加到 2/24，排序仍高度一致，且不构成个性化收益证据，因此下一步优先引入真正的主体—轨迹交互，而不是盲目加密附近时序点。候选机制设计同时冻结了原生轨迹域与需要 actuation/interaction 接口的辅助时机、髋膝负荷分配域。[K=1/2/4/8 灰箱验证](../outputs/myoleg_graybox_cv_development_v1/REPORT.md)完成 576 个 causal-prefix 拟合，全部命中质量参数上界，绝对负荷预测仍失配。V1 的 30-seed 扩展暂停；后续推进证据门控 SAST/EG-CPI-BO、独立 controlled synthetic 验证和原生 MyoLeg null 验证，目标在 10 月 31 日锁定实验、11 月开始论文。

9. [CONTROLLED_ACTUATION_V3 受控辅助 pilot](research/MYOLEG_CONTROLLED_ACTUATION_V3.md)：固定 2 Nm 总峰值的 27 个候选已完成 3 个 development 主体、81 条响应；两种 oracle 的差异很小，最佳公共方案的最大相对 regret 仅 0.003589%。状态为 `HOLD_NO_PRACTICAL_PERSONALIZATION_SIGNAL`，不扩展到 24 主体或确认实验；该结果是受控力矩模型诊断，不是个性化算法或生理验证。
10. [研究执行计划 V3](research/MYOLEG_RESEARCH_EXECUTION_PLAN_V3.md)：在 V3 native/代数辅助均显示决策差异很弱后，建立了角度起始、速度敏感性和髋膝阻力分配的 controlled resistance interaction；6 个 development profile 的 null/positive gate pilot 已完成，随后完成 common、Random、Residual BO、SAST/EG-CPI-BO 的固定预算比较。该层仍是软件机制压力测试，确认集和患者生理主张均未开始。

这些文档是有先后关系的研究记录，不是同时生效的多份“最终方案”。旧架构审查中未实现的 V3 EI、E2 adapter 与时序辨识已由 V2 补齐；V2 文档写作时未运行的算法比较随后已有独立冻结报告。后续 E3 探索没有覆盖冻结 V3/E2 的必要性结论。

新 MyoLeg benchmark 的输出单独写入 `outputs/myoleg_benchmark_v1/<运行时间>/`，完成状态由各次运行的 `completion.json` 标记，报告由该次数据生成。它不覆写上述冻结结果；资产准备和内存路径重定位也不修改冻结 XML。旧 MyoLeg renderer 的绝对路径限制仍保留。

指标名称必须结合研究阶段阅读：[旧 endpoint 设计](history/DESIGN_MECHANICALLY_INTERPRETABLE_ENDPOINT_V1.md)中的 **E3 是关节峰值比的最大值**；`outputs/e3_*` 中的 **E3 是四个关节/分支 RMS 比的均值**。两者不可互换。

## 机器人文档和原始诊断

[机器人操作指南](ROBOT_OPERATIONS.md)汇总当前命令与已知限制。以下两份 2026-08 文档是旧几何验证协议按原路径和精确内容读取的输入，保留在仓库根目录：

- [CURRENT_ARCHITECTURE.md](../CURRENT_ARCHITECTURE.md)：机器人栈的层次、线程、episode 与执行状态机。
- [REAL_ROBOT_EXPERIMENT.md](../REAL_ROBOT_EXPERIMENT.md)：分阶段实验操作和既有放行条件。

二者是原始文档，不能把其中“尚无 Windows 真机验证”的阶段描述当作最新状态。后续只读诊断已发现阻塞问题，详细记录见 [wrench 长测](../diagnostics/wrench_hardware_validation_20260813T110502Z.md)、[进程隔离验证](../diagnostics/wrench_process_isolation_validation_20260813T113755Z.md)及[最新 RT/wrench A/B 对照](../diagnostics/state_wrench_timing_comparison_20260814T093551709145Z.md)。

## 历史记录索引

| 阶段 | 完整记录 |
|---|---|
| 下肢建模、时序、参考与导出 Stage 1–6 | [开发手册](history/LOWER_LIMB_STAGES_1_TO_6.md)；包的当前简短导航见 [lower_limb_sim](../lower_limb_sim/README.md) |
| 早期标量灰箱与 LCB | [测量驱动算法 V1](history/MEASUREMENT_DRIVEN_PERSONALIZATION_ALGORITHM_V1.md) |
| 物理先验可信度探索 | [Adaptive Trust](history/ADAPTIVE_PHYSICS_PRIOR_TRUST_PERSONALIZATION_V1.md)、[Predictive Failover](history/PREDICTIVE_EVIDENCE_PHYSICS_PRIOR_FAILOVER_V1.md)、[Active Diagnostic](history/ACTIVE_PRIOR_DIAGNOSTIC_ARBITRATION_V1.md)、[K4/K5 信息预算](history/K4_VS_K5_PRIOR_IDENTIFICATION_INFORMATION_BUDGET_STUDY_V1.md) |
| 冻结仿真与 endpoint 演进 | [五腿 E0 基准](history/FIVE_LEG_MUJOCO_MECHANICAL_BENCHMARK_V1.md)、[V3 可区分性](history/V3_TRAJECTORY_OBJECTIVE_DISCRIMINABILITY_AUDIT_V1.md)、[endpoint 设计](history/DESIGN_MECHANICALLY_INTERPRETABLE_ENDPOINT_V1.md)、[E2 必要性复核](history/REEVALUATE_PERSONALIZATION_NECESSITY_WITH_E2_V1.md) |
| 架构阶段检查 | [2026-09-18 架构审查](history/ALGORITHM_ARCHITECTURE_REVIEW_V1.md)；对应实现更新见上方 V2 集成 |
| 早期仓库和协议迁移 | [清理前审查](history/PROJECT_AUDIT.md)、[清理记录](history/CODE_CLEANUP_REPORT.md)、[ROM 迁移](history/ROM_MIGRATION_AUDIT.md) |

原报告正文及结果保留。正文中反引号包裹的代码/数据路径以仓库根目录为基准；可点击的相对链接按文档当前位置解析。历史实验的本地绝对路径、当时测试计数或环境说明保留原意，不作为当前环境配置说明。
