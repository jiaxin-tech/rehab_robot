# 数据、实验产物与清理说明

[返回文档索引](README.md) · [代码地图](CODE_MAP.md)

本仓库的输出目录同时保存后续计算输入、冻结实验结果和真实机器人诊断记录。目录名含 `outputs`、`results`、`audits` 或 `formal_artifacts`，不代表可以作为缓存删除。整理以保持现有代码入口、证据链接和冻结字节为原则。

## 从哪里找证据

| 位置 | 用途 | 保留方式 |
| --- | --- | --- |
| [bone_return_3_leg.csv](../bone_return_3_leg.csv) | 实测动作源数据，参考轨迹从中选取周期 | 保留原始数据，不用仿真或重采样结果覆盖 |
| [reference_release](../reference_release/) | 当前发布参考轨迹、闭合与版本说明 | 保留全部文件；以 [release manifest](../reference_release/reference_release_manifest.json) 判断当前版本及用途 |
| [lower_limb_sim/data](../lower_limb_sim/data/) | 源数据处理结果、参考候选及局部状态域 | 被忽略的路径也可能是本地数据；保留已有文件，不按整个目录清理 |
| [lower_limb_sim/formal_artifacts](../lower_limb_sim/formal_artifacts/) | 二维机械腿、参数辨识、有限候选序贯决策、P2 等阶段结果 | 保留报告、协议、表格、图像和冻结结果；可再生大 CSV 按专门清单处理 |
| [personalization/benchmarks](../personalization/benchmarks/) | measurement-driven、ROM-gated、adaptive trust、failover 等算法基准及结果 | 保留基准代码及各版本结果；旧版本用于对照，不能只凭版本号删除 |
| [external_simulation](../external_simulation/) | MyoLeg 模型、参考回放、虚拟受试者、候选域及真值生成代码 | 保留模型和冻结输入；研究脚本与结果目录分别管理 |
| [external_simulation_audits](../external_simulation_audits/) | MyoLeg 各阶段设计、协议、manifest、候选表、真值结果和决策记录 | 保留结果及相互引用，不因与代码目录同名而删除 |
| [outputs](../outputs/) | 轨迹敏感性、MyoLeg 载荷、E3 候选比较和低预算实验 | 当前是可被下游直接读取的实验数据，不是通用临时目录 |
| [diagnostics](../diagnostics/) | 真实机器人状态与 wrench 采集、进程隔离、时序比较及离线诊断 | 按一次运行的 CSV、JSON、Markdown、PNG 组合保留 |

关键入口：

- [机器人指南](ROBOT_OPERATIONS.md)：当前诊断与运行边界；[原架构文档](../CURRENT_ARCHITECTURE.md)和[原实验说明](../REAL_ROBOT_EXPERIMENT.md)是保留原路径与字节的历史协议输入。
- [MyoLeg 虚拟受试者说明](../external_simulation/cohorts/myoleg_virtual_patient_cohort_v1/README.md)：32 个冻结虚拟受试者及独立 nominal control；每人以模型增量、metadata 和参考回放数组保存，不能重新抽样替换。
- [MyoLeg 625 条轨迹载荷说明](../outputs/myoleg_trajectory_loads/README.md)：解释 E0/E2、关节力矩与肌肉力，限定结果所对应的模型和条件。
- [V3 development 真值 manifest](../external_simulation_audits/myoleg_v3_development_truth_landscape_generation_v1/MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_V1_MANIFEST.json)及[访问策略](../external_simulation_audits/myoleg_v3_development_truth_landscape_generation_v1/V3_TRUTH_ACCESS_POLICY_V1.json)：保留既有 development / held-out 边界，不因文件整理改变数据访问用途。
- [真实状态与 wrench 对比](../diagnostics/state_wrench_timing_comparison_20260814T093551709145Z.md)：RT-only 完成 900 秒；RT+wrench 在 169.610 秒发生 worker hung。该记录及其原始数据不能被离线测试结果替代。

## 冻结内容与换行

部分 loader、协议和回归测试检查文件的原始字节 SHA。冻结范围不只包含二进制或 CSV，也可能包含 JSON、Markdown 和被协议绑定的源文件。文件内容看起来相同，Windows 的 LF/CRLF 转换仍会改变这些校验结果。

[.gitattributes](../.gitattributes)对已核对的关键文件使用精确路径规则：原始 LF 文件固定为 LF；原始 CRLF CSV 用 `-text` 保持原始字节。不要对全库 CSV 做统一换行转换，也不要批量重写冻结 JSON 的缩进、字段顺序或数字格式。

整理文档时先检查读取路径和现有校验依赖。若旧文档的路径或内容已被程序、协议绑定，可保留原文并在导航页链接；新说明不要覆盖旧结论。不要通过更新预期 SHA 来掩盖非预期内容变化。

源参考与发布参考有时字节相同，但分别承担来源记录和稳定发布入口职责。例如 `reference_release/reference_measured_asymmetric_closed_slow.csv` 的 manifest 显式指向 `lower_limb_sim/data/reference_candidates/` 中的来源文件。这类重复不是待删除副本。

## 大型可再生产物

已有 [GENERATED_LARGE_ARTIFACT_MANIFEST.json](../lower_limb_sim/formal_artifacts/GENERATED_LARGE_ARTIFACT_MANIFEST.json)登记五个大型 CSV：

- `decision_relevant_global_model_reliability_v1/global_prediction_truth_comparison.csv`。
- `p2_revision_root_cause_audit_v1/truth_landscape_baseline.csv`。
- `p2_revision_root_cause_audit_v1/truth_landscape_hip_stiff.csv`。
- `p2_revision_root_cause_audit_v1/truth_landscape_knee_stiff.csv`。
- `p2_revision_root_cause_audit_v1/truth_landscape_heavy_leg.csv`。

以上路径均相对于 `lower_limb_sim/formal_artifacts/`。它们由 [.gitignore](../.gitignore)精确忽略，普通相关回归不要求这些大文件存在；完整历史实验复现可能需要它们。保留登记清单、生成代码、小型结果和[复现说明](../lower_limb_sim/formal_artifacts/p2_checkpoint_and_large_artifact_reproducibility_v1/FORMAL_ARTIFACT_REPRODUCTION.md)。需要回收空间时，可在确认没有任务读取后清理这些已登记的本地副本；不应把这个规则扩展到其他 CSV。

四个小型前置输入仍须随仓库保存：`reference_full_angles.csv`、`detected_cycles.csv`、对应的 `metadata.json`，以及 `reference_local_active_asymmetric/state_domain_bounds.json`。它们在 `.gitignore` 中有明确例外，不是大数据缓存。

`outputs/e3_candidate_comparison/candidates_with_limits.csv` 和 torque NPZ 被 [E3 低预算实验入口](../lower_limb_sim/e3_low_budget/run.py)及[相关测试](../tests/test_e3_low_budget.py)读取；`outputs/trajectory_sensitivity/` 的 torque NPZ 也属于其输入。不要为了缩减输出目录而删除这些文件。

## SDK 与用户配置

[hardware/windows/xcoresdk](../hardware/windows/xcoresdk/README.md)是[当前 Windows 适配器](../hardware/windows/rokae_xcore.py)实际加载的 SDK。原生 `.pyd` 和 `.dll` 是运行依赖，`.pyi` 是接口证据，不能按 Python 缓存清理。

[hardware/xcoresdk_python-v0.7-2.0](../hardware/xcoresdk_python-v0.7-2.0/README.md)是保留的供应商发行包，含示例、不同平台库和接口声明。[历史项目审查](history/PROJECT_AUDIT.md)明确将其列为 `LEGACY_BUT_KEEP`。虽然其中 Windows DLL、Windows 声明和 CHANGELOG 与当前运行目录存在相同副本，但完整发行包承担供应商 API 证据职责，保留整包；Linux/ARM 库也不能仅因当前 Windows 入口未使用就认定为垃圾。

[lower_limb_sim/rehab_robot.code-workspace](../lower_limb_sim/rehab_robot.code-workspace)同时引用本仓库及相邻的 `rehab_robot_paper`，属于用户的多仓库工作配置。它不参与运行，但不能据此推断已废弃。

## 可以清理什么

| 类型 | 处理 |
| --- | --- |
| `.DS_Store` | macOS 文件夹显示元数据，可删除；仓库已忽略。历史报告提到它不构成运行依赖 |
| `__pycache__/`、`.pytest_cache/` | Python/pytest 可再生缓存；任务未使用时可按明确路径清理，不必为清理而清理 |
| `.cache/` | 先区分包下载缓存、绘图缓存与本次验证输出；仅清理已确认可再生且不再需要的项 |
| `.venv/`、`.tools/` | 当前项目 Python 环境和工具运行时，保留；它们已被忽略，不需要加入 Git |
| diagnostics 中相同图片或只有表头的 CSV | 保留所属运行记录包；相同内容不等于重复实验，单独删除会使报告不完整 |
| 历史 GIF、图表与负结果报告 | 保留；是否仍有科学用途由对应实验和引用决定，不能按扩展名批量删除 |

本次明确的垃圾文件清理范围仅为根目录 `.DS_Store`；删除后可从 Git 历史恢复。当前未把旧 SDK、用户 workspace、实验输出、真实诊断或本地运行环境列为删除对象。
