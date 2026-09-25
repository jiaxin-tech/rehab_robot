# 全库代码导航

[返回文档索引](README.md) · [研究主线](RESEARCH.md)

本页按研究问题和运行链定位代码。当前离线集成入口是 `personalization/integrated_v2.py`；`lower_limb_sim/` 同时包含共享力学、机器人参考轨迹、历史算法和后续 E3 实验，不能仅按目录名判断代码新旧。环境和测试见[上手指南](GETTING_STARTED.md)，真机操作条件见[机器人指南](ROBOT_OPERATIONS.md)。

## 先区分四条工作线

| 工作线 | 主要入口 | 范围与边界 |
|---|---|---|
| V2 集成：冻结 ROM → 二维 V3 → E0/E2 → 低预算个体化 | [integrated_v2.py](../personalization/integrated_v2.py) 的 `OfflineBOConfiguration`、`run_offline_configuration` | 当前可复用离线 API；25×25 个 `beta_flex/beta_extend` 候选；默认总预算 4，包含参考试验 |
| E3 三维候选与低预算探索 | [e3_candidate_comparison/run.py](../lower_limb_sim/e3_candidate_comparison/run.py)、[e3_low_budget/run.py](../lower_limb_sim/e3_low_budget/run.py) | 独立实验；协调/关键姿态参数加时间分配；不直接接入二维 V3 集成入口 |
| 新 MyoLeg 开发比较 | [myoleg_benchmark/run.py](../lower_limb_sim/myoleg_benchmark/run.py)、[experiment.py](../lower_limb_sim/myoleg_benchmark/experiment.py) | 按请求运行 MyoLeg 动力学，比较七算法、两候选族与预算/噪声；仅 native 和原 development 主体，输出与旧 E3 回放分开 |
| Controlled interaction development | [controlled_resistance_cohort.py](../lower_limb_sim/myoleg_benchmark/controlled_resistance_cohort.py)、[controlled_resistance_experiment.py](../lower_limb_sim/myoleg_benchmark/controlled_resistance_experiment.py)、[controlled_resistance_algorithm_comparison.py](../lower_limb_sim/myoleg_benchmark/controlled_resistance_algorithm_comparison.py) | 预声明角度起始、速度敏感性和髋膝阻力分配的 controlled synthetic profile；null/positive gate 与四方法比较；不改 native、V1 或 sealed cohort |
| ROKAE 参考执行与真实测量 | [scripts/](../scripts/)、[control/](../control/)、[collection/](../collection/) | 参考轨迹前馈执行、日志和离线辨识；尚未贯通真实测量驱动的在线个体化，运动默认关闭 |

V3 是轨迹参数化版本，V2 是 ROM 门控及算法集成版本，E0/E2/E3 是不同端点，三个编号体系不能相互替代。E2 使用四项同腿参考归一化分支 RMS 的最大值；E3 使用其等权均值。读取结果时须同时核对候选域、端点、模型和预算。

## 当前离线个体化：按数据流阅读

| 环节 | 文件 | 负责什么 |
|---|---|---|
| ROM 与候选域 | [rom_gated_v2/rom.py](../personalization/rom_gated_v2/rom.py)、[reference.py](../personalization/rom_gated_v2/reference.py)、[episode.py](../personalization/rom_gated_v2/episode.py) | 冻结 `SubjectROMProfile`，构建受试者自己的 625 点 V3 域；ROM 变化需新 episode |
| V3 参数化实现 | [parameterization.py](../external_simulation/myoleg_v3_trajectory_parameterization_design_v1/parameterization.py)、[candidates.py](../personalization/candidates.py) | 轨迹生成、候选身份和二维坐标；当前代码仍依赖 `external_simulation/` 中的此实现 |
| 环境与观测 | [environment.py](../personalization/environment.py)、[observations.py](../personalization/observations.py)、[identification.py](../personalization/identification.py)、[offline_time_series.py](../personalization/offline_time_series.py) | 单次试验求值、标量端点、时序辨识载荷与离线时序环境；`RealRobotEnvironment` 仍未开放 |
| 五参数时序辨识 | [time_series_graybox.py](../personalization/models/time_series_graybox.py)、[parameter_estimator.py](../lower_limb_sim/parameter_estimator.py)、[full_dynamics.py](../lower_limb_sim/full_dynamics.py) | 从已执行试验时序辨识有效灰箱参数，并预测 E0 或 E2 |
| 端点定义 | [mechanical_endpoints.py](../lower_limb_sim/mechanical_endpoints.py) | E0/E2 身份、单位、RMS 与参考归一化语义；避免仅凭数值混用端点 |
| 模型与下一点选择 | [physics_graybox.py](../personalization/models/physics_graybox.py)、[residual_gp.py](../personalization/models/residual_gp.py)、[standard_gp.py](../personalization/models/standard_gp.py)、[selectors/bo.py](../personalization/selectors/bo.py) | 物理模型包装、残差 GP、纯 GP、EI/LCB；另有 Greedy、Random、Space Filling |
| 执行账本与主循环 | [ledger.py](../personalization/ledger.py)、[sequential.py](../personalization/sequential.py)、[integrated_v2.py](../personalization/integrated_v2.py) | 已执行候选、有效性和预算；集成入口检查离线环境、冻结域与端点一致性 |

建议先读集成入口和 [test_model_informed_bo_architecture_v2.py](../tests/test_model_informed_bo_architecture_v2.py)，再沿上表进入实现。`personalization/__init__.py` 保留旧公共 API；不能因它未导出 `integrated_v2` 就判断 V2 尚未实现。

[run_rom_gated_v2_smoke.py](../personalization/benchmarks/run_rom_gated_v2_smoke.py) 检查 ROM → V3 → K=4 的架构衔接，沿用原方法名；它不是当前时序辨识集成的全部科学比较。运行器与冻结报告应分别阅读。

## 力学、参考轨迹与机器人坐标

| 模块 | 核心入口 | 说明 |
|---|---|---|
| 正式协议 | [formal_experiment_manifest.json](../config/formal_experiment_manifest.json)、[formal_protocol.py](../lower_limb_sim/formal_protocol.py) | 机器人正式参考协议来源；髋 0–120°、膝 5–145°，不等同于每个受试者的 ROM 档案 |
| 运动学与受力 | [kinematics.py](../lower_limb_sim/kinematics.py)、[jacobian.py](../lower_limb_sim/jacobian.py)、[force_mapping.py](../lower_limb_sim/force_mapping.py) | `theta_shank = q_hip - q_knee`；等效束带牵引点 L1=0.42 m、L2=0.30 m |
| 动力学与观测 | [full_dynamics.py](../lower_limb_sim/full_dynamics.py)、[dynamic_subject.py](../lower_limb_sim/dynamic_subject.py)、[observation_model.py](../lower_limb_sim/observation_model.py)、[identification_dataset.py](../lower_limb_sim/identification_dataset.py) | 下肢动力学、虚拟主体、观测及辨识数据 |
| 正式参考加载 | [reference_release.py](../lower_limb_sim/reference_release.py)、[reference_release/](../reference_release/) | 加载冻结的实测非对称、周期闭合慢速轨迹并校验；此目录是运行输入 |
| 参考重建 | [run_reference_measured_asymmetric.py](../lower_limb_sim/run_reference_measured_asymmetric.py)、[reference_measured_asymmetric.py](../lower_limb_sim/reference_measured_asymmetric.py) | 保留实测屈/伸路径，做闭合与 C2 修正；默认重建会写参考候选产物 |
| 坐标、锚点与预览 | [robot_coordinate_transform.py](../lower_limb_sim/robot_coordinate_transform.py)、[start_anchor.py](../control/start_anchor.py)、[start_anchored_relative_trajectory.py](../control/start_anchored_relative_trajectory.py)、[preview_rehab_trajectory.py](../scripts/preview_rehab_trajectory.py) | 模型位移经审核坐标旋转后相对 TCP 起点锚定，保持起始姿态 |

`lower_limb_sim/data/` 保存参考源数据与处理结果；部分文件受 Git 跟踪，不能把整个目录当作临时数据。旧对称参考、旧 C2 参考与当前实测非对称参考是不同构造，保留它们用于比较和复现。

## 仿真实验、E3 与可视化

| 实验 | 运行与实现入口 | 主要结果位置 |
|---|---|---|
| 五腿 MuJoCo 力学基准 | [five_leg_mujoco_v1/](../lower_limb_sim/five_leg_mujoco_v1/)，重点 `model.py`、`replay.py`、`run_benchmark.py` | [results/](../lower_limb_sim/five_leg_mujoco_v1/results/) |
| V3 可区分性、端点设计、E2 必要性 | 同包的 [run_discriminability_audit.py](../lower_limb_sim/five_leg_mujoco_v1/run_discriminability_audit.py)、[run_endpoint_design.py](../lower_limb_sim/five_leg_mujoco_v1/run_endpoint_design.py)、[run_e2_necessity.py](../lower_limb_sim/five_leg_mujoco_v1/run_e2_necessity.py) | 包内 `results_discriminability_audit_v1/`、`results_endpoint_design_v1/`、[results_e2_necessity_v1/](../lower_limb_sim/five_leg_mujoco_v1/results_e2_necessity_v1/) |
| 冻结多腿等预算算法比较 | [frozen_multi_leg_algorithm_benchmark_v1/run.py](../lower_limb_sim/frozen_multi_leg_algorithm_benchmark_v1/run.py)、[environment.py](../lower_limb_sim/frozen_multi_leg_algorithm_benchmark_v1/environment.py)、[report.py](../lower_limb_sim/frozen_multi_leg_algorithm_benchmark_v1/report.py) | [包内 results/](../lower_limb_sim/frozen_multi_leg_algorithm_benchmark_v1/results/) |
| 轨迹敏感性 | [trajectory_sensitivity/study.py](../lower_limb_sim/trajectory_sensitivity/study.py)、[report.py](../lower_limb_sim/trajectory_sensitivity/report.py) | [outputs/trajectory_sensitivity/](../outputs/trajectory_sensitivity/) |
| E3 候选族比较 | [e3_candidate_comparison/run.py](../lower_limb_sim/e3_candidate_comparison/run.py)、[report.py](../lower_limb_sim/e3_candidate_comparison/report.py) | [outputs/e3_candidate_comparison/](../outputs/e3_candidate_comparison/) |
| E3 低预算回放 | [e3_low_budget/run.py](../lower_limb_sim/e3_low_budget/run.py)、[report.py](../lower_limb_sim/e3_low_budget/report.py) | [outputs/e3_low_budget/](../outputs/e3_low_budget/) |
| 新 MyoLeg 实际仿真比较 | [myoleg_benchmark/run.py](../lower_limb_sim/myoleg_benchmark/run.py)、[experiment.py](../lower_limb_sim/myoleg_benchmark/experiment.py)、[simulation.py](../lower_limb_sim/myoleg_benchmark/simulation.py)、[report.py](../lower_limb_sim/myoleg_benchmark/report.py) | `outputs/myoleg_benchmark_v1/<运行时间>/`；协议、推荐结果、试验历史、辨识诊断与独立报告 |
| 视频与 MyoLeg 负载分析 | [visualization/](../lower_limb_sim/visualization/)、[render_mujoco_rehab_video.py](../scripts/render_mujoco_rehab_video.py)、[render_myoleg_robot_video.py](../scripts/render_myoleg_robot_video.py)、[evaluate_myoleg_trajectory_loads.py](../scripts/evaluate_myoleg_trajectory_loads.py) | 具体输出目录由各脚本指定；视频属于展示及分析产物 |

E3 的局部 `GP3D` 明确采用第三个输入坐标，不能直接替换成二维 V3 GP。E3 低预算回放的最终指标是“实际执行且合格的最佳已观察值”；无效试验也消耗预算。其结果应与冻结 V3/E2 结论分开陈述。

新 benchmark 的 [prepare_assets.py](../lower_limb_sim/myoleg_benchmark/prepare_assets.py) 只下载并提取官方 MyoSuite 2.12.2 所需资产；[portable_model.py](../lower_limb_sim/myoleg_benchmark/portable_model.py) 在内存中重定位冻结 XML 的资产路径，不安装整套 MyoSuite、不改模型机械参数。`simulation.py` 检查原 XML/delta 内容；跨平台 compiled fingerprint 不同会保留差异，并要求公开参考轨迹的物理分量数值复现。主体参数、缓存和未执行候选响应不提供给算法。

旧 [myoleg_robot_scene.py](../lower_limb_sim/visualization/myoleg_robot_scene.py) renderer 及部分旧回放脚本仍直接加载含原机器绝对路径的 XML，未自动接入新加载器。新 benchmark 可运行不代表旧视频入口已移植；资产配置命令见[上手指南](GETTING_STARTED.md#myoleg-独立开发实验)。

## 真实机器人与测量分析

| 层次 | 核心代码或脚本 | 与下一层的关系 |
|---|---|---|
| SDK 适配 | [hardware/rokae_adapter.py](../hardware/rokae_adapter.py)、[rokae_motion.py](../hardware/rokae_motion.py)、[hardware/windows/](../hardware/windows/) | 机器人状态/wrench 与运动接口；实际 SDK 副本位于 `hardware/windows/xcoresdk/` |
| 观察型诊断 | [rokae_probe.py](../scripts/rokae_probe.py)、[characterize_wrench_longrun.py](../scripts/characterize_wrench_longrun.py)、[audit_state_wrench_timing.py](../scripts/audit_state_wrench_timing.py) | 探测状态、时序和阻塞，报告保存在 [diagnostics/](../diagnostics/) |
| 进程隔离试验 | [wrench_process_isolation.py](../scripts/wrench_process_isolation.py)、[validate_wrench_process_pair.py](../scripts/validate_wrench_process_pair.py)、[validate_wrench_process_live.py](../scripts/validate_wrench_process_live.py) | 独立诊断实现；尚未成为生产 acquire/execute 的采集后端 |
| 锚点、采集、执行 | [capture_start_anchor.py](../scripts/capture_start_anchor.py)、[acquire_robot_data.py](../scripts/acquire_robot_data.py)、[run_rehab_experiment.py](../scripts/run_rehab_experiment.py) | 用户入口；具体操作条件与运动门控见[机器人指南](ROBOT_OPERATIONS.md) |
| 生产采集与日志 | [real_robot_acquisition.py](../collection/real_robot_acquisition.py)、[episode_logger.py](../collection/episode_logger.py) | state/wrench/alignment 当前仍为线程结构；保存五文件 episode |
| 执行预检与安全约束 | [execution_preflight.py](../control/execution_preflight.py)、[robot_trajectory_executor.py](../control/robot_trajectory_executor.py)、[experiment_safety.py](../safety/experiment_safety.py)、[config/](../config/) | 绑定参考、锚点、坐标与人工审核配置；不会自动接入 BO |
| episode 离线辨识 | [identify_real_episode.py](../scripts/identify_real_episode.py) | 读取已记录 episode 做辨识，不自动运行个体化搜索 |
| 静态验证日志与测量分析 | [static_validation_logging.py](../scripts/static_validation_logging.py)、[dry_run_static_validation_logging.py](../scripts/dry_run_static_validation_logging.py)、[measurement_validation/analysis.py](../measurement_validation/analysis.py)、[run_real_measurement_validation_analysis.py](../scripts/run_real_measurement_validation_analysis.py) | 静态有效性、同条件重复性、轨迹敏感性；分析入口接受规范化 JSON 或合成 demo |
| 通用支持 | [utils/](../utils/) | 时钟、日志、来源记录与信号处理 |

五文件 episode 为 `robot_state.csv`、`robot_wrench.csv`、`trajectory_command.csv`、`aligned_snapshot.csv`、`metadata.json`。`collection/collector.py` 是另一套快照采集实现，职责与新 episode 流式采集不同；整理目录时不按相似命名合并。

## 历史分支与外部仿真为何保留

| 分支 | 入口 | 保留原因 |
|---|---|---|
| 连续参考邻域与旧序贯个体化 | [continuous_reference_neighborhood.py](../lower_limb_sim/continuous_reference_neighborhood.py)、[sequential_personalization.py](../lower_limb_sim/sequential_personalization.py)、[mechanical_objective.py](../lower_limb_sim/mechanical_objective.py) | 旧 alpha 参数化、可接受域与旧双关节目标，不等同于当前二维 V3/E2 |
| 旧 alpha 域 BO | [equal_budget_model_informed_bo_baseline.py](../lower_limb_sim/equal_budget_model_informed_bo_baseline.py) | 三维 alpha 网格、可优化核超参数、追加预算 1/2/3/5；与当前固定核/总预算语义不同 |
| P2 与模型可信度研究 | `lower_limb_sim/p2_*.py`、对应 `run_p2_*.py`，以及 [research_decision_guarded_sequential_personalization.py](../lower_limb_sim/research_decision_guarded_sequential_personalization.py) | 分阶段设计、校准、前瞻验证与失败原因分析；部分读取冻结产物 |
| 标量灰箱旧接口 | [models/physics_graybox.py](../personalization/models/physics_graybox.py) | 旧标量正则拟合被明确保留；文件内的协议和 `PhysicsSubjectModel` 仍由当前时序集成使用 |
| 信任、失效切换、主动诊断与 K5 | [adaptive_trust_v1/](../personalization/adaptive_trust_v1/)、[predictive_failover_v1/](../personalization/predictive_failover_v1/)、[active_diagnostic_v1/](../personalization/active_diagnostic_v1/)、[repeated_active_diagnostic_k5_v1/](../personalization/repeated_active_diagnostic_k5_v1/) | 不同试验协议和决策规则；对应运行器位于 [personalization/benchmarks/](../personalization/benchmarks/) |
| MyoLeg 分阶段研究 | [external_simulation/](../external_simulation/)、[external_simulation_audits/](../external_simulation_audits/) | 包含模型搭建、参考回放、队列、V2/V3 域、truth landscape、必要性及测量方案；既有历史构建器，也有当前导入依赖 |

`external_simulation/` 不能整体移至归档目录：当前 V3 直接导入其中的轨迹参数化，MyoLeg 可视化和 E3 还依赖既有回放代码。`external_simulation_audits/` 的 JSON/CSV/SHA 文件包含冻结输入和研究记录，也不是可统一删除的缓存。

## 测试与产物导航

| 位置 | 测试/内容范围 | 使用方式 |
|---|---|---|
| [pytest-core.ini](../pytest-core.ini) | 显式当前核心集合：算法、E3、测量、日志、预检和 fake/离线诊断 | 从仓库根执行 `python -m pytest -c pytest-core.ini -q`；具体环境与已验证记录见[上手指南](GETTING_STARTED.md) |
| [tests/](../tests/) | 当前算法与机器人接口为主，也保留信任/诊断阶段测试 | 不等于所有历史测试；真机集成单独有环境变量门控 |
| `lower_limb_sim/test_*.py` | 数值基础、参考轨迹、历史 alpha/P2、模型误差与延迟研究 | 部分依赖指定历史输入及冻结产物 |
| `external_simulation/test_*.py` | MyoLeg 各阶段协议、输入隔离、产物一致性及来源校验 | 部分校验历史源文件 SHA，当前兼容修改与历史复现应区分 |
| [reference_release/](../reference_release/) | 正式参考 CSV、manifest 与冻结验证记录 | 运行输入，不能按“输出文件夹”清理 |
| [lower_limb_sim/formal_artifacts/](../lower_limb_sim/formal_artifacts/) | 历史协议和算法验证用冻结文件 | 保持来源与相对路径 |
| 各包的 `results*`、[outputs/](../outputs/) | 阶段报告、表格、轨迹或回放数据 | 某些后续实验及测试会读取它们；是否可重建需逐项判断 |
| `.venv/`、`.tools/`、`.cache/`、`__pycache__/` | 本机环境、工具和缓存 | 不属于研究结论；不要与受跟踪实验产物混淆 |

`python -m pytest` 会进一步发现包内历史测试，不能用核心集合通过代替全仓通过。研究运行器常默认写回包内 `results*`，复现前应查看入口的输出参数，避免覆盖已有阶段记录。

## 维护时的路径约束

- [static_strap_pull_geometry_validation_protocol_v1/build_protocol.py](../external_simulation/static_strap_pull_geometry_validation_protocol_v1/build_protocol.py) 硬编码根目录 `CURRENT_ARCHITECTURE.md`、`REAL_ROBOT_EXPERIMENT.md` 和 `README.md`，同时记录内容标记与历史 SHA。调整文档位置或内容会影响该历史构建器的输入校验；冻结 JSON 也保留原路径。
- 多处冻结测试读取固定相对路径；E3 脚本/测试应从仓库根运行。文档整理不需要同步搬动算法包、测试目录或实验产物。
- 目前未证明任何算法模块可无损删除。相似的循环、端点或参考生成器通常对应不同参数域、预算或冻结协议；后续合并应先列出语义差异，再以相应测试验证。
