# 代码地图

日常操作见 [实验指南](EXPERIMENT.md)。唯一配置是 [config/experiment.json](../config/experiment.json)，统一入口是 scripts.lab_session。

## 主线架构

~~~mermaid
flowchart TD
    A[统一参数] --> B[prepare：起点和预览]
    B --> C[run --live：显式启动]
    C --> D[物理限值、数据时效和停止]
    D --> E[机器人与假腿采集]
    E --> F[质量检查和本轮报告]
    F --> G[可选 train：对齐与六参数辨识]
    G --> H[残差 GP Greedy 建议]
    F --> I[可选普通 ZIP 归档]
~~~

代码调用：scripts/run_robot_leg_round → scripts/run_rehab_experiment → control + collection + hardware。离线训练通过 measurement_validation → experiment_algorithm，共享 lower_limb_sim 的力学和参考轨迹。

没有源码哈希、签认表、到期放行或证据审计。保留实际限位、力阈值、时间有效性、故障停止和写盘完整性。

## 文件夹

| 目录 | 用途 |
| --- | --- |
| config/ | 一份参数和兼容读取接口 |
| scripts/ | 四个主线动作与少量设备诊断命令 |
| collection/ | 状态、力、ADS 假腿采集及原始写盘 |
| control/ | 起点、坐标、轨迹数值检查及定时发送 |
| hardware/、hardware/windows/ | 机器人适配和 Windows SDK |
| safety/ | 物理限值 |
| measurement_validation/ | 数据质量、时间对齐和测量分析 |
| experiment_algorithm/ | 六参数模型、残差 GP 与 Greedy |
| lower_limb_sim/ | 公共运动学、动力学及参考轨迹工具 |
| lower_limb_sim/myoleg_benchmark/ | 公共模型加载器，目录名沿用旧名 |
| external_simulation/myoleg_supine_rehab_v1/ | 公共右腿机械模型 XML |
| data/ | 一张结果表和一份原始数据/模型 ZIP |
| tests/ | 一个离线回归文件 |
| utils/ | 时钟、日志、普通数据包工具 |
| docs/ | 操作指南和本代码地图 |

## 逐文件用途

表中的名称相对该节目录。所有 Python __init__.py 用于包入口；厂商 .pyi 用于接口声明，不是额外测试或业务逻辑。

### 根目录

| 文件 | 用途 |
| --- | --- |
| [.gitattributes](../.gitattributes) | 保留既有文本字节和厂商 SDK 换行规则 |
| [.gitignore](../.gitignore) | 排除环境、缓存和本地运行产物 |
| [.python-version](../.python-version) | 指定 Python 版本 |
| [README.md](../README.md) | 主线和最短命令入口 |
| [pytest-core.ini](../pytest-core.ini) | 统一测试的收集与运行配置 |
| [requirements-win-py312.lock.txt](../requirements-win-py312.lock.txt) | Windows Python 3.12 固定依赖版本 |
| [requirements.txt](../requirements.txt) | 项目依赖声明 |

### collection

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../collection/__init__.py) | 采集包入口 |
| [collector.py](../collection/collector.py) | 组合机器人状态和内置力查询，记录年龄、时间偏差和有效性 |
| [episode_logger.py](../collection/episode_logger.py) | 四路 CSV 与元数据写盘；写入失败时使会话失效 |
| [real_robot_acquisition.py](../collection/real_robot_acquisition.py) | 管理实时状态、独立力进程和采集生命周期 |
| [safety_guard.py](../collection/safety_guard.py) | 按时效和阈值检查状态与力数据 |
| [snapshot.py](../collection/snapshot.py) | 将不同时刻的运动状态和力组合为有明确有效性的快照 |
| [state.py](../collection/state.py) | 状态类型、单位、时间有效性及力坐标变换 |
| [state_buffer.py](../collection/state_buffer.py) | 有界保留已接收状态帧，检测溢出 |
| [trajectory.py](../collection/trajectory.py) | 通用笛卡尔路径几何与切向投影 |
| [twincat_leg.py](../collection/twincat_leg.py) | 只读 ADS 假腿过程映像解码与独立采集进程 |
| [wrench_process.py](../collection/wrench_process.py) | 独立进程内查询力，监督预算、心跳和审计写盘 |
| [wrench_test_provider.py](../collection/wrench_test_provider.py) | 离线测试的模拟力源，不是真机采集入口 |

### config

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../config/__init__.py) | 配置包入口 |
| [bundle.py](../config/bundle.py) | 读取精简参数，为旧文件路径接口生成可丢弃的章节视图；无哈希或审批 |
| [experiment.json](../config/experiment.json) | 唯一实验参数文件：连接、限值、假腿、标定、坐标、采集、分析和训练八组 |
| [settings.py](../config/settings.py) | SI 单位、数据结构及兼容代码常量；真实执行还需通过已审核配置 |

### control

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../control/__init__.py) | 轨迹与执行接口包入口 |
| [execution_preflight.py](../control/execution_preflight.py) | 检查轨迹数值、物理限值、实际起点和实时采集健康 |
| [robot_trajectory_executor.py](../control/robot_trajectory_executor.py) | 定时发送轨迹；超限、数据失效或异常时停止 |
| [start_anchor.py](../control/start_anchor.py) | 采集、存储和验证实际起点及固定 TCP 姿态 |
| [start_anchored_relative_trajectory.py](../control/start_anchored_relative_trajectory.py) | 把相对康复参考锚定到实际起点，并审计轨迹 |

### data

| 文件 | 用途 |
| --- | --- |
| [experiment_data.zip](../data/experiment_data.zip) | 原始数据、必要结果 CSV、参考轨迹和公共模型资产，按普通路径存储 |
| [experiment_summary.csv](../data/experiment_summary.csv) | 三列长表：source、metric、value |

### docs

| 文件 | 用途 |
| --- | --- |
| [CODE_MAP.md](../docs/CODE_MAP.md) | 本文：架构、文件夹和逐文件用途 |
| [EXPERIMENT.md](../docs/EXPERIMENT.md) | 安装、八组参数、四个实验动作和结果说明 |

### experiment_algorithm

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../experiment_algorithm/__init__.py) | 无硬件依赖的算法包入口 |
| [config.py](../experiment_algorithm/config.py) | 算法阈值、预算和方法配置；属于研究设定 |
| [coordinate_consistent_model.py](../experiment_algorithm/coordinate_consistent_model.py) | 用公共机械先验构建设计矩阵和六参数模型预测 |
| [coordinates.py](../experiment_algorithm/coordinates.py) | MuJoCo 关节坐标、仰卧映射及虚功切向计算 |
| [coupled_model.py](../experiment_algorithm/coupled_model.py) | 固定六参数的顺序、边界和尺度 |
| [policy.py](../experiment_algorithm/policy.py) | 根据已观测数据选择候选、检查预算和停止；轮次入口使用冻结 Greedy 配置 |
| [surrogate.py](../experiment_algorithm/surrogate.py) | 六参数拟合、残差 GP、参数不确定性及诊断 |

### external_simulation/myoleg_supine_rehab_v1

| 文件 | 用途 |
| --- | --- |
| [myoleg_supine_right_v1.xml](../external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml) | 仰卧右腿公共机械模型，供六参数先验使用 |

### hardware

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../hardware/__init__.py) | 硬件边界包入口 |
| [network_probe.py](../hardware/network_probe.py) | 不加载 SDK 的控制器网络可达性检查 |
| [rokae_adapter.py](../hardware/rokae_adapter.py) | 面向采集的机器人只读适配接口 |
| [rokae_commissioning.py](../hardware/rokae_commissioning.py) | 现场调试接口，包括显式上电、复位、有界点动和停止 |
| [rokae_motion.py](../hardware/rokae_motion.py) | 实际运动 SDK 边界，供已放行的执行器调用 |
| [safety_events.py](../hardware/safety_events.py) | 带主机时间的安全事件缓存，事件过期使许可失效 |
| [wrench_session.py](../hardware/wrench_session.py) | 力查询子进程专用只读 SDK 会话 |

### hardware/windows

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../hardware/windows/__init__.py) | Windows 驱动包入口 |
| [rokae_internal_wrench.py](../hardware/windows/rokae_internal_wrench.py) | 封装 getEndTorque 内置力查询及转换 |
| [rokae_xcore.py](../hardware/windows/rokae_xcore.py) | xCoreSDK Windows 底层封装，接收实时状态并提供设备接口 |

### hardware/windows/xcoresdk

| 文件 | 用途 |
| --- | --- |
| [CHANGELOG.md](../hardware/windows/xcoresdk/CHANGELOG.md) | 厂商 SDK 版本变更 |
| [README.md](../hardware/windows/xcoresdk/README.md) | 厂商 SDK 使用说明 |
| [xCoreSDK.dll](../hardware/windows/xcoresdk/xCoreSDK.dll) | Windows 原生 SDK 动态库 |
| [xCoreSDK_python.cp312-win_amd64.pyd](../hardware/windows/xcoresdk/xCoreSDK_python.cp312-win_amd64.pyd) | CPython 3.12 x64 原生扩展 |

### hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey

| 文件 | 用途 |
| --- | --- |
| [LogReporter.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/LogReporter.pyi) | 日志事件字段 |
| [MoveExecution.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/MoveExecution.pyi) | 运动执行事件字段 |
| [RlExecution.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/RlExecution.pyi) | SDK RlExecution 事件字段 |
| [Safety.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/Safety.pyi) | 安全事件字段 |
| [__init__.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/__init__.pyi) | 事件信息键的命名空间 |

### hardware/windows/xcoresdk/xCoreSDK_python

| 文件 | 用途 |
| --- | --- |
| [RtSupportedFields.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/RtSupportedFields.pyi) | 实时流支持字段 |
| [__init__.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/__init__.pyi) | SDK 顶层类型和接口 |
| [model.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/model.pyi) | 机器人模型接口 |
| [motioncontrolRT.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/motioncontrolRT.pyi) | 实时运动控制接口 |
| [planner.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/planner.pyi) | 轨迹规划接口 |
| [utility.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/utility.pyi) | SDK 工具类型与接口 |

### lower_limb_sim

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../lower_limb_sim/__init__.py) | 下肢力学与轨迹工具包入口 |
| [config.py](../lower_limb_sim/config.py) | 下肢几何及模型配置类型 |
| [derivative_estimation.py](../lower_limb_sim/derivative_estimation.py) | 从实测角度和时间估计速度、加速度 |
| [dynamic_subject.py](../lower_limb_sim/dynamic_subject.py) | 动态虚拟受试者及模型验证支持 |
| [force_mapping.py](../lower_limb_sim/force_mapping.py) | 关节力矩到牵引点力的映射 |
| [formal_protocol.py](../lower_limb_sim/formal_protocol.py) | 共享关节范围、坐标约定和默认参考路径；没有发布清单 |
| [full_dynamics.py](../lower_limb_sim/full_dynamics.py) | 两连杆完整动力学计算 |
| [geometry_error_metrics.py](../lower_limb_sim/geometry_error_metrics.py) | 几何观测误差和数据域评估，含仅供评估使用的真值比较 |
| [jacobian.py](../lower_limb_sim/jacobian.py) | 牵引点对髋、膝角的 Jacobian |
| [kinematics.py](../lower_limb_sim/kinematics.py) | 仰卧位髋膝前向运动学 |
| [mechanical_endpoints.py](../lower_limb_sim/mechanical_endpoints.py) | E0/E2 力学评价端点计算 |
| [observation_model.py](../lower_limb_sim/observation_model.py) | 将牵引点力映射为关节力矩观测 |
| [parameter_estimator.py](../lower_limb_sim/parameter_estimator.py) | 既有离线动力学参数辨识器，供实测 episode 适配器调用 |
| [quasi_static_dynamics.py](../lower_limb_sim/quasi_static_dynamics.py) | 准静态重力及被动刚度力矩 |
| [reference_closed_c2.py](../lower_limb_sim/reference_closed_c2.py) | 闭合 C2 连续参考构造及来源复现 |
| [reference_execution_trajectory.py](../lower_limb_sim/reference_execution_trajectory.py) | 参考执行版本和关节活动范围门禁 |
| [reference_measured_asymmetric.py](../lower_limb_sim/reference_measured_asymmetric.py) | 分别使用实测屈曲、伸展构造非对称周期参考 |
| [reference_trajectory_retiming.py](../lower_limb_sim/reference_trajectory_retiming.py) | 按相位重定时参考路径并作软件动力学评估 |
| [robot_coordinate_transform.py](../lower_limb_sim/robot_coordinate_transform.py) | 人体局部坐标、机器人基座和工具坐标变换 |
| [robot_trajectory_audit.py](../lower_limb_sim/robot_trajectory_audit.py) | 离线审计 TCP 轨迹连续性、数值和速度等指标 |
| [run_robot_trajectory_export.py](../lower_limb_sim/run_robot_trajectory_export.py) | 离线导出机器人轨迹文件并验证，不连接硬件 |
| [state_history_buffer.py](../lower_limb_sim/state_history_buffer.py) | 有界下肢状态历史，保留髋膝原始角度约定 |
| [trajectory_profiles.py](../lower_limb_sim/trajectory_profiles.py) | 软件验证用最小 jerk 关节轨迹 |
| [virtual_subject.py](../lower_limb_sim/virtual_subject.py) | 准静态算法验证的虚拟受试者参数 |
| [visualize_reference_retiming.py](../lower_limb_sim/visualize_reference_retiming.py) | 参考相位和重定时轨迹绘图 |
| [visualize_robot_trajectory.py](../lower_limb_sim/visualize_robot_trajectory.py) | TCP 路径、工作空间和坐标关系的离线预览 |

### lower_limb_sim/myoleg_benchmark

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../lower_limb_sim/myoleg_benchmark/__init__.py) | 公共模型资产包入口 |
| [portable_model.py](../lower_limb_sim/myoleg_benchmark/portable_model.py) | 加载冻结 XML 和公共模型资产，避免外部绝对路径依赖 |

### measurement_validation

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../measurement_validation/__init__.py) | 实测数据离线分析包入口 |
| [analysis.py](../measurement_validation/analysis.py) | 静载、重复性、轨迹敏感性分析和特征提取 |
| [leg_episode.py](../measurement_validation/leg_episode.py) | 导入假腿 episode，应用标定并检查采样与数据质量 |
| [offline_leg.py](../measurement_validation/offline_leg.py) | 假腿 CSV 采样质量、间隔和稳定性统计 |
| [robot_leg_round.py](../measurement_validation/robot_leg_round.py) | 在同一单调时钟上对齐假腿与机器人数据，重建运动和力矩端点 |

### safety

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../safety/__init__.py) | 导出物理限值接口 |
| [experiment_safety.py](../safety/experiment_safety.py) | 运动必需的物理限值及数值校验；没有签认状态 |

### scripts

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../scripts/__init__.py) | 脚本包入口，支持 python -m 调用 |
| [acquire_robot_data.py](../scripts/acquire_robot_data.py) | 仅观测采集机器人状态与力，管理日志和退出 |
| [capture_start_anchor.py](../scripts/capture_start_anchor.py) | 连接并读取一次机器人起点，生成未审核锚点 |
| [check_rt_state_timing.py](../scripts/check_rt_state_timing.py) | 测量实时状态到达时间和间隔 |
| [check_snapshot_alignment.py](../scripts/check_snapshot_alignment.py) | 测量状态与力快照的时间偏差 |
| [check_wrench_query_timing.py](../scripts/check_wrench_query_timing.py) | 测量直接 getEndTorque 查询耗时 |
| [diagnose_collision_events.py](../scripts/diagnose_collision_events.py) | 观测厂商安全回调和事件时效 |
| [import_dummy_leg_episode.py](../scripts/import_dummy_leg_episode.py) | 离线导入假腿 CSV，检查质量和角度标定 |
| [lab_session.py](../scripts/lab_session.py) | 统一命令：check、prepare、run --live、train |
| [preview_rehab_trajectory.py](../scripts/preview_rehab_trajectory.py) | 从起点和坐标配置生成待审查轨迹预览 |
| [rokae_commission.py](../scripts/rokae_commission.py) | 现场设备调试命令入口，含显式上电、复位及点动动作 |
| [rokae_diagnostic_common.py](../scripts/rokae_diagnostic_common.py) | 诊断脚本共享的 SDK 初始化和输出辅助 |
| [rokae_probe.py](../scripts/rokae_probe.py) | 设备只读连接和状态健康检查入口 |
| [run_real_measurement_validation_analysis.py](../scripts/run_real_measurement_validation_analysis.py) | 运行静载、重复性和轨迹敏感性离线分析 |
| [run_rehab_experiment.py](../scripts/run_rehab_experiment.py) | 内部机器人执行和采集集成；统一停止与设备清理 |
| [run_robot_leg_round.py](../scripts/run_robot_leg_round.py) | 一轮实验编排；完成采集后可选训练，不依赖放行文件 |
| [train_robot_leg_round.py](../scripts/train_robot_leg_round.py) | 独立无硬件进程：六参数辨识、残差 GP 和 Greedy 建议 |

### tests

| 文件 | 用途 |
| --- | --- |
| [test_mainline.py](../tests/test_mainline.py) | 唯一离线回归文件：采集、设备适配、物理停止、简化流程和算法 |

### utils

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../utils/__init__.py) | 工具包入口 |
| [assets.py](../utils/assets.py) | 普通 ZIP 数据读取、模型资产还原及完成轮次归档；没有 SHA 检查 |
| [clock.py](../utils/clock.py) | 全链路共用主机单调高精度时钟 |
| [logger.py](../utils/logger.py) | 项目日志工厂，导入时不创建日志文件 |
| [provenance.py](../utils/provenance.py) | 获取 Git 提交等来源信息 |
| [signal_processing.py](../utils/signal_processing.py) | 平滑微分、低通滤波、归一化和加速度计算 |

## 本地生成目录

| 目录 | 用途 |
| --- | --- |
| results/round_*/ | 本轮原始记录、起点、预览和报告；具体文件见实验指南 |
| .cache/config/ | 为旧接口生成的配置章节视图，无需手工编辑 |
| .cache/data/ | ZIP 内参考轨迹及模型资产的自动还原 |
| .cache/training/ | 训练子进程临时请求和结果 |
| .cache/pytest-core/、logs/ | 测试缓存和运行日志 |
| .venv/、.tools/ | 本机环境与辅助运行时 |
| .git/ | Git 历史 |
