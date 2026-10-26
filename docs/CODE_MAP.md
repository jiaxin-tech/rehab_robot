# 代码地图

唯一配置：[experiment.json](../config/experiment.json)。操作和当前进展见[实验指南](EXPERIMENT.md)。

## 真机操作与依赖

箭头表示调用、数据传递或停止信号；框内标明进程边界。所有路径相对仓库根目录。

~~~mermaid
flowchart TD
    C["config/experiment.json"] --> L["scripts/lab_session.py"]
    L -->|prepare| P["capture_start_anchor.py + preview_rehab_trajectory.py<br/>读取起点、生成轨迹预览"]
    P -->|anchor.json + preview/| R["scripts/run_rehab_experiment.py"]
    L -->|run --live| R
    L -->|observe：不运动| O["scripts/acquire_robot_data.py"]
    subgraph HOST["主机主进程"]
        R --> PF["control/execution_preflight.py<br/>起点、轨迹、物理限值"]
        PF --> EX["control/robot_trajectory_executor.py<br/>定时发送、过期停止"]
        R --> A["collection/real_robot_acquisition.py<br/>状态、力消费、对齐线程"]
        O --> A
        A --> AD["hardware/rokae_adapter.py<br/>hardware/windows/rokae_xcore.py<br/>SDK 状态线程 → 最新缓存"]
        AD -->|缓存直读，不等待 CSV 写盘| A
        EX --> M["hardware/rokae_motion.py<br/>xcore 实时目标缓存与 SDK 回调"]
        A -->|状态、力、对齐 CSV| LOG["collection/episode_logger.py"]
        EX -->|命令意图| LOG
        EX -->|实际发送时刻| D["dispatch.csv"]
        A -->|时龄、时差、力限值| EX
    end
    ROB["机械臂控制器<br/>自身碰撞检测与保护停止"]
    subgraph FORCE["独立力查询进程 + 主机监督器"]
        W["collection/wrench_process.py<br/>hardware/wrench_session.py"]
    end
    W -->|SDK getEndTorque| ROB
    W -->|最新样本、查询起止时间| A
    W -->|异步查询日志| LOG
    AD -->|SDK 状态订阅| ROB
    M -->|实时笛卡尔目标| ROB
    ROB -->|RT 故障标志 → 状态失效| AD
    L --> LEG["collection/twincat_leg.py<br/>独立 ADS 采集进程"]
    LEG --> ADS["TcAdsDll → TwinCAT 假腿"]
    LEG -->|原始数据| CSV["leg/leg.csv"]
    LEG -->|采集故障停止信号| EX
~~~

prepare 只读取起点和生成预览；observe 同时记录机器人与假腿；run 才发送运动。状态缓存和原始历史分别读取，保存历史不会决定实时状态的新旧。力查询即使卡住，也不占用主进程的状态和目标发送锁；查询超时、样本过期或状态过期会结束本轮，旧样本不会被重新标时。

默认 robot.collision_source=controller：软件不查询碰撞初始值，不等待事件心跳；反馈保持未知，控制器自身负责碰撞保护。状态线程在 RT 运行时检查 SDK 运动故障，故障使状态失效并触发软件停止。主机仍执行力、位置、速度、数据时效限值；软件不修改控制器碰撞阈值。query、events 仍可用于诊断，不能把其未知状态当作通过。

## 采集后处理

~~~mermaid
flowchart LR
    L["scripts/lab_session.py train"] --> ALIGN["measurement_validation/robot_leg_round.py"]
    RAW["dispatch.csv + leg/leg.csv"] --> ALIGN
    ALIGN --> DATA["measured.csv<br/>按实际发送时间对齐、计算端点"]
    DATA --> WORK["scripts/train_robot_leg_round.py<br/>独立无硬件进程"]
    WORK --> ALG["experiment_algorithm/<br/>六参数辨识 → 残差 GP → Greedy"]
    ALG --> REPORT["report.md + round_status.json"]
~~~

辨识使用标定后的假腿数据，机器人力保留为采集和限值数据。训练建议不自动驱动机器人；observe 没有 dispatch.csv，不能当作动态训练轮次。

| 目录 | 用途 |
| --- | --- |
| config/ | 唯一配置和 SDK 约定 |
| scripts/ | 实验编排与设备诊断 |
| collection/ | 机器人状态、力和 ADS 假腿采集 |
| control/、safety/ | 起点、轨迹、物理限值和停止 |
| hardware/ | 设备适配和厂商 SDK |
| measurement_validation/ | 质量、对齐及论文测量验证 |
| experiment_algorithm/ | 六参数辨识、残差 GP 和 Greedy |
| lower_limb_sim/ | 力学模型、参考轨迹及其复现工具 |
| external_simulation/ | 公共右腿模型 |
| data/ | 一张历史汇总表、一份原始数据与模型 ZIP |
| tests/ | 一个离线回归文件 |
| docs/、utils/ | 操作说明和共享工具 |

## 文件用途

所有 __init__.py 是包入口；厂商 .pyi 是 SDK 接口声明。参考构造和重定时代码用于论文复现，不参与每轮现场操作。

| 文件 | 用途 |
| --- | --- |
| [.gitattributes](../.gitattributes) | 保留既有文本字节和厂商 SDK 换行规则 |
| [.gitignore](../.gitignore) | 排除环境、缓存和本地运行产物 |
| [.python-version](../.python-version) | 指定 Python 版本 |
| [README.md](../README.md) | 主线和最短命令入口 |
| [collection/episode_logger.py](../collection/episode_logger.py) | 四路 CSV 与元数据写盘；写入失败时使会话失效 |
| [collection/real_robot_acquisition.py](../collection/real_robot_acquisition.py) | 管理实时状态、独立力进程和采集生命周期 |
| [collection/snapshot.py](../collection/snapshot.py) | 将不同时刻的运动状态和力组合为有明确有效性的快照 |
| [collection/state.py](../collection/state.py) | 状态类型、单位、时间有效性及力坐标变换 |
| [collection/state_buffer.py](../collection/state_buffer.py) | 有界保留已接收状态帧，检测溢出 |
| [collection/twincat_leg.py](../collection/twincat_leg.py) | 只读 ADS 假腿过程映像解码与独立采集进程 |
| [collection/wrench_process.py](../collection/wrench_process.py) | 独立进程内查询力，监督预算、心跳和审计写盘 |
| [collection/wrench_test_provider.py](../collection/wrench_test_provider.py) | 离线测试的模拟力源，不是真机采集入口 |
| [config/bundle.py](../config/bundle.py) | 直接读取唯一配置；兼容单独诊断参数，定位归档数据 |
| [config/experiment.json](../config/experiment.json) | 唯一实验参数文件：连接、限值、假腿、标定、坐标、采集、分析和训练八组 |
| [config/settings.py](../config/settings.py) | SDK 固定约定；IP、采样频率和数据时效从统一配置读取 |
| [control/execution_preflight.py](../control/execution_preflight.py) | 检查轨迹数值、物理限值、实际起点和实时采集健康 |
| [control/robot_trajectory_executor.py](../control/robot_trajectory_executor.py) | 定时发送轨迹；超限、数据失效或异常时停止 |
| [control/start_anchor.py](../control/start_anchor.py) | 采集、存储和验证实际起点及固定 TCP 姿态 |
| [control/start_anchored_relative_trajectory.py](../control/start_anchored_relative_trajectory.py) | 把相对康复参考锚定到实际起点，并审计轨迹 |
| [data/experiment_data.zip](../data/experiment_data.zip) | 原始数据、必要结果 CSV、参考轨迹和公共模型资产，按普通路径存储 |
| [data/experiment_summary.csv](../data/experiment_summary.csv) | 三列长表：source、metric、value |
| [docs/CODE_MAP.md](../docs/CODE_MAP.md) | 本文：架构、文件夹和逐文件用途 |
| [docs/EXPERIMENT.md](../docs/EXPERIMENT.md) | 安装、八组参数、四个实验动作和结果说明 |
| [experiment_algorithm/config.py](../experiment_algorithm/config.py) | 算法阈值、预算和方法配置；属于研究设定 |
| [experiment_algorithm/coordinate_consistent_model.py](../experiment_algorithm/coordinate_consistent_model.py) | 用公共机械先验构建设计矩阵和六参数模型预测 |
| [experiment_algorithm/coordinates.py](../experiment_algorithm/coordinates.py) | MuJoCo 关节坐标、仰卧映射及虚功切向计算 |
| [experiment_algorithm/coupled_model.py](../experiment_algorithm/coupled_model.py) | 固定六参数的顺序、边界和尺度 |
| [experiment_algorithm/policy.py](../experiment_algorithm/policy.py) | 根据已观测数据选择候选、检查预算和停止；轮次入口使用冻结 Greedy 配置 |
| [experiment_algorithm/surrogate.py](../experiment_algorithm/surrogate.py) | 六参数拟合、残差 GP、参数不确定性及诊断 |
| [external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml](../external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml) | 仰卧右腿公共机械模型，供六参数先验使用 |
| [hardware/network_probe.py](../hardware/network_probe.py) | 不加载 SDK 的控制器网络可达性检查 |
| [hardware/rokae_adapter.py](../hardware/rokae_adapter.py) | 面向采集的机器人只读适配接口 |
| [hardware/rokae_commissioning.py](../hardware/rokae_commissioning.py) | 现场调试接口，包括显式上电、复位、有界点动和停止 |
| [hardware/rokae_motion.py](../hardware/rokae_motion.py) | 实际运动 SDK 边界，供已放行的执行器调用 |
| [hardware/safety_events.py](../hardware/safety_events.py) | 带主机时间的安全事件缓存，事件过期使许可失效 |
| [hardware/windows/rokae_internal_wrench.py](../hardware/windows/rokae_internal_wrench.py) | 封装 getEndTorque 内置力查询及转换 |
| [hardware/windows/rokae_xcore.py](../hardware/windows/rokae_xcore.py) | xCoreSDK Windows 底层封装，接收实时状态并提供设备接口 |
| [hardware/windows/xcoresdk/CHANGELOG.md](../hardware/windows/xcoresdk/CHANGELOG.md) | 厂商 SDK 版本变更 |
| [hardware/windows/xcoresdk/README.md](../hardware/windows/xcoresdk/README.md) | 厂商 SDK 使用说明 |
| [hardware/windows/xcoresdk/xCoreSDK.dll](../hardware/windows/xcoresdk/xCoreSDK.dll) | Windows 原生 SDK 动态库 |
| [hardware/windows/xcoresdk/xCoreSDK_python.cp312-win_amd64.pyd](../hardware/windows/xcoresdk/xCoreSDK_python.cp312-win_amd64.pyd) | CPython 3.12 x64 原生扩展 |
| [hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/LogReporter.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/LogReporter.pyi) | 日志事件字段 |
| [hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/MoveExecution.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/MoveExecution.pyi) | 运动执行事件字段 |
| [hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/RlExecution.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/RlExecution.pyi) | SDK RlExecution 事件字段 |
| [hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/Safety.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/Safety.pyi) | 安全事件字段 |
| [hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/__init__.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/__init__.pyi) | 事件信息键的命名空间 |
| [hardware/windows/xcoresdk/xCoreSDK_python/RtSupportedFields.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/RtSupportedFields.pyi) | 实时流支持字段 |
| [hardware/windows/xcoresdk/xCoreSDK_python/__init__.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/__init__.pyi) | SDK 顶层类型和接口 |
| [hardware/windows/xcoresdk/xCoreSDK_python/model.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/model.pyi) | 机器人模型接口 |
| [hardware/windows/xcoresdk/xCoreSDK_python/motioncontrolRT.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/motioncontrolRT.pyi) | 实时运动控制接口 |
| [hardware/windows/xcoresdk/xCoreSDK_python/planner.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/planner.pyi) | 轨迹规划接口 |
| [hardware/windows/xcoresdk/xCoreSDK_python/utility.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/utility.pyi) | SDK 工具类型与接口 |
| [hardware/wrench_session.py](../hardware/wrench_session.py) | 力查询子进程专用只读 SDK 会话 |
| [lower_limb_sim/config.py](../lower_limb_sim/config.py) | 下肢几何及模型配置类型 |
| [lower_limb_sim/dynamic_subject.py](../lower_limb_sim/dynamic_subject.py) | 动态虚拟受试者及模型验证支持 |
| [lower_limb_sim/force_mapping.py](../lower_limb_sim/force_mapping.py) | 关节力矩到牵引点力的映射 |
| [lower_limb_sim/formal_protocol.py](../lower_limb_sim/formal_protocol.py) | 共享关节范围、坐标约定和默认参考路径；没有发布清单 |
| [lower_limb_sim/full_dynamics.py](../lower_limb_sim/full_dynamics.py) | 两连杆完整动力学计算 |
| [lower_limb_sim/geometry_error_metrics.py](../lower_limb_sim/geometry_error_metrics.py) | 几何观测误差和数据域评估，含仅供评估使用的真值比较 |
| [lower_limb_sim/jacobian.py](../lower_limb_sim/jacobian.py) | 牵引点对髋、膝角的 Jacobian |
| [lower_limb_sim/kinematics.py](../lower_limb_sim/kinematics.py) | 仰卧位髋膝前向运动学 |
| [lower_limb_sim/mechanical_endpoints.py](../lower_limb_sim/mechanical_endpoints.py) | E0/E2 力学评价端点计算 |
| [lower_limb_sim/myoleg_benchmark/portable_model.py](../lower_limb_sim/myoleg_benchmark/portable_model.py) | 加载冻结 XML 和公共模型资产，避免外部绝对路径依赖 |
| [lower_limb_sim/observation_model.py](../lower_limb_sim/observation_model.py) | 将牵引点力映射为关节力矩观测 |
| [lower_limb_sim/parameter_estimator.py](../lower_limb_sim/parameter_estimator.py) | 既有离线动力学参数辨识器，供实测 episode 适配器调用 |
| [lower_limb_sim/quasi_static_dynamics.py](../lower_limb_sim/quasi_static_dynamics.py) | 准静态重力及被动刚度力矩 |
| [lower_limb_sim/reference_closed_c2.py](../lower_limb_sim/reference_closed_c2.py) | 闭合 C2 连续参考构造及来源复现 |
| [lower_limb_sim/reference_execution_trajectory.py](../lower_limb_sim/reference_execution_trajectory.py) | 参考执行版本和关节活动范围门禁 |
| [lower_limb_sim/reference_measured_asymmetric.py](../lower_limb_sim/reference_measured_asymmetric.py) | 分别使用实测屈曲、伸展构造非对称周期参考 |
| [lower_limb_sim/reference_trajectory_retiming.py](../lower_limb_sim/reference_trajectory_retiming.py) | 按相位重定时参考路径并作软件动力学评估 |
| [lower_limb_sim/robot_coordinate_transform.py](../lower_limb_sim/robot_coordinate_transform.py) | 人体局部坐标、机器人基座和工具坐标变换 |
| [lower_limb_sim/robot_trajectory_audit.py](../lower_limb_sim/robot_trajectory_audit.py) | 离线审计 TCP 轨迹连续性、数值和速度等指标 |
| [lower_limb_sim/run_robot_trajectory_export.py](../lower_limb_sim/run_robot_trajectory_export.py) | 离线导出机器人轨迹文件并验证，不连接硬件 |
| [lower_limb_sim/trajectory_profiles.py](../lower_limb_sim/trajectory_profiles.py) | 软件验证用最小 jerk 关节轨迹 |
| [lower_limb_sim/virtual_subject.py](../lower_limb_sim/virtual_subject.py) | 准静态算法验证的虚拟受试者参数 |
| [lower_limb_sim/visualize_reference_retiming.py](../lower_limb_sim/visualize_reference_retiming.py) | 参考相位和重定时轨迹绘图 |
| [lower_limb_sim/visualize_robot_trajectory.py](../lower_limb_sim/visualize_robot_trajectory.py) | TCP 路径、工作空间和坐标关系的离线预览 |
| [measurement_validation/analysis.py](../measurement_validation/analysis.py) | 静载、重复性和轨迹敏感性分析；输出一份结果 JSON 和一张指标表 |
| [measurement_validation/leg_episode.py](../measurement_validation/leg_episode.py) | 导入假腿 CSV、编码器角度换算和质量检查；也可直接作为 CLI |
| [measurement_validation/offline_leg.py](../measurement_validation/offline_leg.py) | 假腿 CSV 采样质量、间隔和稳定性统计 |
| [measurement_validation/robot_leg_round.py](../measurement_validation/robot_leg_round.py) | 按真实发送时刻对齐假腿数据、估计导数、计算论文端点指标 |
| [pytest-core.ini](../pytest-core.ini) | 统一测试的收集与运行配置 |
| [requirements-win-py312.lock.txt](../requirements-win-py312.lock.txt) | Windows Python 3.12 固定依赖版本 |
| [requirements.txt](../requirements.txt) | 项目依赖声明 |
| [safety/experiment_safety.py](../safety/experiment_safety.py) | 运动必需的物理限值及数值校验；没有签认状态 |
| [scripts/acquire_robot_data.py](../scripts/acquire_robot_data.py) | 只观测机器人状态和力；复用独立力查询进程，联动假腿故障退出 |
| [scripts/capture_start_anchor.py](../scripts/capture_start_anchor.py) | 读取现场机器人起点，不发送轨迹 |
| [scripts/diagnose_collision_events.py](../scripts/diagnose_collision_events.py) | 观测厂商安全回调和事件时效 |
| [scripts/diagnose_robot.py](../scripts/diagnose_robot.py) | 诊断状态、力和对齐；汇总真实会话时序并复用到本轮报告，样本只写 CSV |
| [scripts/lab_session.py](../scripts/lab_session.py) | 统一入口：检查、静止观测、准备、运动采集、离线训练；集中写本轮报告 |
| [scripts/preview_rehab_trajectory.py](../scripts/preview_rehab_trajectory.py) | 生成轨迹 CSV 和图片预览 |
| [scripts/rokae_commission.py](../scripts/rokae_commission.py) | 现场设备调试命令入口，含显式上电、复位及点动动作 |
| [scripts/rokae_probe.py](../scripts/rokae_probe.py) | 设备只读连接和状态健康检查入口 |
| [scripts/run_real_measurement_validation_analysis.py](../scripts/run_real_measurement_validation_analysis.py) | 运行静载、重复性和轨迹敏感性离线分析 |
| [scripts/run_rehab_experiment.py](../scripts/run_rehab_experiment.py) | 内部机器人执行和采集集成；统一停止与设备清理 |
| [scripts/train_robot_leg_round.py](../scripts/train_robot_leg_round.py) | 独立无硬件进程：六参数辨识、残差 GP 和 Greedy 建议 |
| [tests/test_mainline.py](../tests/test_mainline.py) | 唯一离线回归文件：采集、设备适配、物理停止、简化流程和算法 |
| [utils/assets.py](../utils/assets.py) | 读取原始数据与模型 ZIP，归档已完成的观测或运动会话 |
| [utils/clock.py](../utils/clock.py) | 全链路共用主机单调高精度时钟 |
| [utils/provenance.py](../utils/provenance.py) | 获取 Git 提交等来源信息 |

## 本地输出

results/ 下每个新会话仅保留一份 report.md 和一份 round_status.json，原始数据分别保存在 leg/、robot/。运动会话另外保留起点、预览与 dispatch.csv，训练后生成 measured.csv。历史输出另汇总在本机 results/history.zip，原目录保留；历史报告不作为当前操作说明。

.cache/data/ 是参考轨迹与模型缓存，.cache/training/ 是训练临时数据；.cache/config/ 仅供旧独立配置接口的离线兼容测试，现场主线不生成配置副本。.venv/、.tools/ 是本机运行环境。
