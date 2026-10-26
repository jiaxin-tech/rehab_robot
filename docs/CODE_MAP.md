# 实验操作与代码地图

这份文档用于快速上手和查文件。日常只需关注统一配置、两个入口和本轮报告；后面的文件表供修改代码时查阅。所有路径相对仓库根目录，表中的文件名链接可直接打开源码。详细现场要求见 [实验指南](EXPERIMENT.md)。

## 1. 最短操作路径

在 Windows PowerShell 中进入仓库。首次安装需要 Windows x64 / Python 3.12：

~~~powershell
cd D:\rehab_robot
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win-py312.lock.txt
$env:PYTHONUTF8 = '1'
~~~

| 顺序 | 做什么 | 看什么 |
| --- | --- | --- |
| 1 | 编辑 [config/experiment.json](../config/experiment.json) | 机器人、标定、坐标、采集预算及实际现场审查记录 |
| 2 | 执行下面三条离线检查 | 测试结果、配置阻塞项、证据完整性 |
| 3 | 现场完成缺项后，准备一轮 | 起点、轨迹预览、安装及负载是否匹配 |
| 4 | 现场复核后显式执行一轮 | 采集、停止、质量检查是否完成 |
| 5 | 阅读本轮 report.md 和 round_status.json | 是否有效；是否进入辨识；停止原因 |
| 6 | 完整关闭后归档 | 汇总表和证据包新增记录 |

~~~powershell
.\.venv\Scripts\python.exe -m pytest -c pytest-core.ini -q
.\.venv\Scripts\python.exe -m scripts.lab_session check
.\.venv\Scripts\python.exe -m utils.evidence verify
~~~

准备命令读取机器人当前位置并生成预览，涉及设备连接，但不发送轨迹；会话名每轮换新：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.run_robot_leg_round prepare --config config/experiment.json --session results/round_001
~~~

下面的 run 会实际运动，只在现场复核完成且全部门禁通过后执行：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.run_robot_leg_round run --session results/round_001 --confirmation 'I CONFIRM REVIEWED ROBOT LEG ROUND'
~~~

完成并关闭后归档：

~~~powershell
.\.venv\Scripts\python.exe -m utils.evidence archive-session --session results/round_001
~~~

当前默认是未放行的只采集模式。dummy_leg_capture_only 采集与分析；dummy_leg_reference 才启用算法更新。check 报告现场缺项时应先处理缺项；测试通过不等于真机放行。当前单参考候选可能返回 DOMAIN_EXHAUSTED，不能据此宣称优化有效。

## 2. 代码架构

~~~mermaid
flowchart TD
    CFG[config/experiment.json] --> LAB[scripts/lab_session.py：统一入口]
    LAB --> ROUND[scripts/run_robot_leg_round.py：一轮编排]
    ROUND --> GATE[safety + control/execution_preflight：证据与执行门禁]
    GATE --> CTRL[control：起点、轨迹、发送与停止]
    CTRL --> HW[hardware：机器人 SDK 边界]
    ROUND --> COL[collection：机器人状态、力与假腿 ADS 采集]
    HW --> COL
    COL --> LOG[results：原始流、命令与状态]
    LOG --> QA[measurement_validation：质量、同步和测量重建]
    QA --> ALG[experiment_algorithm：六参数辨识与残差 GP Greedy]
    MECH[lower_limb_sim：运动学、动力学、参考轨迹] --> CTRL
    MECH --> QA
    MECH --> ALG
    ALG --> OUT[本轮报告与模型更新建议]
    OUT --> ARCH[utils/evidence：归档到 data]
~~~

算法进程与硬件分开，模型建议不会自动成为下一条机器人命令。公共力学模型来自 external_simulation 和证据包内的资产；旧研究代码集中在证据包，主线不依赖原“康复”目录。

推荐阅读顺序：run_robot_leg_round.py → robot_leg_round.py → train_robot_leg_round.py → surrogate.py → policy.py；排查真机执行时再看 execution_preflight.py、robot_trajectory_executor.py 和 hardware。

## 3. 文件夹用途

| 文件夹 | 作用 | 日常关注程度 |
| --- | --- | --- |
| config/ | 唯一实验 JSON、配置读取与兼容常量 | 每次实验复核 |
| scripts/ | 实验入口、诊断与离线处理命令 | 主要用 lab_session 和 run_robot_leg_round |
| collection/ | 采集、数据有效性、缓存与写盘 | 排查采集时 |
| control/ | 轨迹构造、起点和受门禁约束的执行 | 改运动流程时 |
| hardware/、hardware/windows/ | 机器人连接、状态、力、运动与 Windows 驱动 | 排查设备时 |
| hardware/windows/xcoresdk/ | 厂商二进制与接口定义；保留原接口 | 更换 SDK 时 |
| safety/ | 审查记录、哈希绑定和单次放行 | 审查执行时 |
| measurement_validation/ | 采集质量、同步、标定及分析 | 看数据是否可用 |
| experiment_algorithm/ | 六参数先验、拟合、残差 GP 与候选策略 | 改算法时 |
| lower_limb_sim/ | 共享力学、参考轨迹和坐标转换 | 改模型或轨迹时 |
| lower_limb_sim/myoleg_benchmark/ | 公共 MyoLeg 模型加载器，目录名沿用旧名 | 公共先验依赖 |
| external_simulation/myoleg_supine_rehab_v1/ | 公共机械模型 XML | 公共先验依赖 |
| data/ | 一张长期结果表、一份原始证据包 | 查结果与归档 |
| tests/ | 一个统一离线测试文件 | 修改代码后运行 |
| utils/ | 时钟、日志、来源追溯与归档 | 公共支持 |
| docs/ | 操作指南和本代码地图 | 新成员入口 |

## 4. 逐文件说明

以下覆盖当前版本的全部 Git 文件，每个表的文件名相对该节目录。__init__.py 负责 Python 包边界，具体作用逐项列出。压缩包内部的历史文件通过包内 index.json 查询，不再铺开成工作树目录。

### 根目录与文档

| 文件 | 用途 |
| --- | --- |
| [.gitattributes](../.gitattributes) | 固定文本字节与二进制处理规则，避免换行变化破坏哈希 |
| [.gitignore](../.gitignore) | 排除环境、缓存和本地运行产物 |
| [.python-version](../.python-version) | 指定 Python 版本 |
| [README.md](../README.md) | 仓库主线和文档入口 |
| [pytest-core.ini](../pytest-core.ini) | 统一测试的收集与运行配置 |
| [requirements.txt](../requirements.txt) | 项目依赖声明 |
| [requirements-win-py312.lock.txt](../requirements-win-py312.lock.txt) | Windows Python 3.12 固定依赖版本 |
| [EXPERIMENT.md](EXPERIMENT.md) | 安装、现场操作、配置和数据追溯 |
| [CODE_MAP.md](CODE_MAP.md) | 本文：最短流程、架构、目录与逐文件索引 |

### config/

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../config/__init__.py) | 配置包入口 |
| [experiment.json](../config/experiment.json) | 唯一可编辑实验 JSON，集中采集、标定、安全、审查和算法入口配置 |
| [bundle.py](../config/bundle.py) | 读取 JSON 章节、解析旧路径别名，并生成有校验的缓存快照 |
| [settings.py](../config/settings.py) | SI 单位、数据结构及兼容代码常量；真实执行还需通过已审核配置 |

### collection/

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

### control/

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../control/__init__.py) | 轨迹与执行接口包入口 |
| [execution_preflight.py](../control/execution_preflight.py) | 执行前统一核验轨迹、配置、证据和连接状态 |
| [reference_commissioning.py](../control/reference_commissioning.py) | 对冻结参考按速度上限重定时，生成待审查调试轨迹 |
| [robot_trajectory_executor.py](../control/robot_trajectory_executor.py) | 调度已放行的轨迹，记录发送意图，统一处理结束与停止 |
| [start_anchor.py](../control/start_anchor.py) | 采集、存储和验证实际起点及固定 TCP 姿态 |
| [start_anchored_relative_trajectory.py](../control/start_anchored_relative_trajectory.py) | 把相对康复参考锚定到实际起点，并审计轨迹 |

### experiment_algorithm/

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../experiment_algorithm/__init__.py) | 无硬件依赖的算法包入口 |
| [config.py](../experiment_algorithm/config.py) | 算法阈值、预算和方法配置；属于研究设定 |
| [coordinate_consistent_model.py](../experiment_algorithm/coordinate_consistent_model.py) | 用公共机械先验构建设计矩阵和六参数模型预测 |
| [coordinates.py](../experiment_algorithm/coordinates.py) | MuJoCo 关节坐标、仰卧映射及虚功切向计算 |
| [coupled_model.py](../experiment_algorithm/coupled_model.py) | 固定六参数的顺序、边界和尺度 |
| [surrogate.py](../experiment_algorithm/surrogate.py) | 六参数拟合、残差 GP、参数不确定性及诊断 |
| [policy.py](../experiment_algorithm/policy.py) | 根据已观测数据选择候选、检查预算和停止；轮次入口使用冻结 Greedy 配置 |

### data/ 与 tests/

| 文件 | 用途 |
| --- | --- |
| [data/experiment_summary.csv](../data/experiment_summary.csv) | 长期指标汇总，带来源和 SHA-256 |
| [data/evidence.zip](../data/evidence.zip) | 按内容去重的原始数据、正式历史结果、协议及公共资产；内含索引 |
| [tests/test_mainline.py](../tests/test_mainline.py) | 唯一主线回归文件，覆盖采集、门禁、停止、分析、训练和归档 |

### hardware/

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../hardware/__init__.py) | 硬件边界包入口 |
| [network_probe.py](../hardware/network_probe.py) | 不加载 SDK 的控制器网络可达性检查 |
| [rokae_adapter.py](../hardware/rokae_adapter.py) | 面向采集的机器人只读适配接口 |
| [rokae_commissioning.py](../hardware/rokae_commissioning.py) | 现场调试接口，包括显式上电、复位、有界点动和停止 |
| [rokae_motion.py](../hardware/rokae_motion.py) | 实际运动 SDK 边界，供已放行的执行器调用 |
| [safety_events.py](../hardware/safety_events.py) | 带主机时间的安全事件缓存，事件过期使许可失效 |
| [wrench_session.py](../hardware/wrench_session.py) | 力查询子进程专用只读 SDK 会话 |
| [windows/__init__.py](../hardware/windows/__init__.py) | Windows 驱动包入口 |
| [windows/rokae_internal_wrench.py](../hardware/windows/rokae_internal_wrench.py) | 封装 getEndTorque 内置力查询及转换 |
| [windows/rokae_xcore.py](../hardware/windows/rokae_xcore.py) | xCoreSDK Windows 底层封装，接收实时状态并提供设备接口 |

### hardware/windows/xcoresdk/

厂商提供的二进制和接口声明。以下 .pyi 是类型及接口描述，不是额外业务实现或测试。

| 文件 | 用途 |
| --- | --- |
| [README.md](../hardware/windows/xcoresdk/README.md) | 厂商 SDK 使用说明 |
| [CHANGELOG.md](../hardware/windows/xcoresdk/CHANGELOG.md) | 厂商 SDK 版本变更 |
| [xCoreSDK.dll](../hardware/windows/xcoresdk/xCoreSDK.dll) | Windows 原生 SDK 动态库 |
| [xCoreSDK_python.cp312-win_amd64.pyd](../hardware/windows/xcoresdk/xCoreSDK_python.cp312-win_amd64.pyd) | CPython 3.12 x64 原生扩展 |
| [xCoreSDK_python/__init__.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/__init__.pyi) | SDK 顶层类型和接口 |
| [xCoreSDK_python/RtSupportedFields.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/RtSupportedFields.pyi) | 实时流支持字段 |
| [xCoreSDK_python/model.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/model.pyi) | 机器人模型接口 |
| [xCoreSDK_python/motioncontrolRT.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/motioncontrolRT.pyi) | 实时运动控制接口 |
| [xCoreSDK_python/planner.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/planner.pyi) | 轨迹规划接口 |
| [xCoreSDK_python/utility.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/utility.pyi) | SDK 工具类型与接口 |
| [xCoreSDK_python/EventInfoKey/__init__.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/__init__.pyi) | 事件信息键的命名空间 |
| [xCoreSDK_python/EventInfoKey/LogReporter.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/LogReporter.pyi) | 日志事件字段 |
| [xCoreSDK_python/EventInfoKey/MoveExecution.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/MoveExecution.pyi) | 运动执行事件字段 |
| [xCoreSDK_python/EventInfoKey/RlExecution.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/RlExecution.pyi) | SDK RlExecution 事件字段 |
| [xCoreSDK_python/EventInfoKey/Safety.pyi](../hardware/windows/xcoresdk/xCoreSDK_python/EventInfoKey/Safety.pyi) | 安全事件字段 |

### lower_limb_sim/

这里保留主线共享的力学和轨迹工具；历史二维批量实验已经退出日常流程。部分参考构造模块用于复现和审计来源，运行时优先读取冻结参考。

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../lower_limb_sim/__init__.py) | 下肢力学与轨迹工具包入口 |
| [config.py](../lower_limb_sim/config.py) | 下肢几何及模型配置类型 |
| [derivative_estimation.py](../lower_limb_sim/derivative_estimation.py) | 从实测角度和时间估计速度、加速度 |
| [dynamic_subject.py](../lower_limb_sim/dynamic_subject.py) | 动态虚拟受试者及模型验证支持 |
| [force_mapping.py](../lower_limb_sim/force_mapping.py) | 关节力矩到牵引点力的映射 |
| [formal_protocol.py](../lower_limb_sim/formal_protocol.py) | 严格读取统一配置内的正式实验协议 |
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
| [reference_release.py](../lower_limb_sim/reference_release.py) | 只读审计冻结参考的字节和发布状态 |
| [reference_trajectory_retiming.py](../lower_limb_sim/reference_trajectory_retiming.py) | 按相位重定时参考路径并作软件动力学评估 |
| [robot_coordinate_transform.py](../lower_limb_sim/robot_coordinate_transform.py) | 人体局部坐标、机器人基座和工具坐标变换 |
| [robot_trajectory_audit.py](../lower_limb_sim/robot_trajectory_audit.py) | 离线审计 TCP 轨迹连续性、数值和速度等指标 |
| [run_robot_trajectory_export.py](../lower_limb_sim/run_robot_trajectory_export.py) | 离线导出机器人轨迹文件并验证，不连接硬件 |
| [state_history_buffer.py](../lower_limb_sim/state_history_buffer.py) | 有界下肢状态历史，保留髋膝原始角度约定 |
| [trajectory_profiles.py](../lower_limb_sim/trajectory_profiles.py) | 软件验证用最小 jerk 关节轨迹 |
| [virtual_subject.py](../lower_limb_sim/virtual_subject.py) | 准静态算法验证的虚拟受试者参数 |
| [visualize_reference_retiming.py](../lower_limb_sim/visualize_reference_retiming.py) | 参考相位和重定时轨迹绘图 |
| [visualize_robot_trajectory.py](../lower_limb_sim/visualize_robot_trajectory.py) | TCP 路径、工作空间和坐标关系的离线预览 |
| [myoleg_benchmark/__init__.py](../lower_limb_sim/myoleg_benchmark/__init__.py) | 公共模型资产包入口 |
| [myoleg_benchmark/portable_model.py](../lower_limb_sim/myoleg_benchmark/portable_model.py) | 加载冻结 XML 和公共模型资产，避免外部绝对路径依赖 |

### external_simulation/myoleg_supine_rehab_v1/

| 文件 | 用途 |
| --- | --- |
| [myoleg_supine_right_v1.xml](../external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml) | 仰卧右腿公共机械模型，供六参数先验使用 |

### measurement_validation/

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../measurement_validation/__init__.py) | 实测数据离线分析包入口 |
| [analysis.py](../measurement_validation/analysis.py) | 静载、重复性、轨迹敏感性分析和特征提取 |
| [leg_episode.py](../measurement_validation/leg_episode.py) | 导入假腿 episode，应用标定并检查采样与数据质量 |
| [offline_leg.py](../measurement_validation/offline_leg.py) | 离线分段质量、稳定性检查和临时角度标定工具 |
| [robot_leg_round.py](../measurement_validation/robot_leg_round.py) | 在同一单调时钟上对齐假腿与机器人数据，重建运动和力矩端点 |

### safety/

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../safety/__init__.py) | 现场审查和执行安全包入口 |
| [experiment_safety.py](../safety/experiment_safety.py) | 严格加载已审查限位和安全参数，缺失即阻塞 |
| [empty_load_commissioning_release.py](../safety/empty_load_commissioning_release.py) | 核验空载调试的哈希绑定单周期放行 |
| [robot_trial_release.py](../safety/robot_trial_release.py) | 创建与验证绑定代码、配置和现场证据的一次试验放行 |

### scripts/

前两个 Python 入口覆盖日常主线；诊断脚本用于特定现场问题，不要求每轮逐个运行。文件名中的 test/validate/check 不代表额外测试套件，可能是硬件诊断命令，执行前应查看其参数和现场要求。

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../scripts/__init__.py) | 脚本包入口，支持 python -m 调用 |
| [lab_session.py](../scripts/lab_session.py) | 日常统一入口，组织检查、现场诊断、采集、轮次和训练动作 |
| [lab_next_visit.ps1](../scripts/lab_next_visit.ps1) | PowerShell 包装入口，转发动作到 lab_session |
| [run_robot_leg_round.py](../scripts/run_robot_leg_round.py) | 主线一轮编排：准备、确认、放行、采集、质量检查及可选训练 |
| [train_robot_leg_round.py](../scripts/train_robot_leg_round.py) | 无硬件独立进程：验证公共先验，执行六参数辨识与冻结 Greedy 更新 |
| [run_rehab_experiment.py](../scripts/run_rehab_experiment.py) | 底层 preview/acquire/execute 分流及真机执行门禁 |
| [preview_rehab_trajectory.py](../scripts/preview_rehab_trajectory.py) | 从起点和坐标配置生成待审查轨迹预览 |
| [acquire_robot_data.py](../scripts/acquire_robot_data.py) | 仅观测采集机器人状态与力，管理日志和退出 |
| [capture_leg_segment.py](../scripts/capture_leg_segment.py) | 限定时长读取 ADS 假腿数据，不发送使能或运动命令 |
| [capture_start_anchor.py](../scripts/capture_start_anchor.py) | 连接并读取一次机器人起点，生成未审核锚点 |
| [import_dummy_leg_episode.py](../scripts/import_dummy_leg_episode.py) | 离线导入假腿 CSV，检查质量和角度标定 |
| [identify_real_episode.py](../scripts/identify_real_episode.py) | 将实测五文件 episode 转换到既有辨识器，要求已审查的辨识配置 |
| [run_real_measurement_validation_analysis.py](../scripts/run_real_measurement_validation_analysis.py) | 运行静载、重复性和轨迹敏感性离线分析 |
| [prepare_robot_trial_release.py](../scripts/prepare_robot_trial_release.py) | 生成试验放行草稿，或在已有审查证据下显式批准 |
| [prepare_empty_load_commissioning_release.py](../scripts/prepare_empty_load_commissioning_release.py) | 生成有期限、哈希绑定的空载调试放行 |
| [check_empty_load_commissioning_release.py](../scripts/check_empty_load_commissioning_release.py) | 离线核验空载调试放行 |
| [prepare_lab_bundle.py](../scripts/prepare_lab_bundle.py) | 打包字节一致的冻结输入和未放行现场草稿 |
| [rokae_probe.py](../scripts/rokae_probe.py) | 设备只读连接和状态健康检查入口 |
| [rokae_commission.py](../scripts/rokae_commission.py) | 现场设备调试命令入口，含显式上电、复位及点动动作 |
| [rokae_diagnostic_common.py](../scripts/rokae_diagnostic_common.py) | 诊断脚本共享的 SDK 初始化和输出辅助 |
| [diagnose_collision_events.py](../scripts/diagnose_collision_events.py) | 观测厂商安全回调和事件时效 |
| [check_rt_state_timing.py](../scripts/check_rt_state_timing.py) | 测量实时状态到达时间和间隔 |
| [check_wrench_query_timing.py](../scripts/check_wrench_query_timing.py) | 测量直接 getEndTorque 查询耗时 |
| [check_snapshot_alignment.py](../scripts/check_snapshot_alignment.py) | 测量状态与力快照的时间偏差 |
| [audit_state_wrench_timing.py](../scripts/audit_state_wrench_timing.py) | 正式状态/力时序审计、工况比较和报告 |
| [audit_empty_load_attempts.py](../scripts/audit_empty_load_attempts.py) | 离线重建已记录空载尝试的时序 |
| [characterize_wrench_longrun.py](../scripts/characterize_wrench_longrun.py) | 现场长时力查询诊断，统计慢调用并监督子进程 |
| [prepare_wrench_stationary_validation.py](../scripts/prepare_wrench_stationary_validation.py) | 生成静止采集 A/B 对照请求，默认未授权硬件 |
| [run_wrench_stationary_validation.py](../scripts/run_wrench_stationary_validation.py) | 静止工况采集集成与比较，默认离线运行 |
| [static_validation_logging.py](../scripts/static_validation_logging.py) | 可选静态验证标签旁路记录，不改原始数据 |
| [dry_run_static_validation_logging.py](../scripts/dry_run_static_validation_logging.py) | 用合成数据离线核验标签记录链路 |
| [validate_wrench_process_isolation.py](../scripts/validate_wrench_process_isolation.py) | 离线验证诊断用力进程隔离与超时监督 |
| [validate_wrench_process_live.py](../scripts/validate_wrench_process_live.py) | 现场只读验证独立力进程 |
| [validate_wrench_process_pair.py](../scripts/validate_wrench_process_pair.py) | 诊断用实时状态进程与力进程组合验证 |
| [wrench_nonblocking_diagnostic.py](../scripts/wrench_nonblocking_diagnostic.py) | 诊断用非阻塞最新力缓存和工作线程 |
| [wrench_process_isolation.py](../scripts/wrench_process_isolation.py) | 诊断用独立状态/力进程、IPC 和监督工具 |

### utils/

| 文件 | 用途 |
| --- | --- |
| [__init__.py](../utils/__init__.py) | 工具包入口 |
| [clock.py](../utils/clock.py) | 全链路共用主机单调高精度时钟 |
| [evidence.py](../utils/evidence.py) | 证据包查询、哈希校验、按需还原及完整轮次归档 |
| [logger.py](../utils/logger.py) | 项目日志工厂，导入时不创建日志文件 |
| [provenance.py](../utils/provenance.py) | 获取 Git 提交等来源信息 |
| [signal_processing.py](../utils/signal_processing.py) | 平滑微分、低通滤波、归一化和加速度计算 |

## 5. 配置怎么找

在 experiment.json 的 sections 对象中按下面名称查找。旧命令可能仍显示 config/某章节.json；bundle.py 会将其解析为统一配置章节。

| 章节 | 用途 |
| --- | --- |
| robot_leg_round、lab_next_visit | 一轮实验参数和现场入口动作配置 |
| experiment_safety、empty_load_commissioning_safety_20261018 | 正式执行及空载调试的安全配置 |
| site_evidence_records | 汇总各项现场审查记录 |
| twincat_leg_capture、leg_joint_calibration | ADS 假腿采集与关节角标定 |
| leg_round_analysis、real_identification_config | 本轮对齐分析及既有 episode 辨识配置 |
| rehab_frame_config、reference_commissioning | 坐标变换与调试参考配置 |
| formal_experiment_manifest、python_source_scope_manifest | 正式协议及代码来源范围 |
| live_observation_acquisition_template、live_observation_acquisition_20261018、empty_load_live_observation_20261018 | 只读观测采集模板和不同工况配置 |
| static_validation_logging、static_validation_record_schema_v1 | 静态验证标签开关和记录结构 |

config/settings.py 仍有兼容模块常量，算法包也有研究默认参数；“唯一配置”指唯一可编辑实验 JSON。修改源码常量同样属于代码改动，可能使旧审查哈希失效。

## 6. 本轮输出与本地目录

results/round_001 是示例，会话实际内容随模式、成功进度和故障情况变化。先看 report.md 的结论，再查相应细节。

| 路径 | 含义 |
| --- | --- |
| results/round_001/report.md | 本轮可读报告和阻塞/失败原因 |
| results/round_001/round_status.json | 本轮机器可读状态、阶段及处理结果 |
| results/round_001/anchor.json、reviewed_anchor.json | 原始起点和现场确认后的起点 |
| results/round_001/release_draft.json、site_review.json、release_approval.json | 本轮审查草稿、现场记录和放行凭据 |
| results/round_001/trial_setup.json | 需要绑定负载安装时的实际装置记录 |
| results/round_001/leg/leg.csv | 假腿原始采集 |
| results/round_001/robot/robot_state.csv | 机器人运动状态流 |
| results/round_001/robot/robot_wrench.csv | 机器人内置力原始流 |
| results/round_001/robot/trajectory_command.csv | 机器人侧轨迹命令记录 |
| results/round_001/robot/aligned_snapshot.csv | 带时间和有效性判定的组合快照 |
| results/round_001/robot/metadata.json | 采集配置、状态、行数和关闭情况 |
| results/round_001/dispatch.csv | 本轮轨迹发送记录及对齐依据 |
| results/round_001/quality.json | 假腿采样和数据质量检查 |
| results/round_001/measured.csv、measured_metrics.json | 对齐重建后的测量表和统计指标 |
| results/round_001/algorithm_initialization_request.json、algorithm_ready.json | 开启算法时的初始化请求及就绪结果 |
| results/round_001/training_request.json、model_update.json | 开启训练时的输入绑定及模型更新结果 |
| results/commissioning_release_uses/ | 单次许可消费记录，必须保留 |
| results/、logs/ | 当前运行产生的会话和日志，不作为散落历史报告提交 |
| .cache/ | 可重新生成的配置快照、证据还原和检查产物 |
| .venv/、.tools/ | 本机 Python 环境及辅助工具 |
| .git/ | Git 历史与仓库元数据 |

汇总时由归档命令统一追加 data 中的两份文件，原始字节仍可按证据索引找回。归档不会代替正在运行的日志流，也不会删除单次许可消费记录。
