# 机械臂受力驱动 TCP 轨迹实验

2026-11-12。用户已明确优先实际康复条件：不要求患者姿态、髋中心、连接偏移或有效腿长已知。正式主线改为机械臂位姿与受力驱动的 TCP 候选决策，原六参数方法只作离线对照。

## 当前算法与实现

1. 按机器人坐标定义任务：起点、绝对 Z 峰值、允许水平位移、上抬/返回时间范围、最长周期和物理限值。
2. 独立 TCP 规划器生成轨迹，捕获当前机器人起点并固定计划朝向；不读取关节参考 CSV、假腿传感器或患者几何。
3. ROKAE xCoreSDK 实时笛卡尔位置控制执行一轮，同步记录实际 TCP 和六维受力。实时力控制不在本实现中，在线已有力限仍负责停止保护。
4. 完整执行和数据审核通过后，直接计算力/力矩负担及峰值，不重建患者角度或关节力矩。
5. 复用原 MaternGP、expected_improvement、constraint_probability 工具，预测有限 TCP 候选的负担及峰值，选择下一未试候选；预算耗尽、无候选、无预测可行候选或无预期改善时停止。
6. 下一轮重新准备、核对预览，再显式启动。每次重复运动也保存观测并训练，候选 ID 不伪造。

这是经用户确认后的新正式观测/决策方法，不能写成原六参数算法原封不动部署，也不再把受力目标叫作原 E3/E2。

## 目标与约束

每个上抬、返回分支分别计算力向量模长的时间 RMS 和力矩向量模长的时间 RMS，两个分支等权平均得到 F_rms、T_rms。它们使用机械臂实测六维力，先减去已有依据的无接触空载偏置。

~~~text
load_score = 0.5 × [F_rms / motion.max_force_n + T_rms / motion.max_torque_nm]
force_peak_fraction = 全周期最大力模长 / motion.max_force_n
torque_peak_fraction = 全周期最大力矩模长 / motion.max_torque_nm
~~~

目标是降低 load_score，预测峰值分数约束均为不超过 1。这里的归一化和等权目标是明确的工程设计，不是临床疗效指标或患者关节力矩。力矩使用固定 SDK 坐标和参考点的记录，不能称为关节力矩或纯绑带张力。

GP 使用三个真实 TCP 特征：水平位移、实际上抬时间、实际返回时间。候选若经保守限速延长时间，冻结和训练使用延长后的真实特征，避免把原计划时间当成实际时间。GP 以首轮结果作为公共常数基线，学习直接受力目标相对该基线的变化，不含六参数物理先验。

GP 长度尺度、噪声与幅值下限、预测可行性分数和停止门限见 force_training 配置，均为待实验核对的工程设定。模型不描述患者姿态变化、跨轮漂移和重复观测的统计依赖，预测约束分数不等于临床安全概率。

## 防止通过少运动获得低分

所有候选保持同一绝对 Z 峰值、固定保持时间和采样设置，水平位移只能在指定区间内变化，上抬/返回时间有上下限，周期有上限。还要求最小垂直行程和闭合返回。

每次训练先检查实际机器人反馈与完整计划是否一致，命令日志存在或只达到低受力都不代表任务完成。反馈没有完成行程，即使受力接近零也拒绝训练。缺失、失效或超预算的力数据不会作为零力或低负担使用。

这些检查证明的是机器人任务执行；在没有患者姿态测量时，不能据此证明人体关节活动范围、有效牵引或康复效果已达到。后续实验应单独验证实际作用效果。

## 当前默认候选

| 候选 | 水平位移 | 上抬 | 返回 | 起点/最高点保持 |
| --- | --- | --- | --- | --- |
| tcp_reference | -0.2 m | 80 s | 80 s | 各 2 s |
| tcp_slower_lift | -0.2 m | 100 s | 80 s | 各 2 s |
| tcp_slower_return | -0.2 m | 80 s | 100 s | 各 2 s |

目标 Z 沿用现有配置 0.9 m，总周期分别为 164/184/184 s，若限速触发延长则仍须满足任务预算。默认先比较运动节奏，空间端点一致。接口支持改变水平位移，但必须同时明确允许的水平任务区间；程序不自行扩大空间范围或生成无限新候选。

这些候选是待验证的实验计划，不是已验证适合某患者的处方。当前机器人运动方向由 frame 决定，其坐标是实验任务坐标，不要求精确的患者解剖坐标。

## 必需配置

默认 trajectory.mode=independent_tcp，training.mode=tcp_force。正式流程不读取 robot_observation 中旧几何标定，也不读取原 training.candidates 的名义关节 CSV；旧文件保留给离线对照。

| 配置 | 作用 |
| --- | --- |
| trajectory | 峰值、保持和采样等 TCP 任务定义 |
| force_training.candidates | 有限候选目录和三个可变特征 |
| force_training 的位移/时间/周期/最小行程预算 | 保证候选仍执行规定机器人任务 |
| motion、empty_load | 仍使用两组速度/加速度限值的较小值，保留力、力矩、空间、时效与停止保护 |
| force_training.wrench_frame_type | 用于比较的固定机器人受力表达坐标 |
| force_training.unloaded_wrench_bias | 六个有实测依据的无接触偏置值 |
| force_training.compensation_verified | 工具/负载与 SDK 外力补偿条件是否已核实 |
| validation.capture_dummy_leg | true 为台架采集/保护；false 为仅机器人流程，不依赖 ADS/编码器/假腿标定 |

不要求患者几何，不代表可以省略机器人 TCP/工具、工作空间、外力坐标/补偿和数据质量核对。当前空载偏置为 null、补偿条件为 false，正式带载学习会被阻止；不自动填零，不在腿受载时清零来抵消牵引载荷。空载执行不进入训练。

## 操作指南

在仓库根目录 PowerShell 执行。实验台架保持 validation.capture_dummy_leg=true，可采集假腿独立验证数据并保留装置保护；仅机器人工作流设为 false。每轮用新目录。

先核对配置与不运动观测：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session check --config config/experiment.json
.\.venv\Scripts\python.exe -m scripts.lab_session observe --config config/experiment.json --session results/force_observe_01 --duration 30
~~~

完成必要的短运动诊断后，先做空载：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --config config/experiment.json --empty-load --session results/force_empty_01
.\.venv\Scripts\python.exe -m scripts.lab_session run --empty-load --live --session results/force_empty_01
.\.venv\Scripts\python.exe -m scripts.lab_session complete --session results/force_empty_01
~~~

prepare 读取起点、不运动。检查 blockers 为空，以及 preview/、tcp_plan.json、tcp_safety.json 和 tcp_force_binding.json 的整条路径、速度、加速度、时长与现场空间。配置或安装改变后重新准备，不修改旧预览后执行。

空载反馈核对后，正式带载第一轮：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --config config/experiment.json --session results/force_round_01
.\.venv\Scripts\python.exe -m scripts.lab_session run --live --session results/force_round_01
~~~

run 正常结束后自动训练，检查 training_status=COMPLETE 和 tcp_force_training.json 的 action。训练失败仍保留原始记录，修复数据/分析问题后可离线重试：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session train --training-mode passive --session results/force_round_01
~~~

按模型建议准备下一轮，再核对预览并启动：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session repeat --source results/force_round_01 --trained --session results/force_round_02
.\.venv\Scripts\python.exe -m scripts.lab_session run --live --session results/force_round_02
.\.venv\Scripts\python.exe -m scripts.lab_session complete --session results/force_round_02
~~~

不带 --trained 的 repeat 保留相同候选用于重复验证，仍记录并训练。repeat --trained 在 STOP 时拒绝继续，不自行绕过停止。新的正式流程支持重复候选观测，原六参数离线分支的重复限制不改。

## 证据文件

| 文件 | 内容 |
| --- | --- |
| anchor.json、tcp_plan.json、tcp_trajectory.csv、tcp_safety.json | 机器人任务、起点、实际时间及执行限制 |
| tcp_force_binding.json | 完整候选目录、实际延时后的特征及可执行性 |
| prepared_config.json、round_status.json、report.md | 本轮配置、状态、阻断与建议 |
| dispatch.csv、robot/ | 实际命令时刻、机器人状态与六维受力 |
| tcp_force_training.json、training.log | 直接受力指标、GP 预测、决策、输入哈希和实现版本 |
| pretrial_prediction.json | 下一轮运动前冻结的预测，结束后计算预测误差 |
| leg/、passive_response.* | 可选假腿验证记录，不进入正式训练 |

模型会核对历史配置、机器人起点/工具朝向和原始证据变化。阶段结果只支持当前记录，不能据此声称患者状态恒定或因果优化收益。测试使用合成数据，不代表真机受力降低、临床安全或疗效。

原六参数源码与 worker 保留用于离线对照，不能在未知患者几何时填写猜测值来运行。论文架构见 [PAPER_ARCHITECTURE.md](PAPER_ARCHITECTURE.md)，决策依据见 [FORCE_DRIVEN_DESIGN_REVIEW.md](FORCE_DRIVEN_DESIGN_REVIEW.md)。
