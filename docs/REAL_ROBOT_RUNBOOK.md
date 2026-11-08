# 真机实验操作手册

更新时间：2026-11-07

所有命令在仓库根目录的 PowerShell 中执行。真机实验必须由现场操作员执行，运动区域清空，急停可用，机器人状态和控制器报警可见。

## 1. 当前控制链路

当前使用仓库内置的 ROKAE xCoreSDK 0.7.0 Windows 运行时。轨迹先经过起始锚点、安全审核和 TCP 位姿生成，再转换成 16 元素齐次矩阵，进入实时 Cartesian schedule。SDK 通过 setControlLoopCar(callback)、startMove(cartesianPosition) 和 startLoop(False) 启动 1 ms callback；callback 按当前轨迹时间在相邻、已持久化且已准入的目标之间连续插值平移，姿态必须固定，状态流读取实际 TCP 位姿。主线程会在前一端点到达前持久化并准入下一端点，未准入时 callback 不得向该端点插值。状态接收使用零超时 SDK 队列轮询，避免正超时的 pybind 调用占住 GIL 并饿死 1 kHz callback。结束时依次执行 setFinished、等待 callback 静止、stopLoop、stopMove。

这是 TCP 笛卡尔位置控制，不是关节角控制、普通 NRT moveAppend 控制，也不是力控制。

## 2. 安装与 SDK

要求 Windows x64、Python 3.12：

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win-py312.lock.txt
$env:PYTHONUTF8 = '1'
~~~

SDK 文件位于 hardware/windows/xcoresdk：xCoreSDK.dll、xCoreSDK_python.cp312-win_amd64.pyd 和 .pyi 类型声明。SDK 版本必须为 0.7.0；替换 SDK 后必须重新做实时链路诊断。

## 3. 配置

只修改 config/experiment.json。现场必须核对机器人 IP、本机网卡 IP、robot_class、碰撞源、TwinCAT/ADS DLL 路径、实时滤波频率、网络容忍度、运动限制和采集参数。

~~~powershell
python -m scripts.lab_session check --config config/experiment.json
~~~

出现 blocker 不得运动。check、prepare、run 会检查现场 ADS DLL；纯离线调用 check_config 可以只做结构检查。

## 4. 静态观测和准备

先进行不发送运动的观测：

~~~powershell
python -m scripts.lab_session observe --config config/experiment.json --session results/observe_YYYYMMDD_01 --duration 30
~~~

检查状态流和力流持续健康、robot_timing.ever_ready 为 true、invalid_after_ready 为 0、年龄和状态力时间差没有持续超限、没有 fatal_error、样本数不为 0。

再生成正式轨迹但不运动：

~~~powershell
python -m scripts.lab_session prepare --config config/experiment.json --session results/round_YYYYMMDD_01
~~~

检查 round_status.json 为 PREPARED 且 blockers 为空；检查 preview/trajectory_preview.csv 的起点、终点、方向、范围和 TCP 朝向。prepare 后不要移动机器人，不要改变 TCP、工具、负载或假腿安装状态；有任何变化都要新建 session 并重新 prepare。

轨迹方向由 `config/experiment.json` 的 `frame` 定义。两条配置向量表示康复坐标正 X、正 Z 在机器人基坐标系中的方向；实际运动方向还取决于预览中的 `delta_x_R/delta_z_R` 正负。当前参考主行程 `delta_x_R<0`，所以 `[0,-1,0]` 将主行程映射为基座 `Y+`。方向调整后必须核对预览中 `tcp_x_base/tcp_y_base/tcp_z_base` 的最小值、最大值和首段变化，再新建毫米级或空载会话验证。

`config/experiment.json` 的 `trajectory.target_peak_tcp_z_base_m` 是基座坐标系绝对 Z 峰值，单位为米。TCP 原点由每次 `prepare` 只读捕获。空载模式可据此冻结单轴 `rehab_z_scale`；连接假腿时禁止单独放大 Z，因为它会破坏两连杆正运动学，普通联动必须改用经过审核的几何一致关节轨迹。改峰值、移动机器人或改变 TCP 后必须新建会话并重新准备，禁止只改旧会话 CSV。假腿“当前原点”应保存为独立的实验相对原点记录，不能覆盖 `calibration.*.zero_counts` 的编码器/解剖标定。

当前普通联动从准备时实测物理髋膝角出发，在 `22.2～22.55 s` 内自动求膝轨迹周期平移量并保留完整角度行程；计划膝最低角必须比 `-10°` 物理下限高出 2° 跟踪门限及额外 0.5° 余量。随后程序在配置上限内二分求解髋幅值以达到 0.900 m。生成结果必须同时满足髋 `[-5°,145°]`、膝 `[-10°,145°]`、完整 XYZ 工作空间、速度和加速度限制；变换参数冻结在 `trajectory_adjustment.json`，预览、外层预检和调度器内层审计必须一致。

## 5. 实时链路诊断

完整实验前必须先做 0.5--2.0 mm 的独立诊断，推荐 1 mm。

准备阶段不发运动：

~~~powershell
python -m scripts.test_realtime_motion prepare --session results/rt_link_YYYYMMDD_01 --step-mm 1
~~~

人工确认空载、运动区域清空、急停可用，并确认基座 +Z 方向的 1 mm 位移安全。然后明确启动真机运动：

~~~powershell
python -m scripts.test_realtime_motion run --session results/rt_link_YYYYMMDD_01 --live --confirm-unloaded --operator-confirmation "I CONFIRM SUPERVISED SLOW ROBOT MOTION"
~~~

成功必须同时满足：status.json 为 COMPLETE；motion_commanded 为 true；stop_reason 为 trajectory_completed；commands_dispatched 大于 0；far_verified 和 return_verified 都为 true；实际 TCP 位移接近 1 mm；横向位移和姿态误差在限制内；同会话清理完成。

只有 callback 运行、发送日志存在或 COMPLETE 字样，不能单独证明机器人实际运动成功，必须检查反馈位姿。诊断失败后不要复用 session，必须新建目录并重新 prepare。

## 6. 空载实验

空载保留当前 TCP 工具和负载，拆除假腿，不启动 TwinCAT/ADS 假腿采集。先准备：

~~~powershell
python -m scripts.lab_session prepare --config config/experiment.json --empty-load --session results/empty_load_YYYYMMDD_01
~~~

检查 preview/trajectory_preview.csv、empty_load_retiming.json、empty_load_safety.json 和 blocker。确认现场条件后运动：

~~~powershell
python -m scripts.lab_session run --config config/experiment.json --empty-load --session results/empty_load_YYYYMMDD_01 --live
~~~

成功至少需要 EMPTY_LOAD_COMPLETE、完整 dispatch.csv、反馈轨迹符合预览、无未解释报警或安全事件、实时控制清理完成。空载完成不等于假腿联动通过。

## 7. 假腿正式实验

安装假腿后核对编码器零位、方向、量程、参考起始角、ADS 通信和安全状态。安装或零位变化后重新观测并 prepare。

~~~powershell
python -m scripts.lab_session prepare --config config/experiment.json --session results/round_YYYYMMDD_01
python -m scripts.lab_session run --config config/experiment.json --session results/round_YYYYMMDD_01 --live
~~~

运行时必须确认假腿采集健康、起始角在允许误差内、机器人状态/力/碰撞/日志健康、dispatch.csv 和 robot/ 存在、最终状态为 CAPTURE_COMPLETE，并且 TCP 反馈覆盖运动前中后。

若每次以当前 TCP 为起点：

~~~powershell
python -m scripts.lab_session run --config config/experiment.json --from-current --session results/current_YYYYMMDD_01 --live
~~~

该入口会新建 session、捕获当前位姿并生成相对轨迹，不要复用旧 session。

连接假腿的普通 `prepare` 会同时冻结当前轴1/轴4为该 session 的轨迹起点，证据写入 `leg_start_origin.json`。它不重写全局编码器零点；运行前同时检查全局物理角度范围和 session 轨迹起点误差。若准备后假腿被拖动，必须换新 session 重新准备，不能通过增大 `max_start_angle_error_deg` 继续。

运动过程中还会按最近已下发的轨迹时刻检查髋、膝跟踪误差。`acquisition.max_tracking_error_deg` 当前为 2°；超过后立即取消实时调度，不能用放宽物理角度范围代替跟踪误差修复。

## 8. 停止、故障和证据

状态或力过期、碰撞事件、假腿采样过期、实时目标迟到、日志故障或 SDK 错误都应停止本轮。软件停止不能替代急停和控制器保护，不要在真机运动中主动制造断网或 SDK 故障。

若出现 `19-kInstabilityDetection`，含义是控制器连续多个周期未收到新命令。`motion.max_command_lateness_s` 是主机端轨迹端点门限，不能替代控制器的 1 ms 供包要求；不要通过增大 100 N 力限或盲目提高网络容差继续运行。保留该 session，检查 `dispatch.csv` 的端点迟到、机器人状态查询耗时和系统负载，并用新 session 复测。

若一轮被现场急停、需要回到该轮保存的起始 TCP，不要发一个跨越数百毫米的直接 jog。使用 `scripts.restore_session_tcp prepare` 重新读取当前位置，要求当前位置在该轮冻结预览路径 10 mm 内，然后生成不超过 4 mm 的逆序 MoveL 分段。若当前位置不在旧路径附近、但当前位置和整条直线走廊都已确认位于无障碍的受审工作空间，可在 `prepare` 时显式增加 `--direct`，从实测位置生成通往源会话 TCP 的直线小分段；该模式不提供避障，也不会接受与目标工具朝向相差超过 0.02 rad 的起点。`run` 固定使用 10% 速度，切换 NRT 模式后会重新建立状态流；每段下发冻结的绝对路点，并且只有新状态序列确认 TCP 到位后才允许进入下一段。每段前后还会复核状态、工作空间和 100 N/200 Nm 主机力门限；控制器碰撞保护和现场急停仍必须有效。回位会话同样单次使用，prepare 后位置变化需新建会话。

正常结束顺序是：callback 返回 setFinished，等待 callback 确认和静止，调用 stopLoop，再调用 stopMove，然后退出 RT、下电并断开。

每轮保留 round_status.json 或 status.json、report.md、anchor.json、preview、dispatch.csv、robot 状态/力/清理事件、正式实验的 leg/leg.csv 和 stderr、配置副本、git commit 和现场时间。dispatch.csv 只能证明主机记录了发送动作，必须结合机器人反馈和控制器报警判断是否真的运动。

## 9. 代码验证

连接真机前运行：

~~~powershell
python -m pytest -c pytest-core.ini -q
~~~

results/rt_link_004/ 是已完成的 1 mm 实时诊断证据。代码、SDK、配置、TCP、负载或安装状态改变后，必须重新做短诊断。

