# 真机实验操作手册

更新时间：2026-11-06

所有命令在仓库根目录的 PowerShell 中执行。真机实验必须由现场操作员执行，运动区域清空，急停可用，机器人状态和控制器报警可见。

## 1. 当前控制链路

当前使用仓库内置的 ROKAE xCoreSDK 0.7.0 Windows 运行时。轨迹先经过起始锚点、安全审核和 TCP 位姿生成，再转换成 16 元素齐次矩阵，进入实时 Cartesian schedule。SDK 通过 setControlLoopCar(callback)、startMove(cartesianPosition) 和 startLoop(False) 启动控制器 callback；callback 按时间发送目标，状态流读取实际 TCP 位姿。结束时依次执行 setFinished、等待 callback 静止、stopLoop、stopMove。

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

## 8. 停止、故障和证据

状态或力过期、碰撞事件、假腿采样过期、实时目标迟到、日志故障或 SDK 错误都应停止本轮。软件停止不能替代急停和控制器保护，不要在真机运动中主动制造断网或 SDK 故障。

正常结束顺序是：callback 返回 setFinished，等待 callback 确认和静止，调用 stopLoop，再调用 stopMove，然后退出 RT、下电并断开。

每轮保留 round_status.json 或 status.json、report.md、anchor.json、preview、dispatch.csv、robot 状态/力/清理事件、正式实验的 leg/leg.csv 和 stderr、配置副本、git commit 和现场时间。dispatch.csv 只能证明主机记录了发送动作，必须结合机器人反馈和控制器报警判断是否真的运动。

## 9. 代码验证

连接真机前运行：

~~~powershell
python -m pytest -c pytest-core.ini -q
~~~

results/rt_link_004/ 是已完成的 1 mm 实时诊断证据。代码、SDK、配置、TCP、负载或安装状态改变后，必须重新做短诊断。

