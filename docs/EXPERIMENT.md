# 实验指南

## 安装与参数

Windows x64、Python 3.12，在仓库根目录运行：

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win-py312.lock.txt
$env:PYTHONUTF8 = '1'
~~~

只改 [experiment.json](../config/experiment.json)。机器人 SDK 已随仓库保存；leg.ads_dll_path 填实验电脑的真实 TwinCAT DLL 路径，确认 ADS 路由可读。

| 参数组 | 内容 |
| --- | --- |
| robot、leg | 网络、设备与假腿采集；observe 首先需要这些参数 |
| calibration、frame | 编码器零位、角度方向、范围和安装坐标 |
| motion、acquisition | 运动限值、采集频率、时间预算与起始角容差 |
| analysis | 训练时使用的时延、导数窗口、数据间隔和跟踪误差 |
| training | 默认关闭自动训练；候选轨迹和训练超时 |

null 是待填写参数。observe 不要求填写运动参数；run 需要完整物理限值；analysis 可在拿到数据后、训练前填写。安装或零位改变后更新标定。没有签认、发布审批或 SHA/hash 检查。

## 最短操作

**先拿联调数据**：同步读取机器人状态、力与假腿，不发送上电、模式切换或运动命令。厂商连接/断开仍可能影响控制器会话。保持现场装置静止，假腿状态与 leg.guards 一致。

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session observe --session results/observe_001 --duration 10
~~~

observe 的完成只表示观测结束；它没有轨迹发送记录，不能直接作为运动辨识回合训练。故障会中止采集并记录原因。

**运动实验**：检查参数，读取起点并查看预览，核对现场负载、安装和假腿起始角，然后显式启动。

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session check
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --session results/round_001
.\.venv\Scripts\python.exe -m scripts.lab_session run --session results/round_001 --live
.\.venv\Scripts\python.exe -m scripts.lab_session train --session results/round_001
~~~

每轮换新会话名。run 会运动；Ctrl+C、数据过期、超限或故障进入停止与清理。模型建议不自动驱动机器人。training.enabled=true 时采集后自动训练，默认先保留原始数据。

希望每轮以当前位置为起点时，使用新会话直接运行：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session run --from-current --live --session results/current_001
~~~

此入口先检查配置，再自动捕获当前 TCP 位姿、生成相对轨迹和预览，随后运动，不回固定机器人原点。第一点和闭合终点都是捕获的位置，整段保持捕获时的 TCP 朝向；路径方向仍由 frame 中的假腿安装坐标系决定，不随工具旋转。连接假腿时仍需满足参考起始角，整条平移后的路径仍须在工作空间内。捕获后机器人发生位移时，原有起点容差检查仍适用；再次运行换新会话重新捕获。

## 参数建议

以下是调试初值建议，未自动写入运动配置，也不是假腿联动的标定结论。前五项来自 physics 的10月18日空载配置；当时没有完成完整动态验证。厂商 SDK 声明建议滤波截止频率10～100 Hz，50 Hz 位于该范围中。

| 配置项 | 建议初值 | 含义 |
| --- | --- | --- |
| motion.rt_filter_hz | 50 Hz | 平滑控制器运动指令；不是力数据滤波 |
| motion.rt_network_tolerance_percent | 20% | SDK 网络包丢失容差；不是20毫秒，不能补偿力查询阻塞 |
| motion.max_start_anchor_position_error_m | 0.005 m | 捕获起点至启动时允许的位置变化 |
| motion.max_start_anchor_orientation_error_rad | 0.05236 rad，约3° | 同上，允许的朝向变化 |
| motion.max_command_lateness_s | 0.02 s | 主机轨迹命令迟到预算 |
| acquisition.max_leg_start_age_s | 0.05 s | 起始假腿读数时龄；需结合现场抖动复核 |
| acquisition.max_start_angle_error_deg | 2° | 假腿实际角度与参考起始角的容差 |

当前 max_tcp_speed_m_s=1、max_tcp_acceleration_m_s2=2 是上限，不能当作慢速试验速度建议。旧空载配置的对应上限是0.03 m/s、0.01 m/s²，可作为慢速台架调试的历史参考，需核对当前轨迹和安装。

analysis 只影响采集后的时间对齐、导数估计和训练数据筛选，不控制机器人。其建议如下：

| 配置项 | 建议初值 | 含义 |
| --- | --- | --- |
| sensor_receive_delay_s | 待测 | 假腿采样到主机接收的延迟 |
| command_application_delay_s | 待测 | 主机发送到目标实际生效的延迟 |
| max_gap_s | 0.01 s | 运动期间允许插值跨越的最大假腿采样间隔 |
| max_read_duration_s | 0.01 s | ADS 单次读取耗时上限 |
| max_dispatch_time_error_s | 0.02 s | 实际发送时序相对计划的偏差上限 |
| derivative_rate_hz | 100 Hz | 将实测角度重采样后估计导数的频率 |
| derivative_window_s | 0.15 s | 三阶 Savitzky–Golay 导数窗口；约15个样本 |
| max_start_angle_error_deg | 2° | 离线对齐后，实际与参考起始角差 |
| max_tracking_error_deg | 5° | 实际与参考角度的最大偏差；需按实验精度复核 |

两项延迟不能从采样频率或滤波频率直接算出；没有设备时间戳时，通常也不能仅凭现有主机日志唯一拆分它们。只验证训练代码流程时可以明确假设为0，正式定量分析需报告该假设或先做时延估计、敏感性分析，不能称为已标定。导数窗口建议比较0.11、0.15、0.21秒，并检查辨识参数是否稳定；原始力矩不因此平滑或清零。

## 时效处理

已实施独立力查询进程、异步查询日志、状态缓存直读，以及逐条命令 flush、结束时统一 fsync，减少主机排队、写盘对采集与发送的影响。日志在结束同步前仍可能受异常断电影响。命令写入仍有超时限制，丢失数据不会用旧值重新标时。

滤波不能恢复缺失的新样本，网络容差也不控制 getEndTorque 的阻塞。如果原生 SDK 单次调用仍停顿240～312毫秒，主机可以隔离它、检测它并停止本轮，无法保证该时间段仍获得50毫秒以内的新力数据。先按真实报告判断是查询慢、原始状态接收慢，还是快照发布慢；分别处理 SDK 调用、状态线程和写盘。不要把所有时效阈值统一改成300毫秒。训练阶段应剔除或分段处理长缺口，当前训练器会拒绝跨越超过 max_gap_s 的间隔插值。

## 结果与诊断

| 文件 | 用途 |
| --- | --- |
| report.md、round_status.json | 一份简报与一份完整状态；含失败原因、质量和训练结果 |
| leg/leg.csv、robot/ | 不同频率的原始数据分别保存 |
| anchor.json、preview/、dispatch.csv | 运动起点、预览和实际发送时刻；observe 不生成 |
| measured.csv、training.log | 对齐后的辨识数据与训练日志 |

时序问题用一个诊断命令，kind 可选 state、wrench、alignment：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.diagnose_robot state --duration 10
~~~

诊断只输出样本 CSV 和摘要 JSON，不在 JSON 中重复样本。静载、重复性和轨迹敏感性验证保留 scripts.run_real_measurement_validation_analysis，输出 validation_summary.json 和 validation_metrics.csv 两份文件。

observe、run 的现有 report.md 会直接显示机器人快照有效数、是否曾就绪、就绪后失效数，以及力查询耗时、状态时龄、力时龄和时差的 P95/最大值。详细统计并入 round_status.json 的 robot_timing；包含启动及失败样本，不另建报告。快照只反映主机观测，不能证明控制器执行时刻或样本之间绝无缺口。

完成的观测或运动会话可用 python -m utils.assets archive-session --session results/会话名 汇总到现有 data/ 数据包。原始日志保留。本机旧报告集中在 results/history.zip；不要使用历史报告中的旧命令和签认要求。

## 当前进展与下一步

截至2026年10月26日：最新真机记录是10月25日，保存在 physics 分支的 585e8064（“csv们”）提交中。精简主线已包含该提交的独立力查询进程、异步查询日志和分流 CSV 锁改动，无需重复移植。真实假腿静止采集、机器人状态和力读取已完成基础验证；仍缺完整真机轨迹、同步动态数据及实测辨识/优化结果。

10月25日最后三次观测（15:05、15:08、15:09）的力查询最大耗时分别为1.213、1.230、3.743毫秒，状态最大时龄分别为119.849、360.292、175.921毫秒，力样本最大时龄分别为50.188、53.333、60.628毫秒。每次只持续约4秒，均因 safety_event_expired_or_clock_invalid 未就绪退出，没有轨迹发送。14:33记录仍出现228.241毫秒查询，10月18日曾有约240～312毫秒查询；因此后期短片段改善不能证明阻塞完全消失或整链时效通过。

同三次 robot_state.csv 的主机接收时间戳最大间隔仅10.392、15.096、14.866毫秒；按快照时刻回查最近的原始状态帧，最大间隔约9～13毫秒。相比快照报告的120～360毫秒时龄，这支持主机消费/发布滞后是重要因素。此结果来自历史时间戳回查，不能证明在线当时已经读到最新状态，也不等于新版真机验证通过。

当前软件进一步让实时健康判断直读 SDK 最新缓存，避免状态历史写盘拖旧；控制器 RT 故障会使状态失效。默认碰撞模式改为 controller，由机械臂自身保护，不等待缺失的初始碰撞反馈或事件心跳，未知值仍记录为未知。力查询超时与数据时龄限制保留，不通过放宽时间或重复旧样本掩盖停顿。这些新增修改只有离线回归，尚未真机复测。

当前 check 仍需填写起点、命令时限、RT 设置等物理参数，并提供实验电脑实际 ADS DLL 路径。旧假腿静止采集约487～488 Hz、最大接收间隔约39毫秒，不等于稳定无缺口的500 Hz。

下一次依次完成：短时同步观测 → 慢速空载轨迹 → 假腿联动与重复采集 → 时间对齐、辨识与候选比较。单参考自身归一化为1、单候选 DOMAIN_EXHAUSTED 和离线测试通过，均不能作为优化有效的结论。

修改代码后的统一验证：python -m pytest -c pytest-core.ini -q。架构和文件职责见[代码地图](CODE_MAP.md)。

## 下次现场验证

运动前仍需填写7项：motion.max_start_anchor_position_error_m、max_start_anchor_orientation_error_rad、max_command_lateness_s、rt_filter_hz、rt_network_tolerance_percent，以及 acquisition.max_leg_start_age_s、max_start_angle_error_deg。leg.ads_dll_path 必须指向现场存在的 DLL；网络本地 IP、ADS 路由也要对应实验电脑。不能从离线测试推定这些物理参数。

先完成不运动的10秒同步观测，再在相同配置下持续60秒；稳定后可延长至180秒，均使用新会话名。60秒示例：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session observe --session results/observe_060s --duration 60
~~~

核对状态确实为 OBSERVATION_COMPLETE、robot_timing.ever_ready=true、invalid_after_ready=0，并查看是否接近25毫秒状态时龄、50毫秒力时龄/查询预算、40毫秒状态力时差上限。启动阶段也保留统计；短时通过不能证明长时间不阻塞。旧14:33记录仍有228毫秒力查询，故不能只复测几秒便宣称解决。

随后依次验证一轮慢速空载轨迹、假腿联动轨迹和重复采集，检查真实 dispatch.csv 覆盖完整参考、假腿数据覆盖运动前后，Ctrl+C 后确实停止且连接清理完成。断流、SDK 263、控制器故障等异常先用离线故障注入回归；不要在运动中主动制造通信故障。

现场常见阻断包括：控制器初始状态不是 IDLE、RT 启动/网络错误、起点离开预览位置、假腿不在参考起始角、编码器零位改变、假腿使能状态不符合 leg.guards、ADS toggle 不更新、数据过期和磁盘写入停顿。反馈未知本身在 controller 模式下不阻断；实际 RT 错误和数据超时仍结束本轮。

拿到动态数据后再填9项 analysis 参数并 train；缺完整发送记录、长数据缺口、错误时延或导数设置仍会阻止辨识。当前只有一个参考候选，需要有可比较的真实重复轮次和候选轨迹，才能汇报优化收益。

导师汇报可用：“目前完成机器人与假腿采集链路整合，并将力查询放入独立进程，减少阻塞对主链的影响。10月25日后期短片段中力查询最大约1.2～3.7毫秒，但状态最大时龄仍达360毫秒，尚未完成完整动态实验。软件已进一步分离实时状态与写盘，碰撞反馈缺失交由控制器自身保护。下一步先做长时间同步观测，再完成空载及假腿动态重复采集，最后进行六参数辨识和轨迹比较；目前不把离线测试或单候选结果当作优化有效的证据。”
