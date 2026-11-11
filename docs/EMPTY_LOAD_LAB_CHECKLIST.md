# 空载实验现场操作与故障排查

2026-11-11 更新：当前默认使用独立 TCP 牵引轨迹，假腿被动响应，不要求跟踪髋膝参考角。新流程及完整操作见[独立 TCP 牵引实验](INDEPENDENT_TCP_EXPERIMENT.md)。下文涉及 Z 单轴缩放、膝相位和髋幅值求解的内容仅适用于历史模式。

整理日期：2026-11-04。所有命令在仓库根目录的 PowerShell 中执行。

## 1. 出发前和现场准备

- 2026-11-07 的 `empty_load_117_01` 在第 24 个离散目标切换后出现 `118.38 N` 单帧力峰值并由 `force_limit_exceeded` 正确停止。根因是原 callback 对约 0.379 s 间隔的轨迹点做零阶保持，运行时形成阶梯目标；现已改为在 SDK 1 ms callback 内对相邻已准入目标连续插值平移，并拒绝姿态变化。该修改只有离线验证；真机必须新建短诊断和空载 session，不得复用失败 session，也不得提高 100 N 力限。
- 2026-11-07 的 `empty_load_117_02` 已使用连续插值运动约 14.4 s，但控制器以 `19-kInstabilityDetection` 停止。记录中 callback 端点最大迟到 16.30 ms；同时状态线程允许 `updateRobotState` 在 Python 扩展内阻塞 16 ms，可能占住 GIL 并使 1 kHz callback 连续缺包。状态接收现改为零超时 SDK 轮询，队列为空时在 Python 外短睡眠释放 GIL；状态、力和安全门禁均保留。该修复已通过离线回归但尚未真机验证，下一次必须使用新 session。
- 尚未取得当天多次空载运动的完整日志，不能确定唯一根因，也不能据旧观测记录断言没有运动。
- 当前工作区已重新启用 `hardware/rokae_motion.py` 的 SDK 回调与状态采集线程 MMCSS；`scripts/run_rehab_experiment.py` 的主执行线程优先级仍关闭。该组合用于处理 `empty_load_115` 的回调迟到和 `empty_load_116` 的状态采集饿死，尚未完成真机验证。
- 上述修改目前未提交。确认实验电脑带上这些改动；只拉远程分支不会带上未提交修改。
- 本机 `.venv` 已安装 Python 3.12.14 和锁定依赖。换实验电脑需要重新安装，不直接复制虚拟环境。
- 本机离线测试为 334 项通过、1 项失败，失败项要求配置指定的 TwinCAT DLL 存在。空载模式不启动 ADS，也不检查假腿 DLL；离线测试不能代替真机验证。

现场核对 `config/experiment.json` 的机器人 IP、本机网卡 IP、坐标配置，以及控制器中的实际 TCP、工具负载和运行模式。空载指保留当前末端工具、拆除假腿。

保留控制器保护并隔离运动区域；进入区域调试按设备规定使用手动低速和使能装置。软件速度限制不能代替控制器安全功能，启动前确认停止方式和急停可用。

## 2. 先做静态诊断

逐条运行，前一条结束后再执行下一条，避免多个程序同时占用设备连接。这些命令读取诊断，不发运动命令；连接及断开仍可能影响控制器会话。每轮换新输出目录。

```powershell
$env:PYTHONUTF8 = '1'
$robotConfig = (Get-Content -Raw -Encoding utf8 config/experiment.json | ConvertFrom-Json).robot
git rev-parse HEAD
git diff -- hardware/rokae_motion.py scripts/run_rehab_experiment.py

.\.venv\Scripts\python.exe -m scripts.diagnose_robot state --robot-ip $robotConfig.ip --local-ip $robotConfig.local_ip --duration 30 --output-dir results/diag_state_001
.\.venv\Scripts\python.exe -m scripts.diagnose_robot wrench --robot-ip $robotConfig.ip --local-ip $robotConfig.local_ip --duration 30 --output-dir results/diag_wrench_001
.\.venv\Scripts\python.exe -m scripts.diagnose_robot alignment --robot-ip $robotConfig.ip --local-ip $robotConfig.local_ip --duration 30 --output-dir results/diag_alignment_001
```

检查生成的 CSV 和摘要 JSON：是否有 `fatal_error`、有效样本数是否为零、是否持续失效；重点看状态时龄、力查询耗时、力时龄和状态/力时差的 P95、最大值。启动瞬间和稳定后的失效分开看，不能只凭平均耗时小就认定正常。

静止时若持续超时或没有健康样本，先解决采集问题。独立诊断通过只支持继续测试，不保证运动阶段正常。普通 `lab_session check` 和 `observe` 涉及假腿依赖，不能用它们的 DLL 报错判断空载不能运行。当前 CLI 不支持 `check --empty-load`；空载检查在下面的准备阶段执行。

## 3. 空载先预览，再运动

每次用新的会话目录。第一条命令读取当前起点、生成轨迹和预览，不发运动命令：

```powershell
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --empty-load --session results/empty_load_001
```

检查 `report.md`、`round_status.json` 没有阻断项；只显示 `PREPARED` 还不够，需要检查 `blockers`。核对 `preview/trajectory_preview.csv` 的整条路径方向与范围，以及 `empty_load_retiming.json` 的时长和运动限值。准备后不要移动机械臂；移动或改变 TCP、安装状态后，换新会话重新准备。

确认现场具备启动条件后执行。下面的命令会使机械臂运动：

```powershell
.\.venv\Scripts\python.exe -m scripts.lab_session run --empty-load --live --session results/empty_load_001 2>&1 |
    Tee-Object results/empty_load_001/console.log
```

当前空载独立上限为 `0.03 m/s`、`0.01 m/s²`，重定时余量为 `1.01`。参考轨迹约需两分多钟，具体以该轮重定时报告为准。普通 `motion` 中的 `1 m/s`、`2 m/s²` 不是空载实际使用的上限。

## 4. 失败后按原因排查

先记录首个失败原因、控制器报警和实际动作，再决定下一轮改动。不要反复清错重跑而丢失证据。

| 现象或报错 | 先查什么 | 参数处理 |
| --- | --- | --- |
| 起点位置、朝向不匹配 | `anchor.json`、准备后是否移动、TCP 是否一致 | 新建会话重新准备，先不扩大起点容差 |
| 工作空间超限 | 整条预览、坐标轴方向、实际可用区域 | 调整起始姿态或修正坐标配置，不直接扩大边界 |
| 状态过期、`state stale` | 原始状态帧间隔、健康检查时龄、RT 故障、网络与电脑负载 | 当前按现场确认将运动层与采集层统一为 100 ms；必须保留最大时龄记录，不能把通过视为状态链路已修复 |
| 力查询超时、`wrench stale` | 查询耗时和力时龄，区分 SDK 阻塞与主机排队 | 不继续增加 300 ms 运行宽限，先修复数据链路 |
| 命令迟到、`19-kInstabilityDetection` | 首条故障、回调耗时、日志写入、CPU 占用，检查状态 SDK 调用是否阻塞 GIL | 当前按现场决定临时使用 100 ms 主机端迟到预算；控制器仍要求约 1 ms 连续供包，不能靠该预算或提高网络容差掩盖缺包 |
| 碰撞、力或力矩超限 | 示教器报警、实际接触、工具负载与测量坐标系 | 不提高阈值硬跑，先排除真实受力及配置错误 |
| 有发送记录但机械臂不动 | 上电、模式、外部控制权限、报警与反馈位姿 | 优先查控制器状态，不先调速度 |
| 动了一段后停止 | 首条停止原因、对应状态/力记录、控制器报警 | 按首个故障定位，不只看最后的清理异常 |
| MMCSS、优先级注册失败 | 当前设置和完整异常位置 | 回调/状态线程注册失败应安全停止；保留完整日志，不绕过后继续运动 |

## 5. 参数比较原则

参数在 `config/experiment.json`。每轮只改一个变量，记录旧值、新值和现象。

| 参数 | 当前值 | 调试原则 |
| --- | --- | --- |
| `empty_load.max_tcp_speed_m_s` | 0.03 | 链路健康、可稳定启动后，可单独降低到 0.015 比较；不是已验证推荐值 |
| `empty_load.max_tcp_acceleration_m_s2` | 0.01 | 另起一轮可单独降低到 0.005 比较；不能解决通信或启动门禁 |
| `trajectory.target_peak_tcp_z_base_m` | 0.9 | 空载的基座坐标系绝对 Z 峰值，单位 m；每次改值或移动 TCP 后必须新建会话重新 `prepare`，核对完整 XYZ 预览和工作空间；普通联动禁止单轴 Z 缩放 |
| `motion.rt_filter_hz` | 50 | 先保持；不是力数据滤波，不能解除力查询阻塞 |
| `motion.rt_network_tolerance_percent` | 20 | 先保持；不是 20 ms，不控制力查询阻塞 |
| `motion.max_command_lateness_s` | 0.1 | 现场诊断临时放宽；超过 100 ms 仍拒绝下发，并结合回调和发送时序继续定位 |
| 起点位置、朝向容差 | 0.005 m、约 3° | 起点不一致先重新捕获，不扩大容差绕过 |
| 状态、力正常时龄上限 | 100 ms、50 ms | 状态门限为现场确认的空载诊断放宽值，但写在全局配置中；正式或负载运行前必须恢复或重新评审，力门限未放宽 |
| `acquisition.budgets.runtime_stale_grace_s` | 0.3 | 不再增加，当前宽限期间可能缺少有效力监测 |
| 力、力矩、工作空间及碰撞保护 | 见配置和控制器 | 不为使实验跑通而放宽或关闭 |

降低速度、加速度应分轮比较，不同时改动。这是定位手段，不是现场安全认证。不要把软件偏置或清零当作隐藏超限的方法。

修改后必须新建会话重新 `prepare --empty-load`，再检查预览与阻断项。旧会话保存了生成的 `empty_load_reference.csv` 和 `empty_load_safety.json`，修改源配置不会自动更新这些文件。

## 6. 每轮保留证据和判定结果

- 整个空载会话目录：`console.log`、`round_status.json`、`report.md`、`anchor.json`、预览、重定时文件、`dispatch.csv`（若生成）和完整 `robot/` 子目录。
- 对应的三个静态诊断目录。
- 该轮 `config/experiment.json` 副本、提交号和未提交修改的 diff；只有提交号不足以记录工作区改动。
- 示教器报警照片、现场时间、上电及运行模式、TCP/负载，以及完全没动还是动到哪里停止。

`dispatch.csv` 说明主机发送过指令，不单独证明控制器执行了运动。反馈位姿用于核对实际位移，控制器报警帮助解释停止原因，需要结合分析。

成功不只是动了：确认报告为 `EMPTY_LOAD_COMPLETE`、反馈轨迹符合预览且无未解释报警，再逐步增加重复运行。空载完成不代表假腿联动已经通过。

其他入口和结果文件说明见 [实验指南](EXPERIMENT.md)。

## 7. 实时运动链路毫米级确认

`empty_load_118` 虽然完成了 401 次目标切换并写出 `EMPTY_LOAD_COMPLETE`，但反馈 TCP 最大位移仅约 0.0011 mm，不能算机械臂实际运动成功。先用独立的小步测试确认控制器确实跟随实时笛卡尔目标，再继续完整空载轨迹。

准备新会话只读取当前位置并冻结测试轨迹，不会发送运动、上电或模式切换命令：

```powershell
.\.venv\Scripts\python.exe -m scripts.test_realtime_motion prepare `
    --session results/rt_link_001 --step-mm 1
```

人工核对工作区已清空、机械臂为空载、急停可用、基坐标系 +Z 的 1 mm 位移安全，并保持在急停旁。随后执行下列命令会启动真实运动：

```powershell
.\.venv\Scripts\python.exe -m scripts.test_realtime_motion run `
    --session results/rt_link_001 --live --confirm-unloaded `
    --operator-confirmation "I CONFIRM SUPERVISED SLOW ROBOT MOTION"
```

程序只允许基坐标系 +Z 的 0.5--2.0 mm 平滑闭合往返，默认 1 mm、总时长 11 秒。开始实时位置模式后先保持 3 秒，并从保持结束时的实测稳定位置建立跟踪基线；相对准备锚点的绝对位移和姿态限制仍持续生效。远端停留阶段必须从机器人状态反馈观察到实际位移，返回阶段必须观察到回到稳定基线；仅有回调或发送记录不会判为成功。任何失败会停止并锁定该会话，重新测试必须换新目录重新 `prepare`。
