# 下次实验室任务清单

编写日期：2026-10-16。目标是补齐真实采集和标定，条件满足后完成一次参考轨迹采集及训练。不要把“本次必须运动”作为完成标准；带回有效的膝标定和稳定采集数据，同样是关键进展。

## 出发前

- [ ] 带上测角工具、固定工具、标记笔、可备份数据的存储设备；明确测角工具装在哪里、角度零点是什么。
- [ ] 实验电脑准备完整的 `D:\rehab_robot`、项目Python环境、厂家SDK、TwinCAT ADS运行库和算法文件。主配置还引用 `C:\Users\CSR001279\Documents\ChatGPT\康复` 下的算法目录；换电脑必须复制这些真实文件并修正路径，不能换成简化算法。保留内容哈希核对。
- [ ] 打开本清单及 `docs/ROBOT_LEG_ROUND.md`。所有命令在 `D:\rehab_robot` 的PowerShell执行。

推荐用PowerShell 7（终端命令 `pwsh`），下面脚本已在PowerShell 7验证。本机Windows自带PowerShell会阻止未签名脚本；遇到这个情况使用PowerShell 7或文末Python入口，不改系统执行策略。如果实验电脑没有pwsh，直接用Python入口。

```powershell
Set-Location D:\rehab_robot
.\scripts\lab_visit.ps1 -Action Check
```

默认只检查软件依赖与配置，不连接设备。当前未填配置时会报阻断并留下 `results/lab_visit/readiness_时间.json`，这是待办列表，不是需要绕过的报错。若电脑阻止脚本执行，可使用下文对应的Python入口；不需要修改系统执行策略。

## 1. 先确认现场是什么状态

- [ ] 拍照记录假腿安装、髋膝零角定义、测角工具安装、绑带、机器人与假腿连接位置；记录安装编号、工具及工件名称、实际负载和接触状态。
- [ ] 按现有现场规程确认控制器保护和停止方式；操作员能立即停止。此脚本不替你上电、使能、松刹车或复位。
- [ ] 核对机器人IP、本机网卡、机器人类和身份。历史IP为 `192.168.50.103` / `192.168.50.209`，历史SDK为0.7.0；以现场读取为准。
- [ ] 核对假腿AMS路由及端口。历史为 `10.20.4.89.1.1:11`，过程映像1366字节，四轴偏移0/23/46/69，轴1髋、轴4膝，toggle在1365字节bit4。必须对照当前TwinCAT工程，不能只凭历史记录确认。
- [ ] 在 `config/twincat_leg_capture.json` 填真实DLL绝对路径、采集时间预算、允许使能状态和经评审的力矩限值；完成映射核对后才登记reviewed。采集入口只要求这份采集配置，不要求先填完机器人运动配置。

应带回：安装照片、身份和版本、当前TwinCAT映射、DLL路径和位数、配置来源记录。

## 2. 验证真实采集，先不动机械臂

保持同一姿态、同一负载。先做10秒，再做30秒；稳定后做240秒观察长间隔及首尾变化。以下时间是记录长度，不是设备安全阈值。

```powershell
.\scripts\lab_visit.ps1 -Action Capture -Label static_10s -DurationSeconds 10
.\scripts\lab_visit.ps1 -Action Capture -Label static_30s -DurationSeconds 30
.\scripts\lab_visit.ps1 -Action Capture -Label stability_240s -DurationSeconds 240
```

- [ ] 每次检查终端给出的目录中的 `segment_status.json` 和 `quality.json`；必须是 `CAPTURE_COMPLETE`，确认CSV已关闭。
- [ ] 看实际频率、长间隔、重复时间、缺失、故障及错误码。记录最坏间隔和读耗时，不只记平均频率；500Hz名义周期是2毫秒，toggle不能证明精确丢帧数。
- [ ] 对相同静止姿态的力矩首尾变化作记录；不自动去零。运动或负载变化造成的力矩变化不能直接当漂移。
- [ ] 若失败，保留CSV、状态JSON和 `capture_stderr.log`，先定位原因；不进入机器人运动。

应带回：三次原始CSV及报告。中断/失败记录也保留，不改名覆盖成功记录。

## 3. 优先补膝标定，再验证髋

仅在现场规程允许的人工调整方式和已确认范围内调整假腿。不要根据本清单强行摆到某个角度。每个关节至少采两个不同角度用于拟合，再采一个未参与拟合的独立验证角度，并回到前一个实测角度检查往返差异。

- [ ] 膝：选可安全保持的A、B两个角度，独立测量实际角度并分别记录；另选C作独立验证；回到A重测实际角度。A/B/C只是标签，不是角度数值。
- [ ] 髋：采一个不同于旧拟合点的独立角度，并采一个回程点；旧公式仍是暂定，不能把旧拟合点的误差当独立精度。
- [ ] 每个姿态稳定后单独录10秒；在记录本写实际角度、测量方法、姿态/负载、方向、文件目录、有效静止区间。不要把移动过程算进保持段。

```powershell
.\scripts\lab_visit.ps1 -Action Capture -Label knee_A -DurationSeconds 10
# 人工调整、稳定并测角后，再运行下一条
.\scripts\lab_visit.ps1 -Action Capture -Label knee_B -DurationSeconds 10
.\scripts\lab_visit.ps1 -Action Capture -Label knee_C_validation -DurationSeconds 10
.\scripts\lab_visit.ps1 -Action Capture -Label knee_A_return -DurationSeconds 10
.\scripts\lab_visit.ps1 -Action Capture -Label hip_validation -DurationSeconds 10
.\scripts\lab_visit.ps1 -Action Capture -Label hip_return -DurationSeconds 10
```

为本次数据新建 `points.json`，格式见 `measurement_validation/OFFLINE_WORKFLOW.md`。每个file填写相对于Data目录的真实路径，例如 `knee_A_时间/leg.csv`；actual angle填写独立实测值，role分别用fit/validation/return。没有测得回程角度就填null，不假定回到了零。

```powershell
.\scripts\lab_visit.ps1 -Action Calibrate -Data results/lab_visit -Manifest results/lab_visit/points.json
```

应带回：至少膝4段、髋2段原始CSV、角度记录、points.json及计算报告。计算工具不会自动改写运动标定文件，也不会自动宣布通过。

## 4. 补齐运动与训练必须知道的信息

- [ ] 将经独立验证的髋膝公式、允许角度范围填入 `leg_joint_calibration.json`；确认3C1A关节力矩对应逆动力学的正负号，不能仅凭编码器方向猜。
- [ ] 按已有现场调试流程验证厂家安全事件的初始有效状态、断连失效和故障恢复后状态刷新；恢复不能自动续跑。不要求人为制造碰撞，未验证的项目保留未通过。
- [ ] 验证状态与力采集并行稳定性，填真实wrench频率及预算。历史263错误、SDK长阻塞不能用离线测试通过代替现场验证。
- [ ] 测量接收与发送时刻相对于设备动作/反馈的延迟，填写 `leg_round_analysis.json`；若没有可靠测量方法，记录这个缺口，不填0凑数。同主机时间戳不代表硬件同步。
- [ ] 填机器人安全、床面坐标、工具、负载、工作空间及现场评审证据；主配置中记录负责人、操作员、安装编号、过期时间和证据哈希。
- [ ] 确认当前参考整段角度适合实际安装：髋约28.91至112.03度，膝约18.32至124.79度。不适合则停止此参考的执行，另行修订并评审轨迹，不能扩大限值凑通过。
- [ ] 再运行Check，逐条消除真实缺项。不要批量把reviewed或passed改成true。

七类证据：控制器安全、采集稳定性、力语义、安装与停止、假腿安装、关节角标定、跨设备时序。证据文件记录实际观察与测量、人员和时间；哈希只绑定内容，不代替签字身份验证。

## 5. 只有前面满足后，完成一轮

先确认假腿处于参考起点（髋约29.00度、膝约18.35度，仅作辨认，实际容差使用评审配置），再读取机器人当前位姿。不要认为任意机器人当前位置都能当合法假腿起点。

```powershell
.\scripts\lab_visit.ps1 -Action Prepare -Session results/robot_leg_round/site_round_001
```

- [ ] 查看本轮 `anchor.json`、`preview/`、`report.md`，确认实际位置、方向、完整路径、连接及停止条件。Prepare只读机器人，不发运动。
- [ ] 若需要先空载联调，使用另外一份评审过的empty_load配置和新目录；空载不训练，不算假腿参考基线。
- [ ] 现场确认完成后，下面命令才会实际执行一次轨迹：

```powershell
.\scripts\lab_visit.ps1 -Action Run -Session results/robot_leg_round/site_round_001 -Confirmation 'I CONFIRM REVIEWED ROBOT LEG ROUND'
```

程序依次初始化真实算法、启动采集、执行运动、关闭CSV、校验并训练、出报告。同一目录只尝试一次；失败后保留原目录，重新评估并Prepare新目录，不删除锁文件重跑。

- [ ] 查看 `round_status.json` 是否ROUND_COMPLETE，训练状态是否MODEL_UPDATED；若失败，记录具体原因，不把部分数据当训练成功。
- [ ] 查看 `report.md`、原始CSV、quality、measured、training_request、model_update和日志。首轮E3/E2/峰值比都是自身归一化的1，不能写成有效改善。
- [ ] 目前只有一个参考候选，DOMAIN_EXHAUSTED是候选表用完，不代表优化完成；本次不自动继续另一条轨迹。

## 离开前必须做

- [ ] 备份 `results/lab_visit`、本轮 `results/robot_leg_round`、实际使用的配置、照片和现场记录；保留原始CSV不改列名。
- [ ] 交接写清：本次完成哪一步、卡在哪里、错误码及发生时间、设备最终状态、下次最先做什么。
- [ ] 即使未运动，也把新增标定和稳定性记录带回，后续可离线完成误差、往返差异与质量分析。

## Python备用命令

在 `D:\rehab_robot` 执行，下列目录标签每次使用新的名字：

```powershell
# 离线检查，不连接设备；配置未补齐会以退出码1列出缺项
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_robot_leg_round check --config config/robot_leg_round.json

# 定时只读采集，不使能或运动；需先完成真实ADS采集配置
.\.venv\Scripts\python.exe -X utf8 -m scripts.capture_leg_segment --config config/twincat_leg_capture.json --output-dir results/lab_visit/knee_A_001 --duration-s 10

# 单段质量报告
.\.venv\Scripts\python.exe -X utf8 -m measurement_validation.offline_leg quality results/lab_visit/knee_A_001/leg.csv --output-dir results/lab_visit/quality_knee_A_001

# 新角度点计算，不自动发布运动配置
.\.venv\Scripts\python.exe -X utf8 -m measurement_validation.offline_leg calibrate results/lab_visit/points.json --data-dir results/lab_visit --output results/lab_visit/calibration_001.json

# 以下两步必须满足前文的现场条件，第二条会实际运动
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_robot_leg_round prepare --config config/robot_leg_round.json --session results/robot_leg_round/site_round_001
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_robot_leg_round run --session results/robot_leg_round/site_round_001 --confirmation 'I CONFIRM REVIEWED ROBOT LEG ROUND'
```

## 每个角度点的记录表

| 文件目录 | 关节 | 实测角度/度 | fit/validation/return | 前进/回程 | 静止区间/秒 | 测角方式、零点和负载 |
| --- | --- | --- | --- | --- | --- | --- |
|  | 膝A |  | fit |  |  |  |
|  | 膝B |  | fit |  |  |  |
|  | 膝C |  | validation |  |  |  |
|  | 膝A回程 |  | return |  |  |  |
|  | 髋独立验证 |  | validation |  |  |  |
|  | 髋回程 |  | return |  |  |  |
