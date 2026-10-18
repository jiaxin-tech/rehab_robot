# 下次实验室任务清单

## 2026-10-17 安全参数声明处理

已将 jacey 提交的安装 `01` 真机动态运动授权声明归档到 `site_evidence/REAL_ROBOT_SAFETY_DECLARATION_2026-10-17.json`，并在现场签认表中登记。当前主配置为带假腿只采集、不训练模式；髋膝标定、15–145°软件守卫、工作空间、床面方向、力矩符号和机器可校验有效期 `2027-10-31T23:59:59+08:00` 已录入。2026-10-18只读复核确认机器人`192.168.50.103`恢复响应、TCP 6666/7777开放且SDK可读状态/位姿/关节/力数据。临时调试负载按控制器读回值登记为1.085 kg、质心`[0,0,0.100] m`、惯量`[0,0,0] kg·m²`；零惯量仅表示控制器现状，不是物理验证。SDK碰撞查询和事件初始状态均无效，只允许在专用单命令、人工监督调试中临时跳过，不放开自动参考轨迹。此前最长39.1875 ms的ADS主机接收间隔仍未通过稳定性证据。

编写日期：2026-10-16。当前目标已调整为：补齐真实运动门禁后完成一次带假腿参考轨迹采集，不训练。主配置使用 `dummy_leg_capture_only`；保留全部运动安全、采集质量、角度守卫和时序检查，只移除算法初始化与训练步骤。

## 2026-10-16 本机执行结果

已完成项目环境恢复、仓库路径修正、参考数据哈希修复、ADS DLL 核对、TwinCAT 文件级映射取证、机器人网络及只读 SDK 诊断。通信恢复并完成 50 Hz 力/力矩时序验证后，最新检查仍有 48 个阻断，真实采集和运动保持锁止；详情见 `results/lab_visit/2026-10-16_handover.md`、`results/lab_visit/readiness_20261016_after_comms.json` 和 `results/lab_visit/algorithm_dependency_audit_20261016.md`。

机器人通信已经恢复：xCoreSDK 0.7.0 实际使用 TCP 6666/7777，并会在会话中建立 1337 通道；旧的 8055 是未经厂家文档确认的探测假设，现已从默认探测中移除。当前采集模式不要求算法文件。仍必须由现场补齐 TwinCAT 符号语义、物理安全限值、运动范围、时序和人工签认；不得用猜测值或把 `reviewed/passed` 直接改为 `true` 来消除这些门禁。

## 出发前

- [ ] 带上测角工具、固定工具、标记笔、可备份数据的存储设备；明确测角工具装在哪里、角度零点是什么。
- [ ] 实验电脑准备完整项目、Python环境、厂家SDK和TwinCAT ADS运行库。当前 `dummy_leg_capture_only` 模式不读取算法目录；以后恢复训练前再找回并冻结真实算法文件，不能换成简化算法。
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

- [x] 每次检查终端给出的目录中的 `segment_status.json` 和 `quality.json`；三段均为 `CAPTURE_COMPLETE`，CSV已关闭。
- [x] 已检查实际频率、长间隔、重复时间、主机缺口估算、故障及错误码。实际频率486.759–488.201 Hz，最坏间隔39.1875 ms，最大读取35.0398 ms，重复时间0；主机缺口估算不作为设备确切丢帧数。
- [x] 已记录相同静止姿态的力矩首尾变化且未自动去零；240秒髋为+0.00458 N·m，膝为+0.01418 N·m。
- [ ] 若失败，保留CSV、状态JSON和 `capture_stderr.log`，先定位原因；不进入机器人运动。

已保存三次原始CSV及报告，汇总见 `site_evidence/acquisition_stability.md`。采集链路可用，但因三段均有超过4 ms的主机接收间隔，本次不把 `acquisition_stability` 签为通过。

## 3. 优先补膝标定，再验证髋

仅在现场规程允许的人工调整方式和已确认范围内调整假腿。不要根据本清单强行摆到某个角度。每个关节至少采两个不同角度用于拟合，再采一个未参与拟合的独立验证角度，并回到前一个实测角度检查往返差异。

- [x] 膝：A=30°、B=60°、C=0°、A-return=30°均已独立测量和采集；C独立误差+4.439°，同角度往返差-1.752°，因此数据任务完成但标定未通过。
- [x] 髋：独立10°点及10°回程点已采集；旧公式独立误差+4.651°，同角度往返差约-0.0010°，因此历史零位/偏置不适用于当前安装。
- [x] 六个姿态均单独录制10秒并登记测角方法、方向、文件目录和有效静止区间；移动过程未纳入标定区间。

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

已保存膝4段、髋2段原始CSV、角度记录、`points_20261017.json` 及 `calibration_knee_hip_complete_20261017.json`。安全员最终确认最大绝对误差15°后，髋独立误差4.651°通过；膝采用后续九点公式复核。

补充完成膝 `0→15→30→45→60→45→30→15→0°` 双向扫描，九个点均取得合格静止子区间。冻结单一直线公式后，用未参与该拟合的四次既有观测复核，最大绝对误差12.217°，满足安全员最终确认的15°标准；膝公式已写入正式标定，角度换算验证通过。该结论不批准运动范围、力矩符号或机器人运动。

## 4. 补齐真实运动采集必须知道的信息

- [x] 髋膝公式及独立验证状态已填入 `leg_joint_calibration.json`；物理允许范围0–160°按15°测量不确定性收窄为15–145°软件守卫；髋/膝3C1A到屈曲为正的模型符号分别确认为-1/+1。
- [ ] 控制器硬件继续独立执行碰撞保护；按已有现场调试流程验证厂家安全事件或等效的可观测失效停机信号、断连失效和故障恢复后状态刷新，恢复不能自动续跑。不要求人为制造碰撞。当前SDK碰撞状态查询无效，不能把“交给硬件”解释为取消上层失效停机门禁。
- [ ] 验证状态与力采集并行稳定性，填真实wrench频率及预算。历史263错误、SDK长阻塞不能用离线测试通过代替现场验证。
- [ ] 测量接收与发送时刻相对于设备动作/反馈的延迟，填写 `leg_round_analysis.json`；若没有可靠测量方法，记录这个缺口，不填0凑数。同主机时间戳不代表硬件同步。
- [ ] 临时调试负载已按SDK读回登记为1.085 kg、质心`[0,0,0.100] m`、惯量`[0,0,0] kg·m²`；质量和质心可读，但零惯量尚无物理依据，活动HMI工具/工件身份也未被SDK确认。该组值只用于人工监督的单步调试，不构成自动参考轨迹批准。
- [x] 按15°标定误差留足边界裕量：最新声明的髋膝物理允许范围均为0至160°，软件守卫均为15至145°；当前参考髋约28.91至112.03°、膝约18.32至124.79°，整段落在守卫范围内。参考轨迹数值本身未被改写。
- [ ] 再运行Check，逐条消除真实缺项。不要批量把reviewed或passed改成true。

七类证据：控制器安全、采集稳定性、力语义、安装与停止、假腿安装、关节角标定、跨设备时序。证据文件记录实际观察与测量、人员和时间；哈希只绑定内容，不代替签字身份验证。

## 5. 只有前面满足后，完成一轮

### 正式参考轨迹临时调试模式

已增加独立配置 `config/reference_commissioning.json`。它只允许假腿、无人、只采集、不训练、每个新会话一个周期；控制器速度倍率固定为3%，并把正式参考轨迹按同一几何路径均匀放慢，使计算得到的牵引点速度不超过0.03 m/s。原始24秒参考不会被改写，当前调试副本约132.34秒。SDK碰撞状态不可读只被记录为待单独批准的观测豁免；控制器碰撞保护、操作员手持急停、状态/力数据故障即停不能取消。

2026-10-18 17:00在受限命令环境中的重试曾显示TCP端口关闭、SDK错误256和0条样本；17:09后改用正常Windows网络权限复核，SDK已成功连接，确认前述结果是本机执行环境造成的假阴性，不是控制柜通信服务故障。10秒只读状态诊断取得449条样本，力查询取得484条样本；控制器负载读回为1.085 kg、质心`[0,0,0.100] m`、惯量`[0,0,0] kg·m²`。SDK碰撞主动查询仍返回259；事件订阅能够注册，但在未发生事件时不能提供可信初始“无碰撞”状态。详见 `site_evidence/reference_commissioning_gate_attempt_2026-10-18.md`。

17:15经操作员确认静止、无人接触后，会话软件力偏置成功。随后10秒取得465组状态/力快照，力查询465/465可用且最大耗时1.642 ms；完整快照仍因world到Base力方向旋转未经真机验证而保持0/465有效，另观察到31组状态/力时间差超过现有20 ms门限。偏置随诊断进程结束而失效，正式采集进程必须在相同静止条件下重新执行。原始结果为`diagnostics/snapshot_alignment_20261018T091525Z.json`。

17:17完成Base +X轻推检查：相对基线力变化为`[+22.724,+5.950,+0.246] N`，主响应轴X且符号为正，与预期一致。单一方向不足以确定完整三维旋转，`BASE_WRENCH_ROTATION_VERIFIED`暂不改写；还需完成一个不共线方向（Base +Z）检查。原始结果为`diagnostics/wrench_frame_rotation_20261018T091752Z.json`。

17:19完成Base +Z轻推检查：相对基线力变化为`[-26.855,-3.323,+63.647] N`，主响应轴Z且符号为正。结合Base +X结果，两个不共线方向均与预期一致，当前机器人/工具安装的`BASE_WRENCH_ROTATION_VERIFIED`已改为true；更换机器人、工具安装、坐标约定或SDK变换时必须撤销并重验。原始结果为`diagnostics/wrench_frame_rotation_20261018T091939Z.json`。

17:21重新偏置并复核后，方向门禁已实际解除，10秒快照409/454有效；剩余无效均为时序超限。30秒独立力查询又观测到239.844 ms和311.986 ms两次SDK长阻塞。拟议的临时调试预算为状态年龄25 ms、力年龄50 ms、状态/力skew 40 ms、单次力查询50 ms，超限即判无效并停止；该预算回放本次快照可通过452/454，同时拒绝两次长停顿。预算仍需安全员明确签认后才能写入实时采集配置。

先离线检查并生成不可执行的会话工件：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.reference_commissioning check
.\.venv\Scripts\python.exe -X utf8 -m scripts.reference_commissioning prepare --session results/reference_commissioning/site_001
```

查看 `commissioning_manifest.json`。`motion_authorized` 必须保持 `false`；它会列出正式执行链仍未满足的真实阻断。该工具不连接机器人、不上电、不发运动，不能把生成CSV直接送给控制器。只有状态/力采集预算、起点容差、命令延迟、RT参数和现场证据补齐并经独立运动发布流程绑定后，才可增加真正的单次Run入口。

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

当前程序依次启动采集、执行运动、关闭CSV、进行质量检查与时间对齐并出报告，不加载算法、不训练。同一目录只尝试一次；失败后保留原目录，重新评估并Prepare新目录，不删除锁文件重跑。

- [ ] 查看 `round_status.json` 是否为 `CAPTURE_ONLY_COMPLETE`，训练状态是否为 `SKIPPED_CAPTURE_ONLY`；若失败，记录具体原因，不把部分数据当完整运动采集。
- [ ] 查看 `report.md`、原始CSV、`quality.json`、`measured.csv` 和 `measured_metrics.json`；采集模式不应生成 `training_request.json`、`model_update.json` 或训练日志。
- [ ] 本次只采集一条已评审轨迹，不进行候选选择，也不自动继续另一条轨迹。

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
