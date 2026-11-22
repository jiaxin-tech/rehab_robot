# 假腿刚性连接：本次现场操作

更新：2026-11-22。分支：`paper-personalized-load`。本页是当前带腿入口。

## 现在的结论

你已确认：髋基座固定，髋、膝关节可自由转动；机械臂与小腿刚性连接，连接点很靠近膝盖；尚未核对接上腿后的整条路径。真实转轴方向和夹具几何仍需记录。

**当前不能把90秒固定朝向的空载曲线直接用于这套连接。** 小腿转动会带着刚性连接的工具转动。固定工具朝向会增加约束，任意三维闭环即使缩小也不保证满足机构运动条件。靠近膝盖只能改变力臂，不能消除这个约束。若髋中心固定、两关节轴平行，固定小腿朝向时TCP只能沿受实际大腿几何限制的一段圆弧运动。

仓库中的0.42m/0.30m为模型常量，0.30m还是膝至等效牵引点距离，不能拿来当你近膝夹具的实测尺寸。`robot_observation`中的实际髋中心、长度、连接偏移仍为空；现有驱动层也只支持固定朝向。因此本次先取得连接后的真实完整位姿，随后再做位置与朝向联动的轨迹和执行验证。这里的运动规划需要安装信息；它不等于论文学习器必须输入人体参数。

已保存3%缩幅、90秒的**离线对照预览**：抬升约22.53mm、Base Y正向前伸约26.78mm、中段速度约1.05mm/s，首尾各5秒渐变，阶段间无停留。见`results/manual_lab/20261122_loaded_diagnostic_v1/`。它不是这套刚性连接的可执行负载轨迹。配置`config/continuous_taught_90s.loaded_diagnostic.json`如实标为`rigid_fixture`、未复核；即使勾选人工复核，程序也会阻止固定朝向负载运动。不要改连接类型、添加`--empty-load`或关闭假腿采集来绕过。

11月22日空载已完整执行90秒、4501个目标，预记录约122ms，跟随最大误差约2.975mm。但有两处242～246ms受力缺口，不能作为正式负荷比较的合格基线，详见[回传审核](lab/EMPTY_90S_REVIEW_2026-11-22.md)。新负载诊断不会接受空载使用的受力陈旧宽限。当前100N/200Nm只是继承的机器配置，未证明适合此夹具，更不是人的舒适度阈值。

## 1. 更新与建档

在实际连接机器人和TwinCAT的那台旧电脑，用已有PowerShell与虚拟环境。先查看：

```powershell
Set-Location 'C:\Users\liumai\fjx\rehab_robot'
git branch --show-current
git status --short
git diff --cached --name-only
```

路径如已迁移，以现有仓库为准。分支应为`paper-personalized-load`。未回传记录先按第5节保存；其他代码或暂存变更先核对，不覆盖、不reset/clean。妥善保存后：

```powershell
git pull --ff-only origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '更新失败，保留现场文件。' }
$loadPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $loadPython -PathType Leaf)) { throw '未找到已有Python。' }
$loadConfig = 'config/continuous_taught_90s.loaded_diagnostic.json'
$loadRoot = 'results/manual_lab/' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Path $loadRoot -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath 'docs/templates/RIGID_SHANK_SETUP_RECORD.md' -Destination (Join-Path $loadRoot 'setup_record.md') -ErrorAction Stop
Write-Host "本次目录：$loadRoot"
```

保持同一个窗口。配置沿用旧电脑`192.168.50.209`、机器人`192.168.50.103`及原ADS映射。实际网络或TCP/工具不同，记录并核对；不猜测修改网卡或驱动参数。

## 2. 当前可以先做：连接后静止采集10秒

先在原厂界面核对真实TCP、工具负载、假腿支撑、关节状态。假腿是被动装置，不用项目代码使能假腿或复位驱动。保持装置静止且无人触碰工具：

```powershell
$loadObserve = Join-Path $loadRoot 'connected_static10s'
& $loadPython -X utf8 -m scripts.lab_session observe --config $loadConfig --session $loadObserve --duration 10
$loadObserveExit = $LASTEXITCODE
Write-Host "静止采集退出码：$loadObserveExit"
Get-Content -LiteralPath (Join-Path $loadObserve 'report.md') -Raw -Encoding UTF8
```

本命令连接并读取机器人与ADS，不发轨迹、上电或模式命令。SDK连接/断开仍属于设备会话操作。缺DLL、驱动fault/非零error、意外使能、查询超时或采集失败时，保留原状态和原始文件，不跳过检查。`OBSERVATION_COMPLETE`只表示观测流程结束；同时看报告中的受力缺口、有效样本和假腿质量。未测无接触偏置、未核实补偿时保持null/false，不填零或true。

填写`setup_record.md`：髋基座固定情况、髋/膝活动方向、夹具位置、工具/TCP、支撑、实际静止受力与已核实的设备限值。近膝连接对小腿转动的观察与膝关节受力解释尤其需要这些说明。

## 3. 能手动顺畅活动后，录制真实连接轨迹

只有髋基座/支撑稳定、操作者已会使用原厂拖动，并能在关节范围内让连接机构顺畅活动时进入本节。不确定支撑、遇到卡滞/夹具受扭时先停在静止检查阶段。不要用空载曲线去试探能不能走通。

由你在原厂示教器选择已经会用的拖动功能。**允许工具随小腿自然转动**；不要为了保留原固定朝向而强扭机构。先小范围、缓慢地完成可行的抬升—前伸—回收，幅度服从实际机构；不追空载75cm/89cm行程，也不为了90秒赶动作。超出90秒可在新目录录更长原始段，后续再离线设计90秒节奏。

下面只录机器人实际六维位姿，不会帮你移动、切模式或实时保护假腿，也不同时采集ADS/受力。它与上一步静止观测依次进行，不同时打开两套机器人采集进程。

```powershell
$loadTeach = Join-Path $loadRoot 'connected_drag_01'
& $loadPython -X utf8 -m scripts.record_rehab_trajectory --config $loadConfig --session $loadTeach --duration-s 90 --live
$loadTeachExit = $LASTEXITCODE
Write-Host "录制退出码：$loadTeachExit"
Get-Content -LiteralPath (Join-Path $loadTeach 'recording_status.json') -Raw -Encoding UTF8
```

看到`RECORDING`后，起点稳定2～3秒；缓慢走可行小范围；回到起点稳定并等待结束。若连接SDK后原厂拖动不可用，保留记录并停止，不用项目命令强制切模式。需要停机器人时用原厂停止手段；Ctrl+C只中止录制。若一个人无法同时照看夹具、假腿支撑和机械臂，不进行动态录制，先回传静止数据与安装说明。

录制成功后再离线看原始轨迹与朝向变化：

```powershell
$loadPreview = Join-Path $loadRoot 'connected_drag_preview_01'
& $loadPython -X utf8 -m scripts.preview_recorded_trajectory --session $loadTeach --output-dir $loadPreview
if ($LASTEXITCODE -ne 0) { Write-Host '预览失败，原始记录仍保留并回传。' }
```

这个预览不会生成可回放轨迹。保存原始`tcp_rx/ry/rz`，不套用之前丢弃朝向变化的空载拟合。后续处理应保留实际姿态变化，核对工具与机构约束，再增加驱动、预检查及全过程审计对联动朝向的支持；当前尚未实现或真机验证这条执行路线。

## 4. 本次不执行自动负载运动

当前配置的检查命令可用于离线确认限制：

```powershell
& $loadPython -X utf8 -m scripts.lab_session check --config $loadConfig
```

应报告`rigid shank connection requires a coupled position/orientation trajectory`；非TwinCAT电脑还可能提示DLL缺失。前者是机构约束与当前执行能力不匹配，后者是本机采集依赖，不是驱动器故障。不要把该配置送入`prepare/run/repeat/train`；本次没有刚性连接的自动运动命令。

新加入的通用小幅诊断通路仅服务将来**实际采用可转动连接且独立确认整条路径**的装置：新起点、真实安装复核、明确的原始TCP力/力矩上限、冻结绑定、假腿全程采集、严格受力新鲜度、完成后独立审核，不接学习器。`scripts.design_loaded_diagnostic review --help`说明建档参数；它无法复核通过`rigid_fixture`。更换连接形式不是本次要求，不需为运行而换装或改标签。

正式论文试验仍需先解决受力缺口、核实补偿/无接触偏置、冻结可行位姿曲线，再做统一90秒周期的节奏候选及重复测量。诊断结果不混入GP或旧候选历史，不作为舒适度改善结论。

## 5. 成功或失败都回传

先把本次实际现象写入`setup_record.md`，保留所有原始文件、失败状态和版本。只暂存本次新目录，先查看是否已有其他暂存变更：

```powershell
git diff --cached --name-only
git status --short
git add -- $loadRoot
git diff --cached --stat
```

确认暂存只包含本次拟回传内容，再执行：

```powershell
git commit -m "Record rigid shank setup and connected diagnostic data"
if ($LASTEXITCODE -ne 0) { throw '提交未完成，保留目录。' }
git push origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '推送未完成，本地提交仍保留，不要强推。' }
```

若推送提示远端更新，先保留本地提交并回传提示，不reset、强推或覆盖两边原始数据。
