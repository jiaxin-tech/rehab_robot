# 后续实验操作总手册

更新：2026年11月14日。分支：`paper-personalized-load`。历史旧电脑仓库为`C:\Users\liumai\fjx\rehab_robot`，如已迁移，以实际已有仓库为准。每块命令单独执行，查看结果后再进入下一块；本页不是一键连续运行脚本。

**最新现场方向：先录制康复轨迹。** 用户确认当前高度/前后行程不合适，选择原厂拖动示教。先按[拖动录制与离线预览](DRAG_TEACHING_WORKFLOW.md)采集实际曲线；第7–9节保留为路径另行确认后的后续流程，本次不继续重复旧大行程或将录制文件直接传给run。当前录制/预览尚未接入真实回放，`ready_for_replay=false`。

## 0. 现在从哪里开始

| 阶段 | 当前情况 | 下一步 |
|---|---|---|
| 假腿只读连接 | 17:40与17:43的十秒诊断已回传 | 不需重复同一诊断 |
| 轴1状态 | 17:40仍为`0x0218 / 0x0022`；17:43后回传帧为`0x0231 / 0x0000` | 状态变化原因未记录，不能归因代码修复；保留两批原始记录 |
| 正常假腿采集 | 已有4968行、约9.999秒CSV | 约496.745Hz且有接收缺口，质量报告仍不通过，不当完整500Hz验收 |
| 机械臂 | 本次1mm反馈验证与大行程空载已有记录 | 不再重复不合适的幅度，先进行只读拖动录制 |
| 当前正式受力标定 | 无接触偏置null、补偿核实false | 按第5节保留事实，不能填零/true冒充完成 |
| 新空载/带腿完整任务 | 原空载计划范围不适合；受力记录还有14处>50ms缺口 | 冻结新示教路径并另行审查空载；带腿优化尚未完成 |

当前最低有效交付是录制并预览一条真实示教路径，保存TCP/工具与安装说明。后面的运动阶段有各自前提，录制或诊断完成不会自动放行运动。此前各诊断命令继续作为参考，不重复当成当前待做任务。

两个已确认事实：旧正式采集从10月16日就检查fault/非零603F；旧映射诊断只读取，不因fault中断。今天新增十秒`inspect`对应后者用途，所有记录标为`DIAGNOSTIC_ONLY`。它不修复驱动故障，也不能替代正常采集或论文性能数据。

## 1. 每次到场先更新并建立目录

在旧电脑PowerShell进入仓库：

```powershell
Set-Location 'C:\Users\liumai\fjx\rehab_robot'
git branch --show-current
git status --short
git diff --cached --name-only
```

应处于`paper-personalized-load`。有未提交记录时先按第10节保存；有其他代码/配置修改或暂存内容时先核对，不覆盖、不自动stash、不reset/clean。工作区干净后：

```powershell
git pull --ff-only origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '更新失败，保留现场文件并停止。' }
$labPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labPython -PathType Leaf)) { throw '找不到已有项目Python。' }
& $labPython -c "import sys,struct; print(sys.executable); print(sys.version); print('bits=',struct.calcsize('P')*8)"
if ($LASTEXITCODE -ne 0) { throw 'Python启动失败。' }
$labConfig = 'config/experiment.json'
$labRun = 'results/manual_lab/' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Path $labRun -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath 'docs/templates/LAB_RECORD_TEMPLATE.md' -Destination (Join-Path $labRun 'lab_record.md') -ErrorAction Stop
Write-Host "本次诊断目录：$labRun"
```

保持同一个PowerShell窗口。窗口关闭后重新定义变量，已经生成的目录和原始数据继续保留，下一次设备操作用新目录。这里不重建虚拟环境、不重装TwinCAT或SDK。

## 2. 当前可做：十秒独立假腿诊断

保持装置当前静止状态，执行：

```powershell
$legInspect = Join-Path $labRun 'leg_diagnostic10s'
& $labPython -X utf8 -m scripts.manual_leg_check inspect --live --duration-s 10 --config $labConfig --output-dir $legInspect
$legInspectExit = $LASTEXITCODE
Write-Host "诊断退出码：$legInspectExit；目录：$legInspect"
$legStatus = Get-Content -LiteralPath (Join-Path $legInspect 'segment_status.json') -Raw | ConvertFrom-Json
$legStatus | Select-Object status,completed_reads,observed_span_s,data_classification,error
$legStatus.summary | Select-Object toggle_transitions,axis_status_distributions,axis_error_distributions | ConvertTo-Json -Depth 6
```

预期保存`inspection_frames.json`、`segment_status.json`、`manifest.json`、`leg_config_snapshot.json`及`twincat_file_evidence.json`。故障字段作为诊断标记保留，原生读取或关闭失败仍停止并保存错误。`COMPLETE`只表示这一段只读工作完成；检查实际跨度与读取次数，不能把至少5ms的诊断poll称为500Hz源采集验收。

本模式不产正常`leg.csv`，也不能作为后续capture的probe。`DIAGNOSTIC_ONLY`、`capture_authorized=false`、`training_authorized=false`应保留。成功或失败都按第10节回传`$labRun`。

轴1的厂家/型号及0x0022含义仍需设备资料。在线数值已对上，无须反复重做同样的20帧检查。若需要补配置原文，具体只读复制命令见[20帧复核](lab/LEG_INSPECT_REVIEW_2026-11-14.md)；这里没有厂家故障复位或参数写入命令。

## 3. 原因处理后：正常假腿probe与十秒采集

只有设备状态和现场映射已核实、原有守卫能正常通过时使用本节。先核对这次实际处理原因和在线状态，不能删除故障位、修改偏移或放宽阈值来获得通过。

```powershell
$legCheck = Join-Path $labRun 'leg_check'
& $labPython -X utf8 -m scripts.manual_leg_check check --offline --config $labConfig --output-dir $legCheck
if ($LASTEXITCODE -ne 0) { throw '离线检查失败，保存目录并回传。' }
$probeSession = Join-Path $labRun 'leg_probe'
& $labPython -X utf8 -m scripts.manual_leg_check probe --live --config $labConfig --output-dir $probeSession
if ($LASTEXITCODE -ne 0) { throw 'probe失败，本次停止正常采集并回传。' }
```

成功要求manifest和segment status均为`COMPLETE/ok=true`，20读完成、有toggle变化且守卫未失败。在同机、同配置、同代码的成功probe后十分钟内：

```powershell
$legCapture = Join-Path $labRun 'leg_capture10s'
& $labPython -X utf8 -m scripts.manual_leg_check capture --live --duration-s 10 --probe-session $probeSession --config $labConfig --output-dir $legCapture
$legCaptureExit = $LASTEXITCODE
Write-Host "正常采集退出码：$legCaptureExit"
$legCsv = Join-Path $legCapture 'leg.csv'
if (Test-Path -LiteralPath $legCsv -PathType Leaf) {
    & $labPython -X utf8 -m measurement_validation.leg_episode $legCsv --calibration $labConfig --minimum-rate-hz 500 --output (Join-Path $labRun 'leg_quality10s.json')
}
```

即使生成质量报告并退出0，也要看`findings`和`usable_for_dynamics`。失败留下的部分CSV保留失败状态；当前配置中的关节标定只是已存参数，不因报告生成就重新证明准确。主机时间缺口不是确切设备丢帧，跨电脑单调时钟不能直接当作同步时钟。

## 4. 可以单独推进：机械臂三十秒只读观测

假腿仍有异常时，可在机械臂保持静止、无人接触工具的条件下，仅观察机器人状态与六维受力。为了不因假腿诊断阻断这一项，创建一份**仅用于observe的配置副本**；原`config/experiment.json`保持不变，正式带腿保护不从这里关闭。

```powershell
$robotObserveConfig = Join-Path $labRun 'robot_only_observe_config.json'
$siteConfigHashBefore = (Get-FileHash -LiteralPath $labConfig -Algorithm SHA256).Hash
& $labPython -c "import json,sys; from pathlib import Path; s=Path(sys.argv[1]); d=Path(sys.argv[2]); c=json.loads(s.read_text(encoding='utf-8-sig')); c['validation']['capture_dummy_leg']=False; f=d.open('x',encoding='utf-8'); json.dump(c,f,ensure_ascii=False,indent=2); f.write('\n'); f.close()" $labConfig $robotObserveConfig
if ($LASTEXITCODE -ne 0) { throw '只观测配置副本生成失败。' }
if ((Get-FileHash -LiteralPath $labConfig -Algorithm SHA256).Hash -ne $siteConfigHashBefore) { throw '原配置发生变化，停止核对。' }
$robotObserve = Join-Path $labRun 'robot_observe30s'
& $labPython -X utf8 -m scripts.lab_session observe --config $robotObserveConfig --session $robotObserve --duration 30
$robotObserveExit = $LASTEXITCODE
Write-Host "机器人观测退出码：$robotObserveExit；目录：$robotObserve"
```

这条observe只连接并读取机器人，不发上电、模式或轨迹命令；当前代码在`capture_dummy_leg=false`的observe路径不访问ADS。此副本仅用于本节，不能拿它代替带腿实验配置。

看`round_status.json`是否为`OBSERVATION_COMPLETE`，`robot_timing.ever_ready`、就绪后失效、状态/力年龄、skew、原始样本及清理是否正常。`robot/robot_wrench.csv`保存六维原始值。只看“文件存在”或查询均值不足以通过时序检查；失败整目录回传。

## 5. 进入运动和论文学习前的现场配置

先记录实际TCP/工具、负载、安装、机器人IP/本机IP、受力坐标与参考点、单位、停止条件和工作空间。代码不会证明这些现场事实。当前0.9m高度、-0.2m水平位移只是沿用参数，应对真实起点和装置审核。

配置检查不连接设备，可先查看缺项并保存文字：

```powershell
& $labPython -X utf8 -m scripts.lab_session check --config $labConfig 2>&1 | Tee-Object -FilePath (Join-Path $labRun 'configuration_check.txt')
$configCheckExit = $LASTEXITCODE
Write-Host "正式配置检查退出码：$configCheckExit"
```

当前`compensation_verified=false`、偏置null时，正式带载检查报阻断属于预期；它不妨碍第4节只读观察。CLI没有`check --empty-load`，空载准备会执行自己的检查。

| 要完成的事实 | 配置/记录 | 操作边界 |
|---|---|---|
| 无接触六维偏置 | `force_training.unloaded_wrench_bias`，顺序Fx/Fy/Fz/Tx/Ty/Tz，N与N·m | 从确认无接触、有代表性的原始记录确定；腿受载时不清零 |
| 工具/负载及SDK外力补偿核实 | `force_training.compensation_verified`与现场记录 | 不能用把false改true代替验证，应保留设置及不同姿态残差检查依据 |
| 固定表达坐标和参考点 | `force_training.wrench_frame_type`及安装说明 | 与实际SDK记录一致，保持跨轮可比 |
| 实际安装/机械条件 | `force_training.study_identity.setup_id`、`force_training.study_identity.mechanical_condition_id`及照片/物理定义 | 真实条件另起研究历史；不能假设ADS能设置刚度/阻尼 |
| 同周期任务 | `force_training.fixed_cycle_duration_s=164`及实际重定时计划 | 模板80/80、70/90、90/70；重定时后周期改变就重新冻结协议，不提高限速迁就 |
| 尝试预算 | `total_budget`、可选`evaluation_budget` | 参考、失败、自适应与评价成本都记录；评价预算缺省等于自适应预算 |

仓库没有一个可自动完成并写入上述正式标定的现场命令。`diagnose_robot alignment --software-bias`只是会话内软件偏置，不能据此宣布SDK补偿已验证。原始数据回传后再确定校准结果和版本，保留false/null直到有依据。

**当前TCP受力主线不以`training.enabled=false`作为总开关。** 完整带载run后会自动计算/训练；空载不训练，显式`--evaluation`只评价不更新GP。不要靠这个旧字段来实现“带载只采集不训练”。

## 6. 有现场条件后：一毫米空载实时链路诊断

本节开始有真实运动。假腿/人体与机器人脱开，保留已核对TCP工具和负载，现场确认基座+Z一毫米及返回安全、区域清空、急停可用。代码、SDK、工具、配置或安装变化后，应重新做本节。

```powershell
$rtSession = Join-Path $labRun ('rt_link_1mm_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $labPython -X utf8 -m scripts.test_realtime_motion prepare --config $labConfig --session $rtSession --step-mm 1
if ($LASTEXITCODE -ne 0) { throw '微动准备失败，保存记录并停止。' }
Get-Content -LiteralPath (Join-Path $rtSession 'status.json') -Raw
```

prepare读取当前TCP，不发运动。检查`PREPARED`、方向、冻结的trajectory与anchor；确认本节空载条件成立后，才单独执行：

```powershell
& $labPython -X utf8 -m scripts.test_realtime_motion run --session $rtSession --live --confirm-unloaded --operator-confirmation "I CONFIRM SUPERVISED SLOW ROBOT MOTION"
Write-Host "一毫米诊断退出码：$LASTEXITCODE"
```

通过需`status.json=COMPLETE`，实际反馈的远端/返回验证通过，`motion_commanded=true`、完成原因和同会话停止/清理证据齐全。callback运行、命令行退出0或发过命令都不能单独证明运动成功。失败不复用目录，不自动扩大位移或放宽网络/时效门限。

## 7. 微动通过后：完整空载TCP轨迹

仍需机器人与假腿/人体脱开。下列prepare读取起点并做整条路径检查，不运动：

```powershell
$emptySession = Join-Path $labRun ('empty_full_tcp_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $labPython -X utf8 -m scripts.lab_session prepare --config $labConfig --empty-load --session $emptySession
if ($LASTEXITCODE -ne 0) { throw '空载准备失败。' }
$emptyState = Get-Content -LiteralPath (Join-Path $emptySession 'round_status.json') -Raw | ConvertFrom-Json
$emptyState | Select-Object status,blockers
```

要求`PREPARED`且`blockers=[]`，人工核对`preview/`、`tcp_plan.json`、`tcp_trajectory.csv`、`tcp_safety.json`的全路径、朝向、范围、速度、加速度、实际周期和现场障碍物。准备后起点/工具/负载变化就新建会话。

条件满足后逐条执行，run会真实运动：

```powershell
& $labPython -X utf8 -m scripts.lab_session run --empty-load --live --session $emptySession
if ($LASTEXITCODE -ne 0) { throw '空载运行失败，保留整目录并停止。' }
& $labPython -X utf8 -m scripts.lab_session complete --session $emptySession
if ($LASTEXITCODE -ne 0) { throw '空载最终核验失败，停止进入后续阶段。' }
if ((Get-Content -LiteralPath (Join-Path $emptySession 'round_status.json') -Raw | ConvertFrom-Json).status -ne 'COMPLETE') { throw '空载最终状态不是COMPLETE。' }
```

run应为`EMPTY_LOAD_COMPLETE`、训练`NOT_APPLICABLE`；complete后为`COMPLETE`。还要核对实测TCP确实完成整轨迹和返回、原始力/状态覆盖全周期、无未解释异常及清理完成。`dispatch.csv`只证明主机记录，不代表机械臂一定走了。空载成功不等于带腿通过。

## 8. 正式配置冻结后：首个带腿参考回合

先完成第3、5、6、7节相应条件，核对连接、假腿物理范围、被动状态和停止方式。当前轴1未解释状态不能由诊断模式来代替通过。首次正式比较前使用新研究目录及配置副本：

```powershell
$paperStudy = 'results/paper_study/' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Path $paperStudy -ErrorAction Stop | Out-Null
$paperConfig = Join-Path $paperStudy 'paper_config.json'
& $labPython -c "import json,sys; from pathlib import Path; c=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig')); t=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')); assert c['trajectory']['mode']=='independent_tcp' and c['training']['mode']=='tcp_force'; keys=('reference_candidate_id','candidates','fixed_cycle_duration_s','study_identity'); c['force_training'].update({k:t['force_training'][k] for k in keys}); f=Path(sys.argv[3]).open('x',encoding='utf-8'); json.dump(c,f,ensure_ascii=False,indent=2); f.write('\n'); f.close()" $labConfig 'config/paper_personalized_load.template.json' $paperConfig
if ($LASTEXITCODE -ne 0) { throw '论文配置副本生成失败。' }
Copy-Item -LiteralPath 'docs/templates/LAB_RECORD_TEMPLATE.md' -Destination (Join-Path $paperStudy 'study_record.md') -ErrorAction Stop
Write-Host "论文研究目录：$paperStudy；待核对配置：$paperConfig"
```

这条命令以实际`$labConfig`为底，只从模板迁入参考候选ID、候选目录、固定周期与待填写研究身份；现场IP、工具/负载相关设置、工作空间、坐标、限值、采集和已核实标定保留，不整份拿旧模板覆盖现场。没有实测依据的false/null继续保留。

按第5节填写有真实依据的标定、安装与机械条件，保存定义和照片。每个正式机械条件独立目录，观察确认效果前冻结公共候选、端点、预算、评价和顺序。填写并核对之后：

```powershell
& $labPython -X utf8 -m scripts.lab_session check --config $paperConfig
if ($LASTEXITCODE -ne 0) { throw '正式配置仍有阻断，保存清单并停止。' }
```

将`$paperConfig`与第6、7节已通过的实际配置/安装逐项核对。运动设置、候选、方向、采集、工具/负载或安装有变化时，先用`$paperConfig`替换那两节命令中的`--config $labConfig`，另建微动/空载会话并重新通过；不能把原配置的空载通过当作新配置已验证。然后才准备带腿：

```powershell
$paperFirst = Join-Path $paperStudy 'round_001_reference'
& $labPython -X utf8 -m scripts.lab_session prepare --config $paperConfig --session $paperFirst
if ($LASTEXITCODE -ne 0) { throw '带腿准备失败。' }
Get-Content -LiteralPath (Join-Path $paperFirst 'round_status.json') -Raw | ConvertFrom-Json | Select-Object status,blockers
```

只有`PREPARED`、无blocker且全预览及现场核对通过，才执行下一块。准备后任何安装或起点变化都重新建会话。

```powershell
& $labPython -X utf8 -m scripts.lab_session run --live --session $paperFirst
if ($LASTEXITCODE -ne 0) { throw '首轮失败，保留研究根目录并停止。' }
& $labPython -X utf8 -m scripts.lab_session complete --session $paperFirst
if ($LASTEXITCODE -ne 0) { throw '首轮最终核验失败，不能进入下一轮。' }
if ((Get-Content -LiteralPath (Join-Path $paperFirst 'round_status.json') -Raw | ConvertFrom-Json).status -ne 'COMPLETE') { throw '首轮最终状态不是COMPLETE。' }
```

当前主线run后通常为`CAPTURE_COMPLETE`、`training_status=COMPLETE`，不是必须`ROUND_COMPLETE`；complete通过后为`COMPLETE`。需要原始`robot/`、`dispatch.csv`、计划/配置、`tcp_force_training.json`、账本及启用台架采集时的`leg/`和相应分析。单个参考回合只证明链条与测量，不证明个性化收益。

## 9. 后续自适应候选与独立评价

先查看已完成参考的`tcp_force_training.json`和report。以下`$adaptiveParent`必须是本研究最新已计费、有效训练回合；不是任意较早的祖先。首次搜索只初始化一次：

```powershell
$adaptiveParent = $paperFirst
```

每轮用下面命令准备新目录，后续沿成功更新后的父轮继续：

```powershell
$paperNext = Join-Path $paperStudy ('adaptive_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $labPython -X utf8 -m scripts.lab_session repeat --source $adaptiveParent --trained --session $paperNext
if ($LASTEXITCODE -ne 0) { throw '没有可继续候选或准备失败，查看STOP/预算/账本原因。' }
```

repeat只准备。再次核对新候选的预览、完整周期、范围、起点、blockers，再执行：

```powershell
& $labPython -X utf8 -m scripts.lab_session run --live --session $paperNext
if ($LASTEXITCODE -ne 0) { throw '候选运行失败，保存完整研究目录并停止。' }
& $labPython -X utf8 -m scripts.lab_session complete --session $paperNext
if ($LASTEXITCODE -ne 0) { throw '候选最终核验失败，不能更新父轮。' }
if ((Get-Content -LiteralPath (Join-Path $paperNext 'round_status.json') -Raw | ConvertFrom-Json).status -ne 'COMPLETE') { throw '候选最终状态不是COMPLETE。' }
$adaptiveParent = $paperNext
```

成功后父轮已更新为刚完成的`$paperNext`，下一轮沿这个值、使用新编号目录，不再重置为首轮。STOP后不继续自适应；兄弟目录和旧祖先不能重开免费预算。已在共享账本登记的run尝试若FAILED或残留RUNNING，会封闭该研究，不能删账本/锁文件续跑；prepare/check失败通常尚未计费，complete失败也不一定改写run账本，但仍须停止并核查最终证据。

独立评价先冻结推荐。选择已完成的那个候选源会话作为`$evaluationSource`，再运行下列prepare；如果推荐在早前回合，用其真实路径，不能把候选ID改成另一条轨迹：

```powershell
$evaluationSource = Read-Host '粘贴本次预定复测候选的真实已完成源会话目录'
if (-not (Test-Path -LiteralPath (Join-Path $evaluationSource 'round_status.json') -PathType Leaf)) { throw '源会话路径无效。' }
$paperEvaluation = Join-Path $paperStudy ('evaluation_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $labPython -X utf8 -m scripts.lab_session repeat --source $evaluationSource --evaluation --session $paperEvaluation
if ($LASTEXITCODE -ne 0) { throw '独立评价准备失败。' }
```

输入的源路径应对应已经冻结的公共候选或个体推荐，不让默认首轮路径代替实际选择。审核新预览后再运动：

```powershell
& $labPython -X utf8 -m scripts.lab_session run --live --session $paperEvaluation
if ($LASTEXITCODE -ne 0) { throw '评价运行失败，保留整目录并停止。' }
& $labPython -X utf8 -m scripts.lab_session complete --session $paperEvaluation
if ($LASTEXITCODE -ne 0) { throw '评价最终核验失败。' }
if ((Get-Content -LiteralPath (Join-Path $paperEvaluation 'round_status.json') -Raw | ConvertFrom-Json).status -ne 'COMPLETE') { throw '评价最终状态不是COMPLETE。' }
```

评价保存`tcp_force_evaluation.json`、占用评价预算、不更新GP，不能成为下一自适应父轮。STOP后允许的独立评价仍受预算、数据和失败锁存约束。

离线重算只对已完成记录执行，不运动：

```powershell
& $labPython -X utf8 -m scripts.lab_session analyze --session $paperFirst
```

训练/分析报错先保存原始文件和错误。离线修正不等于恢复失败研究的硬件续跑资格；不通过修改旧配置、重写原CSV或删除失败尝试换取通过。

## 10. 每一阶段结束：精确回传整个本次目录

诊断用`$labRun`，论文回合用`$paperStudy`。**后续回合会改首轮目录里的共享账本，所以正式研究要提交整个研究父目录，不能只提交最新一轮。** 失败、部分文件、配置和现场记录一并保留。

```powershell
$returnRoot = Read-Host '粘贴本次要回传的结果父目录：诊断用labRun，正式研究用整个paperStudy'
if (-not (Test-Path -LiteralPath $returnRoot -PathType Container)) { throw '结果父目录不存在，先核对实际路径。' }
Get-ChildItem -LiteralPath $returnRoot -Recurse -File | Select-Object Length,FullName
git diff --cached --name-only
git status --short
```

在新窗口保存旧记录时，也直接粘贴那个实际已有目录，不依赖尚未定义的`$labRun`变量。正式研究输入整个本次`$paperStudy`，不要输入整个results目录。有其他暂存内容先核对，不混入本次提交。确认只包含本次数据，且原始CSV的`git check-attr text -- <实际CSV路径>`显示`text: unset`，再提交：

```powershell
git add -- $returnRoot
if ($LASTEXITCODE -ne 0) { throw '本次目录暂存失败。' }
git diff --cached --stat
git diff --cached --name-only
```

确认清单后，按真实结果填写说明：

```powershell
git commit -m "Record lab session results and failures"
if ($LASTEXITCODE -ne 0) { throw '提交失败，保留文件并停止。' }
git push origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '上传失败，保留本地提交，不要强推。' }
git log -1 --oneline
```

`.gitignore`放行`results/manual_lab/`和`results/paper_study/`。旧示例的`results/force_*`默认忽略，故本手册统一使用这两个范围。不要改写账本中的原机绝对路径以迁移现场运行；回传先用于离线核查。若push因远端更新被拒绝，保留本地提交号和错误，不强推覆盖。

## 11. 异常停止与可选空载回位

运行失败时先确认设备实际停止，按现场急停/控制器流程处理；程序退出不能替代物理停止。保留所有目录、原因及清理信息，不主动制造断网或碰撞测试，不复用失败会话。

回位不是自动步骤。只有已脱开假腿/人体、现场监督，并确认源预览是实际走过且当前仍畅通的路径时，才考虑以下空载方案。源文件存在本身不证明可以回位；不清楚路线时保持停止。

```powershell
$restoreSource = $emptySession
$restoreSession = Join-Path $labRun 'unloaded_restore'
& $labPython -X utf8 -m scripts.restore_session_tcp prepare --source-session $restoreSource --config $labConfig --session $restoreSession
```

prepare只读取当前位置并冻结路线，默认沿原路径逆向，不提供避障。核对`PREPARED`、完整路线、当前TCP/朝向及空载条件后，才单独执行：

```powershell
& $labPython -X utf8 -m scripts.restore_session_tcp run --session $restoreSession --live --confirm-unloaded --operator-confirmation "I CONFIRM SUPERVISED TCP RESTORE"
```

该工具限制分段和速度，并复核状态、力、工作空间与清理；任何失败都保留记录。这里不列`--direct`为常规恢复方式，也不提供挂腿或有人场景的自动回位。

## 12. 完整参考跑通以后，论文实验怎么收敛

先做同一机械条件下三候选的重复性与节奏敏感性先导，预先安排RAB、BRA、ABR三块，每候选三次。人工顺序属于预定实验，不事后称算法选点。当前CLI没有`--candidate-id`供任意切换；先导执行前需冻结每候选配置/准备选择和完整trial plan，不能在现成预览或已完成记录上改ID来凑九回合，也不能用第9节`--trained`强行兑现人工RAB顺序。

先导之后冻结正式空间任务、实际总周期、端点、机械条件、公共候选、尝试预算和评价方案。比较固定公共候选、相同成本的简单选择、GP/EI推荐，并在独立完整回合复测推荐；搜索中最小的一次观测不是最终效果。三候选很少，GP/EI是否有附加价值由实验判断。

力与力矩RMS、全周期峰值、实际周期、任务跟随和失败成本分别报告，不能只报复合load_score。受控假腿配置不是患者，负荷降低只是舒适相关机械指标。不能从假腿实验写出主观舒适或临床获益。

| 日期 | 收敛目标 |
|---|---|
| 11月27日前 | 争取可信完整假腿参考及重复，测出一轮总周转时间 |
| 12月4日前 | 判断节奏差异是否超过重复误差，以及机械条件是否有不同需求 |
| 12月11日前 | 冻结正式协议、基线、候选、端点、预算和评价；之后不继续扩算法 |
| 元旦前 | 尽量完成主要实验与独立推荐评价 |
| 2027年1月15日前 | 完成预定复测、返工和全部实验室实验 |

按每周两次、每次三至四小时且独自调试安排。固定164秒任务的九回合仅运动就需24.6分钟，还要加复位、稳定、检查和异常处理；先测真实周转时间，再决定重复规模。候选差异小于重复误差时先修测量；公共候选对所有条件都同样好时如实记录，不能靠改权重或抹去失败制造个性化。

详细科学口径与周次计划见[当前论文工作流](CURRENT_WORKFLOW.md)，实现原理见[TCP受力方法](TCP_FORCE_EXPERIMENT.md)。旧几何/六参数操作记录仅作历史对照，本页是当前后续操作入口。
