# 示教轨迹空载验证操作

**当前入口：90秒连续空载配置。** 使用 `config/continuous_taught_90s.empty.json`；沿用已确认的空载上限0.13米/秒、0.45米/秒²。设计与几何改动见[90秒说明](CONTINUOUS_90S_WORKFLOW.md)，70秒回传及425秒基线保留为历史记录。

更新日期2026年11月15日。当前采用已确认的真实示教形状，保留上抬、高位前伸、回收下降，三轴幅度均为1，工具朝向固定为本次prepare捕获的起点朝向。原始录制不修改。

当前计划周期90秒，全程没有保持段：0～5秒平滑加速，5～85秒沿曲线约3.50厘米/秒匀速，85～90秒平滑减速。抬升、高位前伸、回收下降只是连续曲线上的阶段标签，约30.148秒和47.467秒经过两处分界时不停车、不重新起步。计划90秒不包含SDK连接、准备和清理时间。

新模式`continuous_taught_tcp`目前只支持空载验证，尚未接入带载GP学习。70秒回传仍有5处受力缺口和查询清理未确认记录，见[回传复核](lab/EMPTY_70S_REVIEW_2026-11-15.md)；连续节奏不代表受力链已修复。

## 更新与检查

在旧电脑PowerShell逐块执行。已有现场修改先保存，不覆盖、不reset/clean。仓库若已迁移，使用实际已有路径。

```powershell
Set-Location 'C:\Users\liumai\fjx\rehab_robot'
git branch --show-current
git status --short
git diff --cached --name-only
```

确认当前分支为`paper-personalized-load`且工作区、暂存区干净，再执行：

```powershell
$taughtBranch = git branch --show-current
if ($taughtBranch.Trim() -ne 'paper-personalized-load') { throw '当前不是论文分支，先停止核对。' }
git pull --ff-only origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '更新失败，保留现场文件。' }
$taughtPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taughtPython -PathType Leaf)) { throw '找不到项目已有Python。' }
$taughtConfig = 'config/continuous_taught_90s.empty.json'
$taughtRun = 'results/manual_lab/' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Path $taughtRun -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath 'docs/templates/LAB_RECORD_TEMPLATE.md' -Destination (Join-Path $taughtRun 'lab_record.md') -ErrorAction Stop
$taughtSession = Join-Path $taughtRun 'continuous_empty_01'
& $taughtPython -X utf8 -m scripts.lab_session check --empty-load --config $taughtConfig
if ($LASTEXITCODE -ne 0) { throw '检查失败，暂不准备运动。' }
```

READY仅代表配置检查通过，不证明设备连通或新轨迹已经实际完成。保持同一个PowerShell窗口，并在`lab_record.md`填写实际工具、负载、起点和连接情况。

## 准备新会话并看预览

机器人与假腿、人体脱开，核对TCP、工具、负载和停止手段。新起点会平移整条基座相对路径，必须审核完整空间；固定朝向下整圈可达性仍需现场验证。

```powershell
& $taughtPython -X utf8 -m scripts.lab_session prepare --config $taughtConfig --empty-load --session $taughtSession
if ($LASTEXITCODE -ne 0) { throw '准备失败，保存目录并回传。' }
Get-Content -LiteralPath (Join-Path $taughtSession 'round_status.json') -Raw
Get-Content -LiteralPath (Join-Path $taughtSession 'tcp_plan.json') -Raw
```

要求`PREPARED`且`blockers=[]`。打开会话内预览：

```powershell
Invoke-Item -LiteralPath (Join-Path $taughtSession 'preview/trajectory_3d.png')
Invoke-Item -LiteralPath (Join-Path $taughtSession 'preview/xyz_time.png')
Invoke-Item -LiteralPath (Join-Path $taughtSession 'preview/speed_time.png')
Invoke-Item -LiteralPath (Join-Path $taughtSession 'preview/acceleration_time.png')
```

核对全路径、两个转折、朝向、范围、完整90秒周期，以及`tcp_safety.json`中0.13米/秒、0.45米/秒²上限。速度图应只在首尾各5秒渐变，中间保持约3.50厘米/秒，两个阶段分界处不降为零。默认相对起点最高抬升75.10厘米、最远Base +Y位移89.25厘米；这是基座方向，不直接等同人体前后方向。原示教起点下最高Z约1.012米，现场起点改变后该绝对值也会改变。准备后移动起点、更换工具或修改配置，须新建会话重新prepare。

## 显式执行空载

核对完成后，下面命令会真实运动：

```powershell
& $taughtPython -X utf8 -m scripts.lab_session run --empty-load --live --session $taughtSession
$taughtRunExit = $LASTEXITCODE
Write-Host "运动退出码：$taughtRunExit"
Get-Content -LiteralPath (Join-Path $taughtSession 'round_status.json') -Raw
```

不调用旧直线任务，不加`--from-current`，不执行带腿或`repeat --trained`。异常按现场停止流程处理，保留完整目录。

运动结束后单独做离线审核：

```powershell
& $taughtPython -X utf8 -m scripts.lab_session complete --session $taughtSession
Write-Host "完成审核退出码：$LASTEXITCODE"
```

`complete`不带`--live`。失败不改成成功；审核核对完整命令、实际位置和朝向、原始受力时间缺口及清理记录。旧会话的`EMPTY_LOAD_COMPLETE`不能替代新会话审核通过。无论成功失败都保留并回传。

## 以后改变幅度

重新设计使用新目录、新配置，例如：

```powershell
& $taughtPython -X utf8 -m scripts.design_taught_trajectory --session results/manual_lab/20261115_143705/drag_teach_01 --config config/experiment.json --scale-z 0.9 --scale-y 0.9 --output-dir results/manual_lab/taught_design_v2 --output-config config/taught_tcp_empty_v2.json
if ($LASTEXITCODE -ne 0) { throw '新设计失败，查看summary并保留目录。' }
```

此处0.9仅演示空间参数，不是本次批准幅度。三轴scale对应基座XYZ；`--scale-x`也可独立设置，默认均为1。此命令生成的是分段几何来源，不能直接当成90秒连续版本执行；重新设计会保守重算节奏。若需继续使用90秒连续节奏，按[90秒说明](CONTINUOUS_90S_WORKFLOW.md)将新分段settings作为来源，另存连续几何、配置和预览，重新核对幅度与限值。采样周期0.02秒与空间控制点间隔不是同一个参数。

新版本重新看图后，将`$taughtConfig`改为新配置的精确路径，以新`$taughtSession`重复check和prepare。不得覆盖原scale1设计，不手改原始CSV。正式论文比较内应冻结同一空间形状、幅度、朝向及总周期；当前新形状尚无正式带载候选目录。

## 回传

先确认暂存区没有其他内容，再只添加本次完整父目录：

```powershell
git diff --cached --name-only
git status --short
git add -- $taughtRun
git diff --cached --stat
git diff --cached --name-only
```

确认只有本次数据后：

```powershell
git commit -m "Record continuous 90-second empty-load validation"
if ($LASTEXITCODE -ne 0) { throw '提交失败，保留数据。' }
git push origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '回传失败，保留本地提交，不强推。' }
git log -1 --oneline
```

失败记录也回传；若另生成新设计和配置，核对后将其精确路径一同保存。示教形状、空载跟随和带载优化是不同证据，本次不能证明人体舒适或临床效果。
