# 2026年11月14日旧电脑静止检查与数据回传

本次在旧实验电脑本地运行，只检查假腿ADS连接、原始反馈和短段静止记录。今天最新探测已确认本机AMS为10.20.4.89.1.1，但在轴1采集守卫处失败；具体是使能、故障、错误码、力矩还是映射异常，尚待新诊断记录确认。因此先更新代码再做一次probe；失败就保留结果并回传，不继续采集，不修改参数来消除报错。

本流程不使用临时远程助手、不要求远程登录，也不执行机器人或假腿运动。临时增加的192.168.137.2是普通网卡IPv4，不是ADS地址；本地ADS读取仍须使用实际TwinCAT环境和已核对的映射。端口11返回ADS状态0不能单独证明TwinCAT处于CONFIG，不据此切换运行状态。

以下命令在旧电脑PowerShell中逐块执行。每块都先看结果，再执行下一块。不要整页一次粘贴运行。

## 1 核对仓库并更新physics分支

历史确认的旧仓库路径如下；如果已迁移，应先进入实际已有仓库，不创建空目录代替。

```powershell
Set-Location 'C:\Users\liumai\fjx\rehab_robot'
git status --short
git branch --show-current
git remote -v
```

若status有本地代码、配置或未保存数据，先停止更新，保留原状并告知协作者。不要使用reset、clean、自动stash或覆盖文件。若已有暂存内容，也不要把它混入本次数据提交。

工作区干净后，逐条更新：

```powershell
git fetch origin physics
if ($LASTEXITCODE -ne 0) { throw '获取更新失败，停止。' }
git switch physics
if ($LASTEXITCODE -ne 0) { throw '切换分支失败，停止。' }
git merge --ff-only origin/physics
if ($LASTEXITCODE -ne 0) { throw '分支不能直接前进，保留现场改动并停止。' }
git status --short
```

若physics分支不存在、发生分叉或更新失败，不临时建立另一条同名研究分支，也不强推；把错误信息交给协作者处理。更新完成后应能找到 `scripts/manual_leg_check.py` 和本说明。

## 2 选择已有Python并建立本次结果目录

优先使用旧仓库已有虚拟环境。此处不安装Python或依赖，也不需要激活环境。

```powershell
$labPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labPython -PathType Leaf)) { throw '找不到已有项目Python，先停止并核对实际解释器路径。' }
& $labPython -c "import sys,struct; print(sys.executable); print(sys.version); print('bits=',struct.calcsize('P')*8)"
if ($LASTEXITCODE -ne 0) { throw 'Python启动失败，停止。' }

$labRun = 'results/manual_lab/' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Path $labRun -ErrorAction Stop | Out-Null
$probeSession = Join-Path $labRun 'probe'
$captureSession = Join-Path $labRun 'capture_10s'
$labCsv = Join-Path $captureSession 'leg.csv'
Write-Host "本次结果目录：$labRun"
```

ADS DLL需要64位Windows Python。若旧虚拟环境失效，先报告现有解释器位置；不要把新电脑的D盘路径当作旧电脑事实。此后保持同一个PowerShell窗口，以保留上面的目录变量。

## 3 先做离线check

```powershell
& $labPython -X utf8 -m scripts.manual_leg_check check --output-dir (Join-Path $labRun 'check')
$labCheckExit = $LASTEXITCODE
Write-Host "check退出码：$labCheckExit"
```

check只检查本地Python、DLL文件和配置，不连接假腿。若check失败，停止设备步骤；保留终端输出以及它打印的结果目录，进入最后的数据回传步骤。DLL存在不等于TwinCAT运行环境、实际映射和设备通信已经通过。

## 4 静止状态下只做一次probe

确认假腿静止；不要使能、复位、松刹车、下载或激活TwinCAT工程，不改变控制模式。机械臂保持原状态，不执行轨迹。

```powershell
& $labPython -X utf8 -m scripts.manual_leg_check probe --live --output-dir $probeSession
$labProbeExit = $LASTEXITCODE
Write-Host "probe退出码：$labProbeExit"
Write-Host "probe结果：$probeSession"
```

probe会读取本机AMS、ADS服务状态、20次输入映像及toggle；既有守卫仍生效。看终端输出和该目录中的 `segment_status.json`、`manifest.json`。成功要求两个文件均为`status=COMPLETE`、`ok=true`，实际AMS一致且`toggle_changes>0`。失败时，新版本会保存最后原始映像、已解码字段及明确原因；`rows=0`不等于没有进行任何ADS读取。

**只要probe退出码非0或结果不是成功，今天就停在这里，直接做第7步回传。** 不重试到“成功”为止，不修改100N·m等限值、使能预期、PDO偏移或reviewed标志，不先让腿运动来试数据。

## 5 只有probe成功后才采10秒静止记录

保持相同电脑、配置和安装，在成功probe后10分钟内执行。采集入口会检查probe与当前配置、主机是否一致。不要把另一台电脑或旧日期的probe目录填进来。

```powershell
& $labPython -X utf8 -m scripts.manual_leg_check capture --live --duration-s 10 --probe-session $probeSession --output-dir $captureSession
$labCaptureExit = $LASTEXITCODE
Write-Host "采集退出码：$labCaptureExit"
Write-Host "原始记录：$captureSession"
```

失败也保留完整目录，停止后续设备动作。完成仅代表本次静止采集和关闭流程结束，不表示数据满足动力学、标定或运动要求。本次先做10秒，不自动追加30秒或进入完整轨迹。

## 6 对已有CSV生成离线质量报告

有leg.csv时才执行；即使采集失败且留下部分CSV，也可以分析，但必须保留其失败状态，不把部分记录当作完整试验。

```powershell
if (Test-Path -LiteralPath $labCsv -PathType Leaf) {
    & $labPython -X utf8 -m measurement_validation.offline_leg quality $labCsv --output-dir (Join-Path $labRun 'quality_10s')
    Write-Host "质量分析退出码：$LASTEXITCODE"
}
```

看 `summary.json`、单文件质量JSON及原始状态：实际行数/频率、主机接收长间隔、时间重复或倒序、故障和错误码、缺失字段、静止计数变化及力矩变化。报告生成成功不等于质量通过；主机时间缺口不是设备确切丢帧数。6077百分比因符号未验证而留空是已声明限制，不能因此忽略真正的3C1A力矩字段；没有换算关节角时保留未标定提示。

静止力矩只描述当前姿态、支撑和安装。不要自动去零，不把它解释为舒适度、患者力或完整带假腿运动效果。

## 7 成功或失败都只提交本次记录

保持设备停止采集后再操作Git。check、probe及可能生成的采集和质量报告均在本次目录内；原始CSV、状态和manifest一起保留。不要修改数值、删除失败文件或只选成功片段，不使用整个results目录的通配添加。

先核对本次目录和暂存区：

```powershell
Write-Host "准备回传：$labRun"
Get-ChildItem -LiteralPath $labRun -Recurse -File | Select-Object Length,FullName
git diff --cached --name-only
git status --short
```

若暂存区已有其他人的文件，先停止，不擅自取消或混合提交。确认本次只有实验记录，没有账号、密码、远控会话connection.json或令牌。若存在CSV，确认仓库对本次结果保留原始字节：

```powershell
if (Test-Path -LiteralPath $labCsv -PathType Leaf) {
    git check-attr text -- $labCsv
}
```

期望为`text: unset`，避免Git转换行尾导致记录哈希失配；若不是，先向协作者报告，不自行改写CSV。

仅添加本次目录，然后核对提交范围：

```powershell
git add -- $labRun
git diff --cached --stat
git diff --cached --name-only
```

确认只有 `results/manual_lab/本次时间/` 下的文件后，按实际结果修改提交说明并回传：

```powershell
git commit -m "Record stationary leg check 2026-11-14"
if ($LASTEXITCODE -ne 0) { throw '数据提交失败，保留文件并停止。' }
git push origin physics
if ($LASTEXITCODE -ne 0) { throw '上传失败，保留本地提交，不要强推。' }
git log -1 --oneline
```

如果push因远端更新被拒绝，不force、不reset；把本地提交号和错误告诉协作者。失败probe的数据同样有价值，提交说明可写 `Record failed stationary leg probe 2026-11-14`。

下一阶段完整实验顺序见[下午实验清单](TODAY_LAB_CHECKLIST_2026-11-14.md)。其中涉及空载或带假腿运动的步骤需要分别满足现场条件，本次静止检查流程不包含这些运动命令。
