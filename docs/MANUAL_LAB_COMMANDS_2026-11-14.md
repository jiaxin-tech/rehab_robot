# 2026年11月14日旧电脑静止检查与数据回传

本次在装有TwinCAT的旧实验电脑本地运行。16:43回传的一帧按当前配置解码为轴1状态字`0x0218`、错误码`0x0022`，程序因此停止；但现场报告TwinCAT界面没有报错。该帧不足以排除旧映像或映射不一致，不能直接断言驱动器物理故障。**本次先执行下方inspect核对并回传，不直接重跑probe/capture。** 具体复核见[本次记录分析](lab/LEG_PROBE_REVIEW_2026-11-14.md)。

**17:03的inspect已经回传并核验，无须重复同一检查。** 20帧均不同，toggle变化9次，轴1始终为`0x0218 / 0x0022`。本次下一步是对照TwinCAT同一物理驱动器的CoE-Online `0x6041:00`和`0x603F:00`，并记录设备名称/型号；界面主状态和这两个驱动器对象应分开核对。见[20帧复核](lab/LEG_INSPECT_REVIEW_2026-11-14.md)。以下inspect命令保留供复现，本次先不重跑或进入采集。

## 本次先执行：连续映像与配置文件核对

在原仓库PowerShell中，先核对当前分支为`paper-personalized-load`、暂存区及工作区已干净，再执行：

```powershell
git pull --ff-only origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '更新失败，保留原状并停止。' }
$legInspect = 'results/manual_lab/' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_inspect'
& .\.venv\Scripts\python.exe -X utf8 -m scripts.manual_leg_check inspect --live --output-dir $legInspect
$legInspectExit = $LASTEXITCODE
Write-Host "核对退出码：$legInspectExit；本次目录：$legInspect"
```

inspect固定读取20次，在`inspection_frames.json`保留每帧原始字节、时间、解码和守卫判断。它统计映像变化与toggle，不因观察到的故障位提前停止；原生读取或关闭错误仍结束本次诊断。`COMPLETE/ok=true`只表示20次读取完成，**不代表驱动健康、映射正确或可以采集**。inspect目录不能代替成功probe用于采集放行。

同时保存`twincat_file_evidence.json`：只读`C:/TwinCAT/3.1/Boot/CurrentConfig.xml`的哈希、修改时间、可识别设备身份及PDO字段原字符串；文件缺失或解析失败也保留状态。启动文件不是在线映射证明，程序不激活工程、不修改配置、不猜设备型号或偏移。

本次到这里结束。核对暂存区没有其他内容后，只提交刚打印的目录：

```powershell
git diff --cached --name-only
git add -- $legInspect
git diff --cached --stat
git commit -m "Record read-only TwinCAT image inspection 2026-11-14"
if ($LASTEXITCODE -ne 0) { throw '提交失败，保留原始记录并停止。' }
git push origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '上传失败，保留本地提交，不要强推。' }
git log -1 --oneline
```

以下原check/probe/capture步骤保留供原因查明后使用，本次不接着执行。

本流程使用`paper-personalized-load`分支及统一配置`config/experiment.json`中的`leg`部分，不依赖旧独立配置、临时远程助手或远程登录。ADS Net ID是AMS路由身份，不是普通IPv4；临时网卡地址`192.168.137.2`不能代替AMS。端口11返回的ADS状态0只是该查询服务的原始值，不能单独证明TwinCAT全局处于CONFIG，也不据此切换运行状态。当前DLL路径来自统一配置，为`C:\TwinCAT\AdsApi\TcAdsDll\x64\TcAdsDll.dll`。

以下命令在旧电脑PowerShell中逐块执行。每块先查看结果，再执行下一块，不要整页一次粘贴。本流程没有机器人或假腿运动命令。

## 1 核对仓库并更新论文分支

历史确认的旧仓库路径如下；如果已迁移，先进入实际已有仓库，不创建空目录代替。

```powershell
Set-Location 'C:\Users\liumai\fjx\rehab_robot'
git status --short
git branch --show-current
git remote -v
git diff --cached --name-only
```

若存在本地代码、配置、未保存数据或已有暂存内容，先保留原状并告知协作者。不要reset、clean、自动stash或覆盖文件，也不要把其他人的改动混入本次数据提交。

工作区和暂存区干净后，逐条更新：

```powershell
git fetch origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '获取论文分支失败，停止。' }
git switch paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '切换论文分支失败，停止。' }
git merge --ff-only origin/paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '分支不能直接前进，保留原状并停止。' }
git status --short
git branch --show-current
```

若分支不存在、发生分叉或更新失败，把错误交给协作者处理，不临时建立另一条同名研究分支或强推。更新完成后应能找到`scripts/manual_leg_check.py`、`config/experiment.json`和本说明；当前分支应为`paper-personalized-load`。

## 2 选择已有Python并建立本次目录

优先使用旧仓库已有虚拟环境，不需要激活，也不在本流程安装依赖。

```powershell
$labPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labPython -PathType Leaf)) { throw '找不到已有项目Python，停止并核对实际解释器路径。' }
& $labPython -c "import sys,struct; print(sys.executable); print(sys.version); print('bits=',struct.calcsize('P')*8)"
if ($LASTEXITCODE -ne 0) { throw 'Python启动失败，停止。' }

$labConfig = 'config/experiment.json'
$labRun = 'results/manual_lab/' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Path $labRun -ErrorAction Stop | Out-Null
$probeSession = Join-Path $labRun 'probe'
$captureSession = Join-Path $labRun 'capture_10s'
$labCsv = Join-Path $captureSession 'leg.csv'
Write-Host "本次结果目录：$labRun"
```

ADS DLL需要64位Windows Python。旧环境失效时先报告已有解释器位置，不把新电脑的D盘路径当作旧电脑事实。后续保持同一个PowerShell窗口及仓库工作目录，以保留变量。

## 3 先做离线check

```powershell
& $labPython -X utf8 -m scripts.manual_leg_check check --offline --config $labConfig --output-dir (Join-Path $labRun 'check')
$labCheckExit = $LASTEXITCODE
Write-Host "check退出码：$labCheckExit"
```

`check`默认即为离线；加`--offline`便于现场确认。它只检查本地Python、DLL文件存在性和配置，不加载ADS DLL、不打开ADS、不访问设备。失败后保留终端输出和结果目录，直接进入第7步。DLL存在不代表实际TwinCAT映射或设备通信通过。

## 4 静止状态下只做一次probe

确认假腿静止，机械臂保持原状态。不要使能、复位、松刹车、下载或激活TwinCAT工程，不改变控制模式，不执行轨迹。实际I/O映射仍需通过现场XAE在线信息独立核对，配置哈希不能代替在线映射证据。

```powershell
& $labPython -X utf8 -m scripts.manual_leg_check probe --live --config $labConfig --output-dir $probeSession
$labProbeExit = $LASTEXITCODE
Write-Host "probe退出码：$labProbeExit"
Write-Host "probe结果：$probeSession"
```

probe核对本机实际AMS与配置一致，读取ADS服务原始状态，并尝试20次输入映像读取，观察toggle变化；既有轴1/轴4守卫保持生效，遇到首个失败立即结束。成功要求`segment_status.json`、`manifest.json`均为`status=COMPLETE`、`ok=true`，完成20次读取且`toggle_changes>0`。

失败目录保存最后成功返回的原始映像及可解码字段，守卫失败会列出具体轴、实际值和配置预期；原生读取失败另存返回码、报告字节数及原始缓冲区，不把失败缓冲区当作有效帧解码。`attempted_reads`、`completed_reads`与`rows`含义不同，`rows=0`不代表未进行ADS读取。探测频率只是诊断节奏，不是500Hz连续采集证明。

**probe退出码非0或结果不是成功时，停在这里，直接做第7步回传。** 不重试到“成功”为止，不修改100N·m守卫、使能预期、PDO偏移或地址来迁就报错，不让腿先运动来试数据。

## 5 probe成功后才采十秒静止记录

保持同一电脑、同一配置、同一诊断代码和安装，在成功probe后十分钟内执行。入口还会检查主机、配置与代码哈希、实际AMS、probe完整性和状态文件哈希，不能借用另一台电脑或旧日期的probe。

```powershell
& $labPython -X utf8 -m scripts.manual_leg_check capture --live --duration-s 10 --probe-session $probeSession --config $labConfig --output-dir $captureSession
$labCaptureExit = $LASTEXITCODE
Write-Host "采集退出码：$labCaptureExit"
Write-Host "原始记录：$captureSession"
```

失败也保存完整目录并停止后续设备动作。成功只代表本次静止采集和关闭流程结束，不表示数据满足动力学、标定或运动要求。本次只做十秒，不自动追加三十秒或进入完整轨迹。CSV保留查询起止、接收时刻与UTC标签；旧电脑的本机单调时钟不能与另一台机器人的本机时钟直接拼接为硬件同步记录。

## 6 对已有CSV生成离线质量报告

有`leg.csv`时才执行；失败留下的部分CSV也可以检查，但要与失败状态一起保存，不能称为完整试验。使用当前分支实际保留的质量检查入口：

```powershell
if (Test-Path -LiteralPath $labCsv -PathType Leaf) {
    & $labPython -X utf8 -m measurement_validation.leg_episode $labCsv --minimum-rate-hz 500 --output (Join-Path $labRun 'quality_10s.json')
    Write-Host "质量报告退出码：$LASTEXITCODE"
}
```

查看`quality_10s.json`的实际行数/频率、时长、接收间隔、时间重复或倒序、故障/错误码、缺失字段、计数分布和原始力矩分布。`--minimum-rate-hz 500`以当前500Hz声明作严格筛查；报告生成和退出码0不等于质量通过，应看`findings`与`usable_for_dynamics`。空CSV或格式错误可能无法生成报告，保留终端错误和原文件即可，不修剪数据以消除错误。

主机时间缺口不是设备确切丢帧数。同步标签只描述采集方法，未独立证明设备内物理同时性。6077百分比因符号未验证而留空是已声明限制，不能因此忽略3C1A原始力矩；未传入关节标定时，保留未标定提示。

静止力矩只描述当前姿态、支撑和安装。不要自动去零，也不能据此声称舒适度、患者力或完整带假腿运动效果。

## 7 成功或失败都只提交本次记录

采集关闭后再操作Git。check、probe及可能生成的采集和质量报告均在本次目录内；原始CSV、状态、配置快照和manifest一起保留，不修改数值、删除失败文件或只选成功片段。

先核对本次目录和暂存区：

```powershell
Write-Host "准备回传：$labRun"
Get-ChildItem -LiteralPath $labRun -Recurse -File | Select-Object Length,FullName
git diff --cached --name-only
git status --short
```

若暂存区已有其他人的文件，先停止，不擅自取消或混合提交。确认本次记录没有账号、密码、临时远控连接文件或令牌。若存在CSV，确认仓库保留本次结果的原始字节：

```powershell
if (Test-Path -LiteralPath $labCsv -PathType Leaf) {
    git check-attr text -- $labCsv
}
```

期望为`text: unset`；否则先向协作者报告，不自行改写CSV或改行尾来匹配哈希。当前分支只放行`results/manual_lab/`和`results/paper_study/`中的新结果，其他旧`results/`仍被忽略。

只添加本次精确目录，不添加整个results目录：

```powershell
git add -- $labRun
if ($LASTEXITCODE -ne 0) { throw '本次数据暂存失败，保留原文件并停止。' }
git diff --cached --stat
git diff --cached --name-only
```

确认暂存区只有`results/manual_lab/本次时间/`下的文件后，按真实结果修改说明并回传：

```powershell
git commit -m "Record stationary leg check 2026-11-14"
if ($LASTEXITCODE -ne 0) { throw '数据提交失败，保留文件并停止。' }
git push origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '上传失败，保留本地提交，不要强推。' }
git log -1 --oneline
```

push因远端更新被拒绝时，不force、不reset；把本地提交号与错误告诉协作者。失败probe同样有价值，提交说明可以写`Record failed stationary leg probe 2026-11-14`。

论文路线和后续实验安排见[当前工作流](CURRENT_WORKFLOW.md)。运动、带假腿连接和完整回合分别按正式流程审核，本次静止检查不包含这些动作。
