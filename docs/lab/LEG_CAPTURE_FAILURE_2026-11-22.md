# 11月22日假腿静止采集失败：核对与复测

## 已确认的错误

回传提交`4e188ae8`，会话`results/manual_lab/20261122_152201/connected_static10s/`。运行代码为`16098e5b`。本次没有发送运动命令。

假腿不是完全读不到：已写入219个接受帧，从北京时间15:22:22.420996到15:22:22.899273，覆盖0.478275秒，接受行速率约455.805Hz。这不是完整十秒记录，也不是精确设备帧率。

首要失败为`leg_capture_failed:TimeoutError: input toggle stopped changing`。采集程序超过100毫秒没有观察到配置的一位刷新标记变化，因此停止；之后主程序关闭机器人观测。失败时刻本身没有记录，不能从CSV末行时间反推出精确异常时刻。

| 核对项 | 结果 |
|---|---|
| 四轴状态字 | 全部记录均为`0x0231` |
| 四轴错误码 | 全部记录均为`0x0000` |
| fault、operation_enabled | 均为false |
| 髋部原始力矩范围 | 约−4.980～−4.541Nm |
| 膝部原始力矩范围 | 约0.079～0.317Nm |
| 已接受帧的ADS查询耗时 | 中位1.0364ms，最大2.0732ms |
| 已接受帧最大接收间隔 | 46.1317ms |
| 机器人状态 | 419条均有效 |
| 机器人力查询 | 两次成功，最大原生耗时1.0007ms；随后正常退出、正常清理 |

报告中的“有效对齐快照0/168”不表示机械臂完全读不到：这168条快照全部早于第一条受力数据，随后整次观测因假腿失败而取消。机器人受力子进程派发到进程启动另耗3.140925秒；现有记录不能据此认定它导致了假腿超时。

## 目前能解释到哪里

原采集CSV只保存刷新位变化后的帧。最后一帧之后发生了多少次读取、读取时映像是否变化、主机是否暂停、发送进度是否阻塞，均未保存。因此不能把错误直接解释为网线断开、驱动故障或TwinCAT停止。

一位0/1标记存在采样混叠：例如真实源每2ms翻转，但每4ms读取时可能一直看到同一个值。离线健康源复现会在约104ms报同样超时。这证明检测存在局限，不能证明本次就是混叠；改变其他PDO字节也不能直接当作正式500Hz新鲜度证明。

另一个已确认的软件问题是：原生产采集循环同步向主程序发送进度；若主程序暂时不能读取管道，输出会阻塞实际ADS采集。新代码已将进度发送移到独立线程，最多保留最新待发进度及最终状态。只替换过期进度消息，所有CSV帧、原始采样时间和故障检查仍保留；最终状态需要在有限时间内送完，发送或关闭失败仍判失败。离线阻塞输出测试验证了600帧仍完整采集并关闭。该缺陷是不是本次的实际诱因，仍需现场复测。

本次没有修改100ms无刷新停止阈值、驱动状态/力矩保护、轮询间隔、PDO偏移或500Hz源声明，也没有复位、使能或重启设备。下一次同时保存：

- `leg/leg_capture_diagnostics.json`：最后256次读取的原始映像、每次查询起止、轮询间隔、刷新位、是否接受、重复刷新位时映像是否变化、错误和Python版本。
- `leg/progress_delivery_diagnostics.json`：发送耗时、进度合并计数、最终状态是否送完及关闭情况。
- 原有`leg.csv`、`capture_stderr.log`和明确的失败状态。诊断记录在采集结束后写盘，不在每次ADS读取中增加磁盘写入。

## 现在只做一次独立假腿复测

先更新旧电脑仓库。已有未提交实验记录先保存，其他代码/暂存变更先核对，不reset、clean、覆盖或强推。

```powershell
Set-Location 'C:\Users\liumai\fjx\rehab_robot'
git branch --show-current
git status --short
git diff --cached --name-only
```

确认分支为`paper-personalized-load`且现场变更已妥善保存后，在同一窗口：

```powershell
git pull --ff-only origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '更新失败，保留本地记录。' }
$legPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $legPython -PathType Leaf)) { throw '找不到原有项目Python。' }
$legConfig = 'config/continuous_taught_90s.loaded_diagnostic.json'
$legParent = 'results/manual_lab/' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$legOnly = Join-Path $legParent 'leg_only10s'
& $legPython -X utf8 -m scripts.diagnose_leg_capture --config $legConfig --session $legOnly --duration-s 10 --live
$legOnlyExit = $LASTEXITCODE
Write-Host "退出码：$legOnlyExit；完整目录：$legOnly"
Get-Content -LiteralPath (Join-Path $legOnly 'report.md') -Raw -Encoding UTF8
```

保持假腿静止、原有支撑和关节状态，不进行拖动或运动。新入口使用生产采集器与相同配置，只采假腿，不连接机械臂，也无需先做probe。程序会自己创建新目录，不提前创建`$legOnly`。十秒从子进程采集就绪后开始计时，启动和关闭另计；若先失败则保存实际较短记录。

**成功或失败都先回传这一个新目录。** 本次只需要它来定位，不连续反复运行联合观测。`CAPTURE_COMPLETE`只证明这一段生产采集正常关闭，仍需审核间隔和原始数据，不等于已达到论文所需的500Hz质量。

```powershell
git diff --cached --name-only
git status --short
git add -- $legParent
git diff --cached --stat
```

确认暂存只包含本次拟回传内容，再提交推送：

```powershell
git commit -m "Record standalone leg capture diagnostic"
if ($LASTEXITCODE -ne 0) { throw '提交未完成，保留原目录。' }
git push origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '推送未完成，保留本地提交，不要强推。' }
```

若独立采集通过，下一步才在新目录对照联合观测，检查是否仍只在机器人启动时失去刷新。若独立采集也失败，优先看实际轮询间隔、同位映像变化和发送阻塞证据，再核对刷新信号含义或已有设备计数器；不猜偏移、不用变化字节顶替刷新条件，也不直接放宽超时。

当前刚性连接的自动负载运动限制继续生效，流程见[带腿操作说明](../LOADED_DIAGNOSTIC_WORKFLOW.md)。
