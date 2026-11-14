# 原厂拖动示教轨迹录制与离线预览

更新日期2026年11月14日。本次只录下你认可的实际TCP运动形状，再看图核对；录制程序不生成运动目标，不回放，不缩放，不改变当前实验配置。现有约614mm上升、200mm水平移动的直线任务不适合作为你这次的示教替代，不再重复那条大行程空载轨迹。

本次空载离线复核见`results/manual_lab/20261114_174300/empty_full_tcp_20261114_174525/review/`。实测TCP相对计划的最大误差约1.539mm，当前问题主要是计划本身沿基座Y正向约200mm、Z正向约613mm的几何行程，并非先用同一条大行程反复排查跟随。另有14处力日志间隔超过50ms、最大约325ms，运动区间含192条无效快照，因此`EMPTY_LOAD_COMPLETE`不等于正式受力实验已合格。

成功力样本CSV不能完整反映异常查询。原`robot/wrench_query_events.jsonl`末尾`AUDIT_WRITE_SUMMARY`还记录14次原生查询超时预算、最大约302.6102ms；这是查询耗时，与约325ms的采样间隔是不同量。该审计也有非正常清理标记，后续需单独复核，不因进程退出码为0就认定受力链健康。

正式研究仍保留机器人位姿/受力学习：先冻结一条确认后的示教空间曲线，再研究节奏调整与机械负荷。**当前录制文件尚未接入真实轨迹回放和候选学习**，文件生成或预览成功都不能代替该接入工作。

## 本次程序与原厂拖动各做什么

由你在示教器/原厂界面进入已经会使用的原生拖动示教流程。本页不猜测具体菜单，不用项目SDK命令切换拖动模式，不增加上电、自动模式或机器人运动指令。

`scripts.record_rehab_trajectory`只连接并读取机器人状态，保存实际TCP六维位姿及可取得的关节状态、时间和有效性。它不读wrench，不访问TwinCAT/ADS，也不发上电、模式、拖动或运动目标。连接/断开属于厂商会话操作，不能保证厂商内部没有会话副作用；连接后仍应从示教器确认实际模式和装置状态。如果连接令原厂拖动不可用，保留记录，不用项目命令强行切模式。

本次只读状态，将示教形状采集与有长尾的受力查询链分开。它没有修复wrench采集问题；后续做受力学习前仍需复核原受力链的间隔、无效数据和同步质量。

拖动轨迹由你实际带动机械臂产生。TCP工具、负载和坐标应与示教器上的实际设置核对；同一条记录中不要改变工具/TCP定义。当前是否连接假腿、支承和绑带状态要如实写入现场说明，录制并不验证未知带腿安装是否适合运动。

CSV的`operation_state`是适配器缓存标签，不是每帧重新查询的拖动模式证明；原厂是否处于可拖动状态仍以示教器现场显示核对。程序不会为了刷新此标签增加模式切换或控制指令。

## 在旧电脑准备目录

在PowerShell进入原有仓库。若路径已经迁移，使用现有实际仓库，勿再创建另一套环境：

```powershell
Set-Location 'C:\Users\liumai\fjx\rehab_robot'
git branch --show-current
git status --short
git diff --cached --name-only
```

当前统一分支为`paper-personalized-load`。已有未提交记录先按[总手册回传步骤](NEXT_LAB_OPERATIONS.md)保存，不reset、clean或覆盖现场配置。工作区已妥善保存且允许正常快进时更新：

```powershell
git pull --ff-only origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '更新失败，保留现场文件，不继续。' }
$teachPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $teachPython -PathType Leaf)) { throw '找不到已有项目Python。' }
$teachParent = 'results/manual_lab/' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Path $teachParent -ErrorAction Stop | Out-Null
$teachSession = Join-Path $teachParent 'drag_teach_01'
$teachPreview = Join-Path $teachParent 'drag_teach_preview_01'
Write-Host "录制目录：$teachSession"
```

保持同一PowerShell窗口。`$teachSession`和`$teachPreview`都应是尚不存在的新目录，由各自程序创建。已经开始录制或生成预览的目录不复用；重录另起`drag_teach_02`及对应的新预览目录。

## 由你完成一条完整示教

先在原厂界面准备好拖动功能，核对当前TCP/工具、负载和坐标。将机器人放在你认可的起点，周围留出这次实际手动路径所需空间，停止手段保持可用。一人操作用下面的定时录制即可，不要求边拖边按键。

启动90秒录制：

```powershell
& $teachPython -X utf8 -m scripts.record_rehab_trajectory --config config/experiment.json --session $teachSession --duration-s 90 --live
$teachExit = $LASTEXITCODE
Write-Host "录制退出码：$teachExit；目录：$teachSession"
```

程序连接成功并启动状态记录后会打印`RECORDING ... seconds`。这表示采集循环已启动，不代表已经取得首个有效点或整条数据质量通过。没有预倒计时；从这行出现开始计时，不是在输入命令时开始。确认示教器上的原厂拖动仍可用后，按以下顺序慢慢完成：

1. 起点先保持约2–3秒，让记录包含稳定起始段。
2. 沿你希望以后机器人跟随的康复曲线慢速抬起；最高点和前后范围由你依据真实装置确认，不追现有0.9m绝对高度，也不为了录制填满行程。
3. 在最高点短暂停住约2–3秒，保留清楚的上抬/回程转折。
4. 沿你期望的回程回到起点。若期望回程与上抬不同，就按实际期望拖，不要求原路折返。
5. 回到起点后保持约2–3秒，再保持机器人静止，等待程序定时结束。不要为了赶90秒而加快或突然收回。

90秒不足以从容完成时，保留本次记录，在新目录使用更长定时重新录，例如180秒。参数支持不超过300秒；不要拼接两段记录冒充一条完整示教。

尽量保持你希望回放时采用的工具朝向。若示教中实际需要转动工具，正常记录这一变化并在现场说明中指出；不要为了“固定姿态”把实际旋转从数据中删掉。当前正式执行器采用固定姿态，含明显姿态变化的示教需要后续明确处理，不能静默忽略旋转或自动改成固定姿态。

## 结束状态与部分记录

定时结束且状态采集/清理正常时，`recording_status.json`为`COMPLETE`。这只表示定时记录完整结束，不证明你完成了整条示教、数据质量已达标或路径可回放。`ready_for_replay`始终为false。

Ctrl+C会保留已经录下的部分数据，状态为`CANCELLED`、`partial=true`；读取错误或清理失败为`FAILED`、`partial=true`。部分记录可用于诊断和看已经发生的运动，但不能改成完成，也不能当作完整回合。若现场需要停止，先按原厂流程稳定/停止机器人；Ctrl+C是停止记录，不是机械臂急停。

录制目录保存：

| 文件 | 用途 |
|---|---|
| `recorded_tcp.csv` | 实际状态/位姿、时间及有效性原始记录 |
| `recording_status.json` | COMPLETE/CANCELLED/FAILED、样本与清理状态、partial和不可回放标记 |
| `recording_manifest.json` | 配置、源码/Git及记录文件的SHA和证据范围 |
| `config_snapshot.json` | 录制当时的配置副本；不会改原配置 |

查看状态：

```powershell
Get-Content -LiteralPath (Join-Path $teachSession 'recording_status.json') -Raw
```

不要编辑CSV、覆盖配置快照或重写SHA来“修好”记录。完整原始目录连同失败原因保留，后续处理另存结果。

## 录后纯离线预览

机器人已经静止后，可在同一电脑直接生成离线预览。该命令不连接SDK或设备，不访问ADS，不发运动：

```powershell
& $teachPython -X utf8 -m scripts.preview_recorded_trajectory --session $teachSession --output-dir $teachPreview
$teachPreviewExit = $LASTEXITCODE
Write-Host "预览退出码：$teachPreviewExit；目录：$teachPreview"
```

预览用于核对空间路径、坐标范围、高度变化、回程与闭合情况、时间变化、采样有效性和工具姿态变化。先看这是否就是你刚才拖出的形状，再判断是否符合希望的康复动作。诊断或部分记录仍应按实际状态解释，不能因生成了图就视为完整可执行轨迹。

预览目录包含：

| 文件 | 查看内容 |
|---|---|
| `trajectory_3d.png` | 基座绝对路径与相对起点路径，按相同物理比例展示 |
| `projections.png` | 空间平面投影，检查前后方向、弧度和回程 |
| `xyz_time.png` | 三轴位置随时间变化、停顿与运动阶段 |
| `orientation_time.png` | 工具姿态变化，不能只看位置忽略转动 |
| `closure_review.png` | 起终点闭合差及回程检查 |
| `summary.json` | 源文件身份、记录/预览状态、范围和诊断标记 |

`FAILED/CANCELLED`或含无效行的记录若至少有两个有效点，仍可生成诊断图，预览状态为`PARTIAL`。源文件SHA不匹配、有效样本时钟重复/乱序或有效坐标非有限时，预览拒绝，仅保存`FAILED`摘要。最高Z只作为人工查看的候选转折点；间隔、闭合和姿态变化标记是筛查，不自动批准切分或回放。

预览继续保持`ready_for_replay=false`。它不自动平滑成另一条曲线，不拉伸到旧0.9m目标，不缩放到旧200mm水平位移，也不覆盖`config/experiment.json`。以后确需去抖、重定时或姿态处理时，要保留原始示教和单独的新版本，明确处理带来的偏差。

## 回传什么与下一步

把整个`$teachParent`目录回传，包含录制、预览及失败记录。另写几行现场说明：原厂拖动是否正常、是否连接假腿、实际TCP/工具/负载、示教器坐标表达、希望的上抬和回程、哪里停顿、是否有转动工具、是否完成回起点。照片或装置说明可放同一父目录，不需边拖边记录文字。

先看暂存区没有其他内容，再只添加这次父目录：

```powershell
git diff --cached --name-only
git status --short
git add -- $teachParent
if ($LASTEXITCODE -ne 0) { throw '录制与预览目录暂存失败。' }
git diff --cached --stat
git diff --cached --name-only
```

确认只有这次录制父目录，再提交推送；失败/取消的原始记录也保留：

```powershell
git commit -m "Record OEM drag teaching and offline preview"
if ($LASTEXITCODE -ne 0) { throw '记录提交失败，保留数据。' }
git push origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '回传失败，保留本地提交，不要强推。' }
git log -1 --oneline
```

更多Git冲突/分支情况见[总手册](NEXT_LAB_OPERATIONS.md)。不把未审阅配置或其他暂存代码混进数据提交，本页不自动发送、推送或重放数据。

下一步先根据真实记录确认路径，再单独完成：示教TCP与执行TCP的一致性、完整性/闭合/速度加速度审查、姿态策略、显式路径来源与哈希绑定，以及受限的空载回放准备。当前只录状态，没有受力学习结果，也没有录制后自动执行能力。

论文后续可以将这条示教空间shape冻结，在同空间任务与固定总周期下比较节奏，由机器人实测受力进行轮间候选选择。示教数据属于任务来源/诊断证据，不能写成个性化优化收益或舒适度改善。空间shape由人确认与学习端点采用机器人受力相互独立，选择TCP学习并不要求空间轨迹是直线。

## 本次软件验证

新增录制、预览和闭合检查后，完整离线核心集合557项通过，耗时102.10秒，无失败、错误或跳过；[测试报告](validation/PAPER_BRANCH_TESTS_2026-11-14_DRAG_TEACHING.xml)为本次独立记录。覆盖只允许状态接口、重复帧/过期/读错、取消和清理失败、来源哈希、原始录制至预览衔接、单位/时间/有效性核对、姿态与闭合，以及图片写入失败不能报告成功。七段PowerShell命令仅静态解析，未执行设备动作。

五类预览图使用明确标注的合成数据检查布局，示例没有进入实测结果。本次原始空载数据另经纯离线重算；没有连接机器人、切换拖动或执行新运动，也尚未取得真实拖动示教记录。
