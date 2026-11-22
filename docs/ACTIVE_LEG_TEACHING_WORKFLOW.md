# 主动假腿带腿示教：能记录什么、何时使用

2026-11-22。操作者反馈：主动施力条件下仍可手动带动关节，希望将其用于机械臂轨迹。当前仅准备参考位姿录制流程，没有运行或授权自动负载回放。

## 结论

可以把机械臂与小腿连接后，在已确认的安装条件下通过原厂拖动记录的**实际六维TCP位姿**作为参考几何。假腿处于主动状态不必然妨碍示教，但目标力矩、力矩环与限速会影响手感、运动形状和机器人所测负荷。“手能推”“单独给力矩能动”都不等于零力矩低阻力跟随、力控稳定性或整条路径已验收。

drive1动作时drive4模组整体跟着移动，也不能仅凭目视认定膝关节相对转动；需看相对关节位置变化与TCP记录。后续轨迹应采用连接状态下机械臂的实际TCP，不把两轴角度曲线直接当作机械臂空间轨迹。

本次刚性连接在靠近膝盖的小腿处，工具朝向必须随小腿自然运动并完整保留。现有`design_taught_trajectory.py`明确会丢弃原始朝向变化，现有自动执行器主要支持固定朝向；**本次六维带腿记录不能直接送入旧的XYZ固定朝向设计/回放链。** `loaded_trial`对刚性连接的自动运动阻止继续生效。

## 当前先保持停用，准备好以后再录

前一轮3C18=100引起主动运动后，最近一次明确确认是两轴停用、两轴目标力矩均读回0；不据此推断之后现场状态未改变。先保持支撑并核对实际状态，不以100Nm或其他未经验证的非零目标开始带腿示教，也不把−5Nm等有方向目标视为已经验收的拖动设置。

以下条件尚需现场确认，不能仅凭手推成功当作已通过：正常使能与抱闸释放稳定；松手时无非预期自行运动；膝伸直端边界与停止方式已在本次坐标下验证；实际工具/TCP、负载及刚性夹具可靠。当前3D00已被修改并重新上电，需先核对反馈与限位坐标，步骤见[膝伸直端限制](KNEE_EXTENSION_LIMIT_WORKFLOW.md)。

满足条件后的示教由操作者通过原厂拖动功能进行，允许位置与朝向随机构活动。先在已经确认的范围内做小幅闭合动作，不追原空载的大行程，也不为录满90秒而加速。两个系统的正常停止/急停应能由操作者实际控制；一人无法同时照看机械支撑和装置时，先只做静止建档。

## 录制前建档

使用[主动带腿示教记录单](templates/ACTIVE_LEG_TEACHING_RECORD.md)，逐项填写实际值与尚未核实项。力矩目标是目标，不是实测输出；同一记录中不改变参数。首次只记录参考几何，不作为论文负荷改善、人体舒适度或个性化效果样本。

录制器只读取机器人状态，不采集ADS假腿数据或机器人wrench，也不会使能、切换模式、控制关节或在结束时替操作者停车。现场填写的参数只是声明，不能证明全程未被主站修改。正式受力试验还需对应的实时参数/运动/负荷记录与审核。

## 条件通过后使用的只读录制入口

以下命令只用于上述条件已验收后的示教。当前没有自动回放、驱动使能或驱动参数写入命令。先妥善保存旧电脑已有记录与暂存变更，然后在`paper-personalized-load`仓库更新：

```powershell
git pull --ff-only origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '更新失败，保留本地记录。' }
$teachPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $teachPython -PathType Leaf)) { throw '找不到原有项目Python。' }
$teachParent = 'results/manual_lab/' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Path $teachParent -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath 'docs/templates/ACTIVE_LEG_TEACHING_RECORD.md' -Destination (Join-Path $teachParent 'teaching_setup.md') -ErrorAction Stop
$teachSession = Join-Path $teachParent 'connected_pose_teaching_01'
$teachPreview = Join-Path $teachParent 'connected_pose_preview_01'
```

先填完`teaching_setup.md`，在原厂界面准备好已经会用的拖动功能及支持状态；`$teachSession`必须尚不存在。录制器使用新配置`config/robot_pose_teaching_only.json`，只保留机器人连接与状态新鲜度参数，不混入被动假腿配置，也不是运动安全配置：

```powershell
& $teachPython -X utf8 -m scripts.record_rehab_trajectory --config config/robot_pose_teaching_only.json --session $teachSession --duration-s 90 --live
$teachExit = $LASTEXITCODE
Write-Host "录制退出码：$teachExit；目录：$teachSession"
```

看到`RECORDING`后起点稳定2～3秒，在已确认的小幅范围内缓慢引导一圈并回到起点，同时让工具自然转动；不够时间就另起新目录录更长原始段，不能赶动作。SDK连接可能影响厂商会话，连接后确认原厂拖动仍可用。Ctrl+C或录制结束只终止数据记录，不是机械臂或假腿停止命令；装置停止由操作者按原厂流程完成。

结束并使装置稳定后，离线预览：

```powershell
& $teachPython -X utf8 -m scripts.preview_recorded_trajectory --session $teachSession --output-dir $teachPreview
Write-Host "预览退出码：$LASTEXITCODE"
```

检查`trajectory_3d.png`、`orientation_time.png`、`recording_status.json`与预览报告中的朝向变化、闭合和缺口。朝向变化是刚性连接的重要信息，不把它削成固定朝向；本次不调用旧设计脚本。COMPLETE只表示录制结束，不代表能自动回放。

成功或失败都保留原始目录及现场记录；确认暂存内容后回传：

```powershell
git diff --cached --name-only
git status --short
git add -- $teachParent
git diff --cached --stat
```

确认只包含本次拟回传内容再执行：

```powershell
git commit -m "Record connected leg pose teaching and declared setup"
if ($LASTEXITCODE -ne 0) { throw '提交未完成，保留原目录。' }
git push origin paper-personalized-load
if ($LASTEXITCODE -ne 0) { throw '推送未完成，保留本地提交，不要强推。' }
```

后续交付是审核后的位姿曲线及相应执行/守卫支持，再逐步验证自动负载任务。正式比较需冻结机械条件、控制方式和目标力矩；主动关节的助力或阻力本身不能当作轨迹学习算法的负荷降低收益。
