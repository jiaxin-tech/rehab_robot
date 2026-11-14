# 2026年11月14日下午实验室操作清单

今天用3至4小时推进真实采集与运动链，优先带回可复核的数据。目标按顺序为：重新确认静止采集和测角条件、复测机器人状态与力采集、有条件时完成一个完整空载回合，再决定是否具备连接假腿的条件。当前没有完整带假腿动态回合，不能把历史空载数据当作假腿训练结果。

所有命令在实验电脑的项目根目录执行，统一入口为 `scripts/lab_next_visit.ps1`。默认Check只查文件和配置；带 `-Live` 的动作会访问真实设备，必须有现场操作者监督。连接和断开仍可能有厂家会话副作用；“不发送轨迹”不等于设备连接绝无副作用。下述命令是操作清单，本次代码审查没有执行它们。

## 先留20分钟检查环境与安装

```powershell
Set-Location D:\rehab_robot
.\scripts\lab_next_visit.ps1 -Action Check
```

保存终端显示的新目录，阅读其中 `session_status.json`、`session_report.md` 和 `blockers.json`。Check报阻断是当前缺项，不是让你将reviewed或passed直接改成true。Check完成也不是现场运动放行。

拍照并记录当前机器人、工具/工件、连接件、假腿是否连接、测角工具固定方式与零角参考。核对现场网卡、设备身份和当前安装，历史IP仅供对照。若项目缺Python、SDK、ADS DLL或路径失效，先恢复实验电脑对应依赖；不要把缺文件误当通信故障。若PowerShell阻止脚本，使用文末Python入口，不修改系统执行策略。

今天先记录清楚假腿到底支持哪些实际工况：能否被动回拖，是否能设置髋/膝目标力矩、刚度或阻尼，参数在哪里设置、单位和模式是什么、怎样确认生效。当前Python采集器只有ADS读取功能；仿真中的K/B不是已经实现的硬件设置接口。尚未确认的能力直接写未知。

## 接着用30至40分钟复测ADS静止采集

先保持假腿处于允许的静止姿态，机器人不执行轨迹。当前ADS配置要求轴1/4未使能；如果现场状态不符，应停下来按现场规程核对，不让采集脚本替你切换状态。

```powershell
.\scripts\lab_next_visit.ps1 -Action Capture -Live -Label static_10s -DurationSeconds 10
.\scripts\lab_next_visit.ps1 -Action Capture -Live -Label static_30s -DurationSeconds 30
```

每条命令都会新建目录。先读第一段结果，再决定是否执行第二段，不一次粘贴批量运行。记录 `leg/segment_status.json`、`leg/quality.json`、原始 `leg/leg.csv` 与日志。

必须分清“记录完整”和“数据适合动力学”：CAPTURE_COMPLETE只说明采集程序正常结束。检查实际行数、频率、长间隔、重复或倒序时间、故障码、力矩缺失和采集退出状态。历史频率约487至488Hz，但最长间隔约39ms，平均频率不能掩盖长缺口。原始CSV没有换算后的关节角时，质量报告会指出尚无关节角标定字段，这是原始通道的已知限制，不应通过伪造角度列消除。

若10秒已发生通信、故障、超限或清理失败，保留全目录并停止本轮设备动作。若只有时间质量问题，今天可继续独立的静态测角记录和离线分析，但不据此进入动态训练。需要观察长尾时，可在前两段和现场条件均允许后另录240秒：

```powershell
.\scripts\lab_next_visit.ps1 -Action Capture -Live -Label stability_240s -DurationSeconds 240
```

## 用40至60分钟补独立测角验证

当前膝扫描覆盖0至60度，而既有参考膝最高约125度，约64.83%的参考时间高于60度。九点扫描拟合内误差约1.08度，但未参与拟合的四点RMSE约7.16度、最大约12.22度。今天优先弄清独立测角和安装重复性，不再以同次拟合内误差代替验证。

固定手机或测角工具的安装与参考线。每一点都记录实际读数、大腿是否移动、方向、安装编号、文件名及稳定区间；膝角是小腿相对大腿的角度，不能直接用小腿相对地面的角度代替。只在现场允许的人工调整方式和范围内摆位，不强行追求某个高角度。

先用预定拟合点冻结公式，再采未参与拟合的2至3个测试姿态；至少重新放置测角工具或重新装夹后复测一组。若现场允许验证更高角度，覆盖准备执行的轨迹范围；若做不到，记录未覆盖范围，另行设计和审核较小范围轨迹，不能直接改写冻结参考。

```powershell
# 每次先人工摆位、稳定并独立测角，再单独运行一条；Label写实际点名
.\scripts\lab_next_visit.ps1 -Action Capture -Live -Label knee_validation_A -DurationSeconds 10
```

按 `measurement_validation/OFFLINE_WORKFLOW.md` 建立本次新的points.json，写入实际角度与原始文件相对路径。fit、validation、return的用途在分析前确定，不能挑好看的时段或修改实际角度标签。离线计算入口：

```powershell
.\scripts\lab_next_visit.ps1 -Action Calibrate -Data results/next_lab_visit -Manifest results/next_lab_visit/points_20261114.json
```

这只生成分析结果，不发布新运动标定。15度是历史工程容差，不能因此宣称测角高精度或小幅负荷差异已可信。

## 用30至40分钟分别复测机器人静止诊断

机器人与工具静止、无人接触，操作员监督。先核对当前配置与安装，再单条执行：

```powershell
.\scripts\lab_next_visit.ps1 -Action Events -Live
.\scripts\lab_next_visit.ps1 -Action Observe -Live
```

Events只观察厂家安全事件，不主动制造碰撞。未收到可信初始状态就保留未知，不能当作无碰撞。Observe使用未准备RT的静止采集配置，不会替你上电或切换模式；若现场电源/模式不满足，应停止并核对，不能改配置冒充匹配。

历史力查询出现过240至312ms长阻塞，空载第四次还出现过“查询很快，但结果进入可用日志链时已陈旧”。本次检查状态年龄、力年龄、skew、查询/审计耗时、故障锁存和清理确认，不能只看平均查询耗时。SDK263、259、数据陈旧、ACK超时或关闭失败均保留原始文件。

这两步与ADS是顺序诊断，不能证明机器人与ADS同轮动态对齐。没有可靠的跨设备延迟测量，就记录未知，不把延迟填0。

## 剩余40至60分钟按条件选择一个运动分支

今天以一个完整空载回合为优先运动目标。若前面的采集、安装或现场放行条件尚未满足，今天停在诊断与标定并带回数据；这仍是有效交付。

空载分支仅用于无假腿、无人体的夹具空载联调。`config/lab_next_visit.json`中的reference、anchor和review_request仍引用10月18日工件，不能把它们默认当作今天的安装/起点。必须由现场负责人核对或按原流程重建当前起点、预览、负载、配置和审核请求，再生成本次短有效期发布。不要只输入旧确认文字就复用旧安装说明。

```powershell
# 仅在当前空载审核请求及其全部绑定输入已核对后，现场人工签认
.\scripts\lab_next_visit.ps1 -Action Release
# 用上一步实际打印的本次release路径；此步会准备RT并实际运动一次
.\scripts\lab_next_visit.ps1 -Action Empty -Live -Release results/next_lab_visit/本次目录/release.json
```

发布、起点、预检任一失败就不运行Empty。运行失败后保留目录、清理结果与一次性使用凭据，不能删除锁或核销记录重用发布。完整回合需要记录最终完成原因、命令发送、状态/力数据和原生停止/断连确认，再由现场核对实际设备状态；零命令行或程序退出不能独立证明物理下电。

只有空载链和带假腿安装、标定覆盖、采集、对齐、现场发布都分别具备后，才考虑带假腿分支：

```powershell
# 使用当前现场评审配置；默认dummy_leg_capture_only只采集，不训练
.\scripts\lab_next_visit.ps1 -Action Round -Live
```

入口会先检查、读取起点并生成预览，再要求现场确认，最后最多执行一轮。当前默认应得到 `CAPTURE_ONLY_COMPLETE`、`training_status=SKIPPED_CAPTURE_ONLY`，并有原始CSV、dispatch.csv、measured.csv和measured_metrics.json；不能把只生成预览或原始部分CSV当作完成。若现场配置确实为已评审的dummy_leg_reference，才要求ROUND_COMPLETE与MODEL_UPDATED。不要为了赶下午进度将capture_only改成训练。

## 离开前留20分钟

带回所有新会话目录、配置快照、安装照片、实际角度表与失败日志。记录今天是否连接假腿、是否发送轨迹、完成了几次完整回合、当前设备最终状态，以及下一步最早失败的位置。空载与带假腿目录明确分开；不把人工姿态标定算成机器人带载运动。

今天采集的力、力矩和平滑性只能作为舒适相关机械指标的准备资料。假腿不能给出主观舒适度；今天也不据此声称患者获益或个性化有效。个性化实验后续仍需固定ROM和总时长、不同真实机械配置、共同轨迹基线及独立重复。

Python备用入口与同一助手完全对应，例如：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.lab_session check
.\.venv\Scripts\python.exe -X utf8 -m scripts.lab_session capture --live --label static_10s --duration-s 10
.\.venv\Scripts\python.exe -X utf8 -m scripts.lab_session events --live
.\.venv\Scripts\python.exe -X utf8 -m scripts.lab_session observe --live
```
