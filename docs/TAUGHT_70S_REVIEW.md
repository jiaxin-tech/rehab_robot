# 完整70秒示教空载方案

本页保留70秒历史方案及当时离线核验。该方案已于回传提交`130aa452`执行，实测与受力质量见[回传复核](lab/EMPTY_70S_REVIEW_2026-11-15.md)。用户随后将当前任务改为[连续90秒方案](CONTINUOUS_90S_WORKFLOW.md)；本页下方“尚无执行记录”和操作命令均是旧版本当时状态，不作为当前现场入口。

2026年11月15日，用户明确确认：“接受，保留幅度，按70秒继续”。确认范围包括本份空载配置的速度上限0.13米/秒、加速度上限0.45米/秒²。它是离线配置决定，尚无本方案真机执行或带载验收记录。

当前配置为`config/taught_tcp_70s.empty.json`，离线设计与核验归档为`results/manual_lab/20261115_taught_70s_approved_v1/`。固定工具朝向、XYZ幅度参数1和三维空间路径均保留。配置相对旧`config/taught_tcp_70s.review.json`仅修改empty_load的两个速度/加速度字段；原现场`config/experiment.json`与旧425秒基线保留。

| 阶段 | 时间区间，秒 | 时长，秒 |
|---|---:|---:|
| 起点保持 | 0～2 | 2 |
| 抬升 | 2～24 | 22 |
| 高位前伸 | 24～41 | 17 |
| 远端保持 | 41～43 | 2 |
| 回收下降 | 43～68 | 25 |
| 终点保持 | 68～70 | 2 |

三运动段平缓起停，段端速度和加速度为零。最高点停速但不额外保持。处理后路径约3.029米，抬升最高点、前伸最远点及闭合起点与原425秒版本一致。

## 离线核验

密集解析导数和全部样条节点核验：峰值速度约0.123829米/秒，加速度约0.430314米/秒²；乘原1.01余量后分别约0.125068和0.434617，低于已确认上限0.13和0.45。独立更密集网格复核一致。完整周期锁定70秒；检查、准备、运行分别核对周期和限值，冲突时拒绝，不自动延长。

配置SHA为`8550c0f33b8f8442d13d4b73d956a7bc13c5056cb986de001fa4fa9aad06aff6`。新预览复制的几何与原草案逐字节相同，SHA为`f565ad65aa7c4b5106bd9b5f634fd4197f31272cca2609d84cf049a8af25994a`。执行配置引用原不可变几何，批准版预览保存同字节副本；两者位姿、节奏数值一致，文件路径不同会体现在各自计划声明中。

新摘要状态为`OFFLINE_TIMING_COMPLETE`，`execution_review.allowed=true`，空载配置检查返回READY。本次相关8项离线测试通过，报告`docs/validation/TAUGHT_70S_APPROVED_TESTS.xml`，包含真实配置在模拟设备下prepare和run保持精确70秒、冻结执行上限、旧限值拒绝与周期漂移拒绝。模拟会话不作为实验数据。

新上限仅适用于这份空载配置；力/力矩、工作区、碰撞、数据时龄、清理和空载专用检查继续执行。固定朝向整圈可达性、当前现场起点、实际起停及原始受力采集质量仍需现场验证。旧受力日志缺口和清理问题没有因改节奏而解决。

## 下一次现场步骤

按[现场操作文档](TAUGHT_TRAJECTORY_WORKFLOW.md)逐块执行更新、检查、prepare、预览、显式run和离线complete。当前配置变量必须为`config/taught_tcp_70s.empty.json`，使用新会话目录，不复用旧425秒或旧超限会话。

独立配置检查不连接设备：

```powershell
$timingPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $timingPython -X utf8 -m scripts.lab_session check --config config/taught_tcp_70s.empty.json --empty-load
```

预期输出READY。prepare会采集当前起点并生成该起点下预览；必须核对完整空间、固定朝向、70秒周期和0.13/0.45执行上限，再由操作者显式启动空载。READY不是设备连通或实际执行成功证明。

若需复现一份离线预览，指定新目录：

```powershell
& $timingPython -X utf8 -m scripts.preview_taught_timing --settings results/manual_lab/20261115_taught_70s_v1/settings.json --config config/taught_tcp_70s.empty.json --output-dir results/manual_lab/taught_70s_approved_copy --cycle-duration-s 70 --durations-s 22 17 25
```

命令成功只生成文件；summary单独记录离线限值检查。`ready_for_replay=false`保留，表示预览不自动启动真实运动。本次未连接或驱动机械臂。

## 历史记录

原`results/manual_lab/20261115_taught_70s_v1/`与`config/taught_tcp_70s.review.json`记录原0.03米/秒、0.01米/秒²下的超限事实，仍应被拒绝。原425秒版本记录保守限值下的基线。保留这些历史结果，不覆盖为新上限通过。
