# 90秒连续示教轨迹

2026-11-16补充：已收到本版本的真机尝试，但起步约2.4秒因命令记录后的准入截止检查失败而停止，不能称整圈验证通过。见[失败复核](lab/CONTINUOUS_90S_FAILED_REVIEW_2026-11-16.md)。下文数值仍为离线计划；本次审查未改变该计划。

用户反馈70秒版本偏快，并要求三段之间没有间隔、中途匀速、完整周期90秒。当前配置为`config/continuous_taught_90s.empty.json`，模式`continuous_taught_tcp`，轨迹ID为`drag_taught_continuous_v1`。本版本先用于空载验证，正式带载候选学习尚未迁移。

归档目录为`results/manual_lab/20261115_continuous_90s_v1/`。70秒及425秒版本和原始示教均保留。

## 连续节奏

| 计划时间 | 行为 |
|---|---|
| 0～5秒 | 从静止平滑加速 |
| 5～85秒 | 沿曲线约3.50厘米/秒匀速 |
| 85～90秒 | 平滑减速至静止 |

完整90秒包含首尾加减速，没有保持段。抬升、高位前伸、回收下降是同一条连续曲线上的阶段标签，约30.148秒到达最高点、47.467秒到达前伸最远点；两处均以非零速度通过，不停住、不重新起步。90秒指计划运动时间，不包含SDK连接、运动准备和清理时间。

这里匀速指沿空间曲线的速度大小近似恒定。运动方向持续改变，因此XYZ速度分量会变化，转弯仍有向心加速度。中段理想速度约0.034969774米/秒；按0.02秒计划采样产生4501个目标，实际相邻目标弦速相对该值的最大偏差约0.1131%。

## 空间形状与幅度

原三段几何在两个接头的切向分别相差约48度和99度，不能只删去停留后按非零速度直接换向。本版本将完整三维路径重新均匀取控制点，形成一条连续二阶可导的空间样条，再按弧长分配时间。

该处理保留“抬升—高位前伸—回收下降”和不同的去回分支，没有投影为二维或合并去回路径。为了恢复原幅度，Base Y、Z分别作约1.00631和1.00271的轻微恢复；最高抬升仍约75.10厘米，最远Base +Y前伸仍约89.25厘米。执行用三轴幅度参数默认均为1，工具朝向固定为本次prepare采集的起点朝向。

相对原70秒几何，旧曲线到新曲线的最大最近点距离约9.213毫米，反向约8.866毫米；这是几何修改量，不是实测跟随误差。路径长度从约3.02871米变为2.97243米，变化及来源SHA均保存在`geometry.json`和`summary.json`。形状对照见`shape_comparison.png`。

离线展示使用最近70秒会话`results/manual_lab/20261115_161440/taught_empty_01/anchor.json`中的起点。现场prepare会重新采集当前起点并平移整条Base相对路径，仍须核对完整范围和固定朝向下的可达性；旧起点下的绝对最高Z不能直接套用到新起点。

## 限值和证据范围

沿用已确认的空载速度上限0.13米/秒、加速度上限0.45米/秒²。计划峰值约0.03496978米/秒、0.2886594米/秒²，计入1.01余量后仍低于上限。阶段间不断速，首尾速度和加速度为零。固定周期检查、几何与配置SHA、预览重建和执行前二次审计继续生效；不自动延长周期或降低幅度以取得通过。

70秒真机记录显示最大位置跟随误差约2.653毫米，未发现意外倍速，但受力数据仍有5处约322～324毫秒缺口和查询进程清理未确认记录，整轮质量审核未通过。详见[70秒回传复核](lab/EMPTY_70S_REVIEW_2026-11-15.md)。90秒和连续节奏没有修复受力采集链，不放宽缺口、时龄、力/力矩或清理检查，也不跨缺失区间插值构造负荷结果。

本次仅为离线设计和软件接入，没有执行90秒真机运动，没有新的带腿结果或人体舒适度结论。

完整离线回归634项通过，报告为`docs/validation/PAPER_BRANCH_TESTS_2026-11-15_CONTINUOUS_90S.xml`。测试包含新曲线及弧长时钟、实际采样匀速性、两个交界不停速、来源与内容防篡改、周期与物理限值、空载专用检查以及模拟设备下的prepare、执行、清理和重复流程。配置检查返回READY；软件验证不替代现场运行。

## 当前现场操作

更新论文分支后，直接使用已归档配置，无需先重新生成曲线。逐块操作见[现场步骤](TAUGHT_TRAJECTORY_WORKFLOW.md)：

1. 使用`config/continuous_taught_90s.empty.json`进行`check --empty-load`，预期READY。
2. 以新会话目录执行`prepare --empty-load`，查看`blockers=[]`、完整90秒计划、当前起点下空间图和速度图。
3. 人工核对后显式`run --empty-load --live`；本版本不用于连接假腿或人体后的执行。
4. 结束后离线`complete`，分别审核跟随、完整命令、原始受力覆盖与清理证据；失败记录完整保留并回传。

## 离线复现

以下命令只生成另一份预览与配置，输出路径必须是新路径，不覆盖当前归档：

```powershell
$continuousPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $continuousPython -X utf8 -m scripts.design_continuous_taught_trajectory --source-settings results/manual_lab/20261115_taught_70s_v1/settings.json --config config/taught_tcp_70s.empty.json --anchor results/manual_lab/20261115_161440/taught_empty_01/anchor.json --output-dir results/manual_lab/continuous_90s_copy --output-config config/continuous_90s_copy.json --cycle-duration-s 90 --ramp-duration-s 5
if ($LASTEXITCODE -ne 0) { throw '离线设计未通过，保留目录并查看summary。' }
```

查看`speed_profile.png`确认中段平台和两个不停速分界，查看`shape_comparison.png`核对接头圆顺改动。`ready_for_replay=false`表示离线预览不自动启动真实运动；现场仍需新prepare、审核和显式run。
