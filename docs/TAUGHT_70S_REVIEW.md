# 完整70秒示教轨迹

用户将完整周期目标改为70秒。固定工具朝向、XYZ幅度参数1和已处理的三维空间路径保持不变，仅改变节奏。原425秒版本保留在`results/manual_lab/20261115_taught_design_v1/`。

新70秒方案为`results/manual_lab/20261115_taught_70s_v1/`，对应`config/taught_tcp_70s.review.json`。几何文件逐字节复制，SHA仍为`f565ad65aa7c4b5106bd9b5f634fd4197f31272cca2609d84cf049a8af25994a`。

| 阶段 | 时间区间，秒 | 时长，秒 |
|---|---:|---:|
| 起点保持 | 0～2 | 2 |
| 抬升 | 2～24 | 22 |
| 高位前伸 | 24～41 | 17 |
| 远端保持 | 41～43 | 2 |
| 回收下降 | 43～68 | 25 |
| 终点保持 | 68～70 | 2 |

三运动段仍平缓起停，各段端点速度和加速度为零。最高点停速但不额外保持。处理后路径约3.029米，抬升最高点、前伸最远点及闭合起点与425秒版本一致。

## 当前限制冲突

密集解析导数和全部样条节点核验，70秒方案峰值速度约0.123829米/秒，加速度约0.430314米/秒²。当前空载限制为0.03米/秒和0.01米/秒²；70秒方案分别超过约4.13倍和43.03倍。仅按0.03米/秒恒速走完整路径就需约101秒，尚未计起停和6秒保持，因此不能同时保留当前幅度、70秒和原有限值。

本次只生成70秒离线方案，现有限值未提高；其状态为`OFFLINE_TIMING_REVIEW_REQUIRED`。若按此形状做70秒空载，待确认的新上限为0.13米/秒和0.45米/秒²，可容纳计算峰值与原1.01余量，但这些数值不是现场设备适用性证明。固定朝向的可达性、实际起停与原始受力采集质量仍需现场验证。

新配置的`trajectory.fixed_cycle_duration_s=70`将周期锁定。检查、准备和运行分别核对周期及限值；存在冲突时明确拒绝，不把70秒悄悄延长到425秒，也不改变幅度。

完整离线回归606项通过，报告为`docs/validation/PAPER_BRANCH_TESTS_2026-11-15_70S.xml`。测试覆盖超限时在硬件起点采集前拒绝、可行固定周期不被自动延长、运行前周期漂移拒绝及旧非固定周期行为。未执行真实机械臂运动。

## 查看与复现

在已更新的论文分支内使用已有Python，以下检查不连接设备：

```powershell
$timingPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $timingPython -X utf8 -m scripts.lab_session check --config config/taught_tcp_70s.review.json --empty-load
```

在当前未提高限值的版本，预期输出`FIXED_CYCLE_LIMIT_CONFLICT`，不能执行运动。可打开`results/manual_lab/20261115_taught_70s_v1/`内的`trajectory_3d.png`、`xyz_time.png`、`speed_time.png`和`acceleration_time.png`核对70秒曲线。

若需另建一份离线方案，用新目录与新配置路径：

```powershell
& $timingPython -X utf8 -m scripts.preview_taught_timing --settings results/manual_lab/20261115_taught_design_v1/settings.json --config config/taught_tcp_empty.template.json --output-dir results/manual_lab/taught_70s_review_copy --output-config config/taught_tcp_70s_review_copy.json --cycle-duration-s 70 --durations-s 22 17 25
```

命令成功仅代表离线文件生成成功。summary单独记录执行审查结果和冲突原因；`ready_for_replay=false`，不自动启动运动。不要覆盖原始示教数据、旧425秒设计或本次70秒方案。
