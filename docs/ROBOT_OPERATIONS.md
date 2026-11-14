# 机器人操作与诊断指南

[返回文档索引](README.md) · [研究主线](RESEARCH.md) · [代码地图](CODE_MAP.md)

2026年11月14日现场优先使用[今日逐步清单](TODAY_LAB_CHECKLIST_2026-11-14.md)与 scripts/lab_next_visit.ps1。当前配置已有现场填写，不能再按“全部空模板”理解；是否可执行由当前完整配置、安装、有效发布及实时检查共同决定。

本文集中维护原 README 中的机器人命令、运行契约与当前诊断状态，所有命令从仓库根目录运行。[CURRENT_ARCHITECTURE.md](../CURRENT_ARCHITECTURE.md) 与 [REAL_ROBOT_EXPERIMENT.md](../REAL_ROBOT_EXPERIMENT.md) 保留在根目录，是历史协议按原路径和 SHA 引用的原始输入；其旧测试记录或“尚无 Windows 验证”的表述需结合本文的后续证据阅读。安装步骤统一见[上手指南](GETTING_STARTED.md)。

## 冻结机器人参考与模型约定

模型约定不可更改：

```text
theta_shank = q_hip - q_knee
ROM_PROTOCOL_V2: hip 0–120 deg, knee 5–145 deg
```

正式参考现在保留 CSV 中实测的屈曲和实测的伸展两条不同路径，并用小幅
periodic cubic B-spline 修正达到 C2 周期闭合：

- `reference_measured_asymmetric_closed_slow`：24 s，401 点；冻结域覆盖 100%，是唯一 first-trial candidate，但 reference freeze 本身不批准物理机器人运动。
- `reference_measured_asymmetric_closed_nominal`：12 s，401 点；冻结域覆盖 66.334%，低于既有 90% 门，保留离线且 fail closed。

旧 `reference_closed_symmetric` 和 `reference_closed_c2` 都保留为
legacy/software comparison，`active_reference=false`；它们的反向屈曲构造不再
进入正式机器人 reference。

Stage 1–6 离线建模与结果仍是研究基础。下文介绍独立的 ROKAE 观察型诊断、锚点、预览、采集和门控执行入口；这些入口不会调用上面的个体化搜索。

正式协议的唯一可编辑来源是 `config/formal_experiment_manifest.json`；
机器人参考链的 workspace、IK、reference、candidate、identification、preview/preflight 读取
这一协议；个体化研究的 subject-specific ROM 与 V3 域另外由冻结的 `SubjectROMProfile` 约束，不能与机器人参考白名单混用。迁移前 5–130° 数据只作 legacy provenance，不是默认 active 输入。

tracked release bundle 位于 `reference_release/`；默认 active loader 只接受其中的 slow CSV，并同时校验 release/source SHA、source skeleton SHA、ROM、闭合、C2、asymmetry 和 duration。SHA-256 为：`f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`；等效牵引点几何固定为 `L1=0.42 m`、`L2=0.30 m`。文件内容或几何参数变化都会 fail closed。

> 真机运动状态仍为 **NO-GO**。2026-08-13 至 08-14 已完成多项 Windows ROKAE 只读诊断，发现 SDK 原生阻塞和并发采集故障；这些记录不是空载运动验证。xCoreSDK 运动 API 的本地 `.pyi` 与厂商 examples 只提供静态接口证据。

## 已有真机诊断与当前工程断点

- [2026-08-13 wrench 长测](../diagnostics/wrench_hardware_validation_20260813T110502Z.md)：20 Hz 请求测试中出现 49 次 SDK 263 错误及约 10 s 的原生阻塞，同进程 RT 和主循环也受影响；报告明确判定线程隔离不足。
- [进程隔离验证](../diagnostics/wrench_process_isolation_validation_20260813T113755Z.md)：独立 RT/wrench 会话能够让 supervisor 和 RT 在 wrench 阻塞期间继续推进，但整体结论为 `PARTIAL`，未形成运动放行依据。
- [2026-08-14 最新 A/B 对照](../diagnostics/state_wrench_timing_comparison_20260814T093551709145Z.md)：仅 RT 的 Test A 完成约 900 s；并发 wrench 的 Test B 在 **169.610/900 s** 因 `RuntimeError:RT worker hung` 终止，出现 3 次 SDK 263 错误，最大当前状态年龄 724.138 ms，`READY_FOR_FIRST_MOTION_TEST=false`。

生产路径已显式接入 collection/wrench_process.py 的独立进程provider，原生力查询在子进程；父进程保留状态、IPC消费和对齐线程，不再直接查询原生力。10月18日空载尝试仍遇到数据新鲜度与清理问题，10月19日修复查询计时、审计批次与退出检查，现场复测尚未完成。进程隔离已接入不等于现场实时性已验收；最新复核见[10月19日审计](../site_evidence/OFFLINE_EMPTY_LOAD_REVIEW_2026-10-19.md)。

## 机器人参考数据与执行链

```text
measured flexion + measured extension natural cycle
  -> full-joint closure audit + small periodic C2 correction
  -> reference-local excitation / five-parameter identification
  -> frozen-domain compatibility check
  -> L1/L2 FK equivalent pull point
  -> start_anchored_relative TCP reference
  -> offline preview/audit
  -> reviewed frame + anchor + safety
  -> independent state/wrench acquisition + logger-ready barrier
  -> explicit slow-only realtime Cartesian execution
  -> offline five-parameter identification adapter
```

起点锚定模式使用：

```text
p_R(t) = [x_pull(t), 0, z_pull(t)]
delta_p_R(t) = p_R(t) - p_R(0)
p_tcp_B(t) = p_tcp_start_B + R_base_from_rehab @ delta_p_R(t)
```

TCP 姿态全程固定为 StartAnchor 中的起始姿态。该模式不需要绝对 hip center，也不把骨架 ankle 当成牵引点。既有 `absolute_calibrated` 导出模式仍保留，没有被覆盖。

纯离线重建命令为：

```powershell
.\.venv\Scripts\python.exe -B -m lower_limb_sim.run_reference_measured_asymmetric
```

它在 `lower_limb_sim/data/reference_candidates/` 保存完整候选 closure 表、未改动的
`reference_measured_raw`、新 slow/nominal、manifest、metadata 和六幅审计图。
`lower_limb_sim/data/` 的新增文件默认被 Git 忽略，但其中已有部分参考产物受 Git 跟踪；
已有文件的修改仍会进入 diff。pinned SHA 能发现文件漂移，新生成的 source CSV 与输出文件
仍需核对跟踪状态并显式归档。

## 真机前配置

新安装的空白模板位于 config/templates/。config 中的现场文件已填写过部分值，审核标志、说明文字和实际缺失字段需要逐项核对：

- config/rehab_frame_config.json 已记录床面方向并标为已核对，仍须与今天实际安装一致。
- config/experiment_safety.json（schema v3）已有部分数值和审核标志，但起点误差、command lateness、软限位及RT参数仍有缺项；正文与总审核标志还有矛盾。单个true不会替代缺失事实，完整执行校验仍会拒绝。
- StartAnchor：每次观察捕获后绑定 robot model/serial/controller 与人工声明的 tool/workpiece，固定写入 `reviewed=false`。

不要把仿真力值写入真实安全配置。设置 `reviewed=true` 代表现场人员已经核对该机器人、工具、负载、受试者、workspace、wrench 语义和实验流程。

`BASE_WRENCH_ROTATION_VERIFIED` 默认保持 `False`，任何脚本都不会自动修改。`getEndTorque()` 的补偿、作用点、world/base 方向和与 RT state 的物理同步仍需按实机流程验证。

## 观察型命令（项目代码不发运动目标）

以下命令不会由项目代码调用 automatic、power-on、clear-error、标定、drag 或发送运动目标。但厂商 `.pyi` 说明对象初始化可能执行 `moveReset`，disconnect 会在断开前停止既有运动；因此不能把 connect/capture/acquire 称为“零运动侧效应”，首次会话必须确认机器人已 idle 并在监督下进行。

```powershell
# 1. 最小连接、状态和内部 wrench 探测
.\.venv\Scripts\python.exe -m scripts.rokae_probe --ip 192.168.50.103

# 2. 在操作员已通过外部安全流程放置好机器人后捕获锚点
.\.venv\Scripts\python.exe -m scripts.capture_start_anchor `
  --ip 192.168.50.103 `
  --output anchors/subject_001_slow.json `
  --anchor-id subject_001_slow `
  --tool-name reviewed_tool_name `
  --workpiece-name reviewed_workpiece_name

# 3. 纯离线预览；不 import/连接机器人
.\.venv\Scripts\python.exe -m scripts.preview_rehab_trajectory `
  --anchor anchors/subject_001_slow.json `
  --frame-config config/rehab_frame_config.json `
  --output-dir previews/subject_001_slow

# 4. 不发送运动目标的定时采集
.\.venv\Scripts\python.exe -m scripts.acquire_robot_data `
  --ip 192.168.50.103 `
  --episode-dir data/subject_001/acquire_001 `
  --duration-s 30
```

既有专项诊断仍可用：

```powershell
.\.venv\Scripts\python.exe -m scripts.check_rt_state_timing --duration 10
.\.venv\Scripts\python.exe -m scripts.check_wrench_query_timing --duration 10 --target-hz 50
.\.venv\Scripts\python.exe -m scripts.check_snapshot_alignment --duration 10
.\.venv\Scripts\python.exe -m scripts.check_wrench_frame_rotation --direction X --confirm-unloaded
.\.venv\Scripts\python.exe -m scripts.check_wrench_pose_dependence --poses 3 --confirm-unloaded
```

## 离线 preview 输出

`preview_rehab_trajectory` 生成：

```text
trajectory_preview.csv
preview_metadata.json
trajectory_3d.png
xyz_time.png
speed_time.png
acceleration_time.png
```

它检查首末目标等于 anchor、固定姿态、有限值、位置/速度/加速度跳变、正式 ROM、FK 和 `q_hip-q_knee`。`preview_metadata.json` 同时记录当前 Git commit、pinned reference SHA-256、`L1/L2` 与等效束带牵引点物理定义；预览有效也不会把 `robot_execution_approved` 设为真。

## 五文件 episode

真实 acquire/execute 使用统一高分辨率 host monotonic 时钟 `time.perf_counter_ns()`，并在 logger-ready 后创建：

```text
EPISODE_DIR/
  robot_state.csv
  robot_wrench.csv
  trajectory_command.csv
  aligned_snapshot.csv
  metadata.json
```

父进程使用状态与IPC消费/对齐线程，原生wrench查询由独立进程拥有。CSV对不可用值留空并保存valid/invalid_reason，不伪造零。metadata.json保留配置、安装、参考身份、实时检查与退出结果。目标频率与实际发布率分开记录；平均频率达标不能代替最大年龄、时序对齐、查询长尾与清理结果的检查。

## 真实执行：默认关闭

执行前，实验人员必须核对外部控制器/HMI及当前安装。外部RT network tolerance声明不等于SDK已回读确认。通用执行默认不准备RT；带假腿单轮入口使用显式same-session准备，会切模式、上电并在结束时执行停止/下电/断连，不能称零副作用。下面的限制适用于通用默认路径；具体动作以所选现场入口为准：

- 切 automatic 或上电；
- clear error、回零或移动到 anchor；
- 修改 controller collision/tool/load 配置；
- 执行 nominal/fast/C1–C8。

机器人必须已经位于人工审核的 StartAnchor；程序会在 preflight 和 RT hold 前各检查一次起点位置/姿态误差。

唯一执行入口示例：

```powershell
.\.venv\Scripts\python.exe -m scripts.run_rehab_experiment `
  --mode execute `
  --enable-motion `
  --ip 192.168.50.103 `
  --local-ip 192.168.50.10 `
  --episode-dir data/subject_001/slow_001 `
  --anchor anchors/subject_001_slow.json `
  --anchor-id subject_001_slow `
  --frame-config config/rehab_frame_config.json `
  --safety-config config/experiment_safety.json `
  --trajectory reference_measured_asymmetric_closed_slow `
  --operator-confirmation "I CONFIRM SUPERVISED SLOW ROBOT MOTION"
```

必须同时通过：显式 execute/enable、精确 operator confirmation、本机 RT 网卡 IP、SDK connected、frame/anchor/safety reviewed、runtime↔anchor↔config 的 model/serial/controller/payload/软限位一致、anchor↔config 的 tool/workpiece 声明一致且名称存在于 SDK available lists、collision 查询有效且未触发、起点一致、pinned slow candidate 白名单、release manifest 的单独 first-trial approval、C2/ROM/FK/闭合/有限性、由 xyz/time 重算而非信任 CSV 声明的速度/加速度、workspace、state/wrench thread 和新鲜度、力/力矩阈值、reviewed RT 配置与 logger healthy。当前 release approval=false，因此 execute 在连接前 fail closed。live preflight 同时绑定 exact trajectory digest 和完整 safety digest；offline preflight、手工 dataclass、事后改表或更换 safety snapshot 都不能交给 executor。SDK 不能证明当前 HMI 激活的是哪个 tool/workobject，必须由操作员另行审核。attach 后以及首个 hold 紧邻启动前会再次检查 collision、identity/payload/软限位/current joint、idle、stream 和 anchor；首个 RT hold target 及所有后续 command 都必须在审核的 lateness 内完成 flush + `fsync`，否则不会开始/下发并进入 `request_stop(reason)`。

执行器和 motion facade 都是 single-use；stop intent 一旦发布就不能再 attach/start/send，native stop 失败可重试，且 `stopLoop`/`stopMove` 未确认成功时不会报告 completed。调度器不会为追赶进度突发补发过期点，并在每次持久化 command 后重新检查缓存健康和绝对 deadline。execute 当前完全不消费 wrapper 的 `has_motion_error()`；collision 只在 live preflight、attach 后和紧邻 start 前查询，不在 command 热路径连续轮询。控制器碰撞保护必须由现场预先配置并保持有效，这也是当前仍为真机 **NO-GO** 的证据缺口之一。

运动实现使用本地 SDK 已确认的 realtime Cartesian callback API：`getRtMotionController`、`setFilterFrequency`、`setControlLoopCar`、`startMove(cartesianPosition)`、`startLoop`、`stopLoop`、`stopMove`。第一版只附着到实验人员已外部准备的 automatic/power/RT 状态，不切模式、不自动上电、不移动到起点；filter 只能来自 schema-v3 safety 中的人工审核值。静态 API 证据不能解释为真机慢速空载验证。

### 2026-10-15 发布链路更新

首次空载试验现在使用独立的 `FIRST_ROBOT_TRIAL_EMPTY_LOAD` 批准文件，冻结参考 manifest 仍为 `NO_GO`。生成与审批命令、`--wrench-hz`、执行入口的 `--release-approval`、配置/证据绑定和离线验证范围见 [首次空载发布流程](ROBOT_TRIAL_RELEASE.md)。没有有效批准文件仍在连接前拒绝执行。

碰撞状态可显式选 `query` 或厂家 `events` 回调。事件模式会在命令循环中检查缓存有效性；这不等于对 SDK 做连续查询，也不证明真机有初始事件或心跳。控制器保护和现场验证要求仍保留。

## 真实 episode 离线辨识

五参数模型本身没有修改。命令为：

```powershell
.\.venv\Scripts\python.exe -m scripts.identify_real_episode EPISODE_DIR
```

辨识前必须在 episode 内提供人工审核的 `identification_config.json`；可从 `config/real_identification_config.json` 复制。模板不会提供默认人体参数、wrench frame/sign、变换或延迟。缺配置、未审核、数据不足或优化失败时，命令不生成假结果。成功后只写：

```text
identified_parameters.json
prediction_metrics.csv
```

该入口复用现有 approved-ROM IK、离线导数、`StateHistoryBuffer` 时间匹配和五参数 `least_squares` estimator，并在 JSON/metrics 中记录源 episode 与本次辨识的 Git commit。它与当前 V2 离线个体化共用估计器，但不会自动生成 V3 试验、验证真实 E2 指标或启动 BO；真实 recorded-data 到当前 observation/payload 的转换仍是独立待完成工作。
