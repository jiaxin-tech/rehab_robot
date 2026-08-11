# ROKAE 仰卧位髋膝康复实验代码

当前仓库已收口为论文对应的真实实验版本：仰卧位、被动髋膝屈伸运动，机器人 TCP 等效作用于小腿束带牵引点。当前任务不是力反馈控制，也不包含在线学习或个体化控制。

模型约定不可更改：

```text
theta_shank = q_hip - q_knee
approved ROM: hip 0–120 deg, knee 5–145 deg
```

正式参考现在保留 CSV 中实测的屈曲和实测的伸展两条不同路径，并用小幅
periodic cubic B-spline 修正达到 C2 周期闭合：

- `reference_measured_asymmetric_closed_slow`：24 s，401 点；冻结域覆盖 100%，第一轮机器人运动唯一白名单。
- `reference_measured_asymmetric_closed_nominal`：12 s，401 点；冻结域覆盖 66.334%，低于既有 90% 门，保留离线且 fail closed。

旧 `reference_closed_symmetric` 和 `reference_closed_c2` 都保留为
legacy/software comparison，`active_reference=false`；它们的反向屈曲构造不再
进入正式机器人 reference。

仓库仍保留 Stage 1–6 离线研究代码与结果作为论文证据，但默认入口已经切换到真实 ROKAE 的观察型诊断、锚点、预览、采集和严格门控执行。

第一轮 slow CSV 绑定的 SHA-256 为：`f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`；等效牵引点几何固定为 `L1=0.42 m`、`L2=0.30 m`。文件内容或几何参数变化都会使 execute fail closed，而不是与同一份可变文件自我比较后放行。

> 当前软件收口已完成，但真机运动状态仍为 **NO-GO**。macOS 环境只能完成离线/fake 回归；xCoreSDK 运动 API 已由本地 `.pyi` 和厂商 examples 静态确认，尚未完成 Windows 空载/真机验证。任何软件 stop 都不能替代急停、安全控制器和现场实验人员。

## 当前数据与执行链

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

```bash
python3 -B -m lower_limb_sim.run_reference_measured_asymmetric
```

它在 `lower_limb_sim/data/reference_candidates/` 保存完整候选 closure 表、未改动的
`reference_measured_raw`、新 slow/nominal、manifest、metadata 和六幅审计图。
`lower_limb_sim/data/` 当前被 Git 忽略，因此 pinned SHA 能发现文件漂移，却不能
让普通 commit 自动保存这些产物；正式归档前必须显式保存 source CSV 与输出文件。

## 安装与离线回归

```bash
python3 -m pip install -r requirements.txt
python3 -B -m pytest -q
```

当前最终离线结果为 `640 passed, 5 skipped in 96.90 s`（645 项已收集）。跳过项是 Windows 原生 SDK 和显式真机 opt-in 测试；该结果不是实机证据。

平台边界：

- macOS/Linux：主项目可 import，真实 SDK 测试自动 skip。
- Windows + Python 3.12 x64、无机器人：可运行 SDK import/fake 测试。
- Windows + SDK + 机器人：只有设置 `ROKAE_HARDWARE_TEST=1` 和 `ROKAE_TEST_IP` 后，才运行观察型 connection integration test；厂商 session 副作用仍需人工监督。

`xCoreSDK_python` 不是 pip 依赖。仓库实际运行副本在 `hardware/windows/xcoresdk/`，要求 Windows x64 + CPython 3.12；普通开发机只会在实际创建硬件会话时尝试加载它。

## 真机前配置

以下模板默认全部 fail-closed：

- `config/rehab_frame_config.json`：床面 rehabilitation x/z 轴，默认 `null`、`reviewed=false`。
- `config/experiment_safety.json`（schema v3）：速度、加速度、起点误差、command lateness、力/力矩、时效、skew、workspace、机器人身份、工具/工件、payload、六轴软限位、RT filter 与外部 network-tolerance 声明均默认 `null`；六类专项审核和总 `reviewed` 均为 `false`。
- StartAnchor：每次观察捕获后绑定 robot model/serial/controller 与人工声明的 tool/workpiece，固定写入 `reviewed=false`。

不要把仿真力值写入真实安全配置。设置 `reviewed=true` 代表现场人员已经核对该机器人、工具、负载、受试者、workspace、wrench 语义和实验流程。

`BASE_WRENCH_ROTATION_VERIFIED` 默认保持 `False`，任何脚本都不会自动修改。`getEndTorque()` 的补偿、作用点、world/base 方向和与 RT state 的物理同步仍需按实机流程验证。

## 观察型命令（项目代码不发运动目标）

以下命令不会由项目代码调用 automatic、power-on、clear-error、标定、drag 或发送运动目标。但厂商 `.pyi` 说明对象初始化可能执行 `moveReset`，disconnect 会在断开前停止既有运动；因此不能把 connect/capture/acquire 称为“零运动侧效应”，首次会话必须确认机器人已 idle 并在监督下进行。

```powershell
# 1. 最小连接、状态和内部 wrench 探测
python -m scripts.rokae_probe --ip 192.168.50.103

# 2. 在操作员已通过外部安全流程放置好机器人后捕获锚点
python -m scripts.capture_start_anchor `
  --ip 192.168.50.103 `
  --output anchors/subject_001_slow.json `
  --anchor-id subject_001_slow `
  --tool-name reviewed_tool_name `
  --workpiece-name reviewed_workpiece_name

# 3. 纯离线预览；不 import/连接机器人
python -m scripts.preview_rehab_trajectory `
  --anchor anchors/subject_001_slow.json `
  --frame-config config/rehab_frame_config.json `
  --output-dir previews/subject_001_slow

# 4. 不发送运动目标的定时采集
python -m scripts.acquire_robot_data `
  --ip 192.168.50.103 `
  --episode-dir data/subject_001/acquire_001 `
  --duration-s 30
```

既有专项诊断仍可用：

```powershell
python -m scripts.check_rt_state_timing --duration 10
python -m scripts.check_wrench_query_timing --duration 10 --target-hz 50
python -m scripts.check_snapshot_alignment --duration 10
python -m scripts.check_wrench_frame_rotation --direction X --confirm-unloaded
python -m scripts.check_wrench_pose_dependence --poses 3 --confirm-unloaded
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

state、wrench、alignment 各自运行；wrench 阻塞不会直接占用命令目标更新路径。CSV 对不可用值留空并保存 `valid/invalid_reason`，不会伪造零。execute 的 `metadata.json` 固化完整 safety snapshot、frame/anchor/config 路径、轨迹生成审计、reference SHA、`L1/L2`、live preflight 与 execution result；结束时再记录 host 观察到的各流平均发布率。125 Hz state 和 50 Hz wrench 仍只是目标，能否稳定达到必须由 Windows 真机 episode 证明。

## 真实执行：默认关闭

执行前，实验人员必须在外部控制器/HMI 和既定安全流程中完成必要准备。外部 RT network tolerance 的审核值会写入 safety/episode 证据，但当前 SDK 路径无法回读确认；程序只把 `reviewed_rt_filter_hz` 显式传给 `setFilterFrequency`。程序不会自动：

- 切 automatic 或上电；
- clear error、回零或移动到 anchor；
- 修改 controller collision/tool/load 配置；
- 执行 nominal/fast/C1–C8。

机器人必须已经位于人工审核的 StartAnchor；程序会在 preflight 和 RT hold 前各检查一次起点位置/姿态误差。

唯一执行入口示例：

```powershell
python -m scripts.run_rehab_experiment `
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

必须同时通过：显式 execute/enable、精确 operator confirmation、本机 RT 网卡 IP、SDK connected、frame/anchor/safety reviewed、runtime↔anchor↔config 的 model/serial/controller/payload/软限位一致、anchor↔config 的 tool/workpiece 声明一致且名称存在于 SDK available lists、collision 查询有效且未触发、起点一致、pinned slow 白名单、C2/ROM/FK/闭合/有限性、由 xyz/time 重算而非信任 CSV 声明的速度/加速度、workspace、state/wrench thread 和新鲜度、力/力矩阈值、reviewed RT 配置与 logger healthy。live preflight 同时绑定 exact trajectory digest 和完整 safety digest；offline preflight、手工 dataclass、事后改表或更换 safety snapshot 都不能交给 executor。SDK 不能证明当前 HMI 激活的是哪个 tool/workobject，必须由操作员另行审核。attach 后以及首个 hold 紧邻启动前会再次检查 collision、identity/payload/软限位/current joint、idle、stream 和 anchor；首个 RT hold target 及所有后续 command 都必须在审核的 lateness 内完成 flush + `fsync`，否则不会开始/下发并进入 `request_stop(reason)`。

执行器和 motion facade 都是 single-use；stop intent 一旦发布就不能再 attach/start/send，native stop 失败可重试，且 `stopLoop`/`stopMove` 未确认成功时不会报告 completed。调度器不会为追赶进度突发补发过期点，并在每次持久化 command 后重新检查缓存健康和绝对 deadline。execute 当前完全不消费 wrapper 的 `has_motion_error()`；collision 只在 live preflight、attach 后和紧邻 start 前查询，不在 command 热路径连续轮询。控制器碰撞保护必须由现场预先配置并保持有效，这也是当前仍为真机 **NO-GO** 的证据缺口之一。

运动实现使用本地 SDK 已确认的 realtime Cartesian callback API：`getRtMotionController`、`setFilterFrequency`、`setControlLoopCar`、`startMove(cartesianPosition)`、`startLoop`、`stopLoop`、`stopMove`。第一版只附着到实验人员已外部准备的 automatic/power/RT 状态，不切模式、不自动上电、不移动到起点；filter 只能来自 schema-v3 safety 中的人工审核值。静态 API 证据不能解释为真机慢速空载验证。

## 真实 episode 离线辨识

五参数模型本身没有修改。命令为：

```bash
python -m scripts.identify_real_episode EPISODE_DIR
```

辨识前必须在 episode 内提供人工审核的 `identification_config.json`；可从 `config/real_identification_config.json` 复制。模板不会提供默认人体参数、wrench frame/sign、变换或延迟。缺配置、未审核、数据不足或优化失败时，命令不生成假结果。成功后只写：

```text
identified_parameters.json
prediction_metrics.csv
```

该入口复用现有 approved-ROM IK、离线导数、`StateHistoryBuffer` 时间匹配和五参数 `least_squares` estimator，并在 JSON/metrics 中记录源 episode 与本次辨识的 Git commit。

## 文档与证据边界

- [PROJECT_AUDIT.md](PROJECT_AUDIT.md)：清理前仓库、数据和 SDK 审计。
- [CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md)：当前模块和线程/数据结构。
- [REAL_ROBOT_EXPERIMENT.md](REAL_ROBOT_EXPERIMENT.md)：首次真机分阶段 checklist 与 release gate。
- [CODE_CLEANUP_REPORT.md](CODE_CLEANUP_REPORT.md)：删除、保留、测试与未动用户数据。
