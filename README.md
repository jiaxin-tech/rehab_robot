# ROKAE 仰卧位髋膝康复实验代码

本项目研究仰卧位被动髋膝康复：机器人 TCP 等效作用于小腿束带牵引点，结合下肢力学模型、测量数据和低试验预算搜索，研究固定活动范围内的轨迹协调是否值得个体化。仓库包含离线研究算法、仿真与结果，以及独立的真实 ROKAE 诊断、采集和轨迹执行代码。真实执行目前仍是默认关闭的参考轨迹前馈路径，尚未接入在线个体化或力反馈控制。

## 当前研究范围与入口

| 研究部分 | 当前状态 | 入口与说明 |
|---|---|---|
| 固定 ROM 的 V3 与 E2 | 冻结的二维 `beta_flex/beta_extend` 域，共 25×25=625 点；E2 为四个关节/分支 RMS 相对同腿参考比值的最大值 | [FINAL_PERSONALIZATION_RESEARCH_MAINLINE_V1.md](FINAL_PERSONALIZATION_RESEARCH_MAINLINE_V1.md)、`personalization/rom_gated_v2/`、`lower_limb_sim/five_leg_mujoco_v1/` |
| 时序灰箱与 BO 集成 | 已实现过去完整时序的五参数辨识、E0/E2 预测、残差 GP、当前二维 V3 的 EI；保留 LCB、Greedy 和 Pure BO | [MODEL_INFORMED_BO_ARCHITECTURE_V2.md](MODEL_INFORMED_BO_ARCHITECTURE_V2.md)、`personalization/integrated_v2.py:run_offline_configuration` |
| E3 三参数探索 | 独立的候选族与低预算回放实验，比较 β＋时间分配和关键姿态点样条＋时间分配 | [候选族报告](outputs/e3_candidate_comparison/REPORT.md)、[低预算报告](outputs/e3_low_budget/REPORT.md)、`lower_limb_sim/e3_candidate_comparison/`、`lower_limb_sim/e3_low_budget/` |
| 真实测量分析 | 已有静态有效性、同条件重复性、轨迹敏感性分析；真实数据到当前个性化环境的适配仍未贯通 | [REAL_MEASUREMENT_VALIDATION_ANALYSIS_V1.md](REAL_MEASUREMENT_VALIDATION_ANALYSIS_V1.md)、`measurement_validation/analysis.py`、`scripts/run_real_measurement_validation_analysis.py` |
| 机器人诊断与执行 | 已有 Windows 真机只读诊断；采集可靠性仍有阻塞问题，运动未放行 | 本页下方真机诊断、配置与命令；`hardware/`、`collection/`、`control/`、`safety/` |

冻结的五腿 V3/E2 结果仍是 `SIMULATED_MECHANICAL_PERSONALIZATION_NECESSITY = NOT_SUPPORTED`：四条腿的最优点为共同参考，共同参考相对个体最优的队列平均相对 regret 仅为 `0.014137%`（以个体最优值为分母）。这一结论只适用于该仿真模型、候选域和机械指标，不能推出真实患者不需要个体化。ROM 个体化与固定 ROM 内的协调个体化是两个层次；后者需要真实测量的有效性、重复性、轨迹敏感性及跨主体决策差异证据。

后续 E3 探索单独使用四项参考归一化 RMS 的等权均值，并保留 E2 观察最坏分量取舍；它没有替换冻结的 E2/V3。其三维输入为两项协调/姿态参数及 `time_share_shift`，不能当作原二维 625 点实验。最新低预算实验覆盖 6 个模型×2 个轨迹族×3 档约束，在参考＋3 次追加的预算下，Model-Informed BO EI 与 Adaptive Greedy 的最终已执行合格最佳 E3 在 **36/36** 个组合相同；当前结果未显示 BO 探索相对该 Greedy 基线的最终收益优势，也不表示两种算法完全相同。候选网格中更低的 E3、边界最优或模型间差异均不直接构成真实患者个体化收益证据。

V2 集成已经接通 `EpisodeObservation + TimeSeriesIdentificationPayload → 五参数时序辨识 → E0/E2 → 残差 GP → EI`，默认总预算 4 包含参考试验。五参数是有效灰箱参数，E0/E2 仍为模型派生指标。`personalization/environment.py:RealRobotEnvironment` 与真实 ROM 测定接口仍关闭；机器人 episode 的离线辨识入口也不会自动运行 BO。

[ALGORITHM_ARCHITECTURE_REVIEW_V1.md](ALGORITHM_ARCHITECTURE_REVIEW_V1.md) 是 2026-09-18 的阶段审查，其中“当前 V3 未接 EI、E2 adapter 与时序辨识”的缺口已由上述 V2 集成补齐。历史标量拟合、旧 LCB 方法、alpha/EI、P2 和信任/诊断分支仍保留各自作用域；阅读旧报告时应结合 V2 集成文档及后续 E3 报告，不将历史状态当作整个仓库的最新结论。

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

- [2026-08-13 wrench 长测](diagnostics/wrench_hardware_validation_20260813T110502Z.md)：20 Hz 请求测试中出现 49 次 SDK 263 错误及约 10 s 的原生阻塞，同进程 RT 和主循环也受影响；报告明确判定线程隔离不足。
- [进程隔离验证](diagnostics/wrench_process_isolation_validation_20260813T113755Z.md)：独立 RT/wrench 会话能够让 supervisor 和 RT 在 wrench 阻塞期间继续推进，但整体结论为 `PARTIAL`，未形成运动放行依据。
- [2026-08-14 最新 A/B 对照](diagnostics/state_wrench_timing_comparison_20260814T093551709145Z.md)：仅 RT 的 Test A 完成约 900 s；并发 wrench 的 Test B 在 **169.610/900 s** 因 `RuntimeError:RT worker hung` 终止，出现 3 次 SDK 263 错误，最大当前状态年龄 724.138 ms，`READY_FOR_FIRST_MOTION_TEST=false`。

生产路径 `collection/real_robot_acquisition.py` 仍使用 state/wrench/alignment 线程；进程隔离实现在 `scripts/wrench_process_isolation.py` 等诊断脚本中，尚未接入生产 `acquire/execute`。因此“Python 线程分开”不能作为原生调用不阻塞调度与监测的保证。下一步工程工作需要先解决 RT/wrench 长时间并发采集与进程隔离整合，再推进物理坐标/符号/同步、重复性及轨迹敏感性验证。任何软件 stop 都不能替代急停、安全控制器和现场实验人员。

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

```bash
python3 -B -m lower_limb_sim.run_reference_measured_asymmetric
```

它在 `lower_limb_sim/data/reference_candidates/` 保存完整候选 closure 表、未改动的
`reference_measured_raw`、新 slow/nominal、manifest、metadata 和六幅审计图。
`lower_limb_sim/data/` 的新增文件默认被 Git 忽略，但其中已有部分参考产物受 Git 跟踪；
已有文件的修改仍会进入 diff。pinned SHA 能发现文件漂移，新生成的 source CSV 与输出文件
仍需核对跟踪状态并显式归档。

## 安装与离线回归

推荐 CPython 3.12 x64，与仓内 xCoreSDK 的 `cp312-win_amd64` 扩展一致。本机已在项目内配置 Python 3.12.14 和 `.venv`；可直接使用 `.venv\Scripts\python.exe`，无需激活环境或修改系统 PATH。`.tools/`、`.venv/`、`.cache/` 均为忽略的本地产物。

Windows 上使用已有 `uv` 重建同一环境（从仓库根目录运行）：

```powershell
$uvExe = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uvExe) { $uvExe = Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
& $uvExe python install 3.12.14 --install-dir .tools/python --no-bin --no-registry --cache-dir .cache/uv
& $uvExe venv .venv --python .tools/python/cpython-3.12.14-windows-x86_64-none/python.exe --cache-dir .cache/uv
& $uvExe pip install --python .venv/Scripts/python.exe -r requirements-win-py312.lock.txt pip --cache-dir .cache/uv
```

若已自行安装 Python 3.12，可用 `py -3.12 -m venv .venv`，再用 `.venv\Scripts\python.exe -m pip install -r requirements-win-py312.lock.txt`。`requirements.txt` 声明通用依赖范围，`requirements-win-py312.lock.txt` 固定本次 Windows/Python 3.12 的直接及间接依赖；其中 MuJoCo 3.6.0 与既有仿真记录一致。NumPy 下界为 2.0，因为当前端点计算使用 `np.trapezoid`；`imageio-ffmpeg` 提供视频编码所需程序。

当前核心回归使用显式文件集合，避免将真机连接或历史大规模实验混入默认命令：

```powershell
$env:PYTHONUTF8 = '1'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache/matplotlib'
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -c pytest-core.ini -q --junitxml=.cache/core-regression.xml
```

[pytest-core.ini](pytest-core.ini) 覆盖 V3/ROM、E0/E2 时序辨识与 EI、E3、测量分析、日志、轨迹预检/执行器 fake、离线进程诊断及 Windows SDK 加载/fake 测试；不包含 `test_rokae_hardware_integration.py`。从仓库根目录运行，因为 E3 测试读取已跟踪的相对路径 CSV/NPZ。完整历史套件另用 `python -m pytest`，不能把核心回归结果称为全仓通过。

2026-09-23 本机验证：Windows x64、CPython 3.12.14，使用上述锁定依赖，`pip check` 通过；核心集合的 **37 个文件、333 项测试全部通过，耗时 62.70 s**，无失败或跳过。JUnit 记录位于本地 `.cache/core-regression.xml`。该结果验证离线算法与 fake/诊断代码，不代表真机运动验证或历史全仓回归。

冻结产物按原始字节校验 SHA。[.gitattributes](.gitattributes) 对核心依赖的冻结文件分别保留原始 LF 或 CRLF，避免 Windows `core.autocrlf` 改写字节后触发校验失败；此次仅还原原始换行，未修改冻结校验值或重生成实验结果。

核心回归不需要安装 MyoSuite。完整 MyoLeg 渲染/回放另依赖其模型资产，现有 XML 中包含原机器的绝对路径，需另行配置；不属于本次环境验证。Windows 不提供 `resource` 时，回放模块的可选峰值内存统计返回 `null`，不会阻断 E3 导入或伪造内存值。这个兼容修复改变了历史 replay builder 的源码，原阶段针对源码校验值的测试需结合对应历史版本复现；本次不重写其冻结记录。

旧文档中的 `667 passed, 5 skipped` 是 reference freeze 阶段的 macOS 历史记录，不代表当前代码或 Windows 环境的回归结果。

平台边界：

- macOS/Linux：主项目可 import，真实 SDK 测试自动 skip。
- Windows + Python 3.12 x64、无机器人：可运行 SDK import/fake 测试。
- Windows + SDK + 机器人：只有设置 `ROKAE_HARDWARE_TEST=1` 和 `ROKAE_TEST_IP` 后，才运行观察型 connection integration test；厂商 session 副作用仍需人工监督。

`xCoreSDK_python` 不是 pip 依赖。仓库实际运行副本在 `hardware/windows/xcoresdk/`，要求 Windows x64 + CPython 3.12；普通离线研究入口不会创建硬件会话。核心集合中的 SDK 测试只加载扩展并使用 fake robot，不连接真实机器人。

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

state、wrench、alignment 在当前生产实现中各自使用线程；命令目标更新不直接调用 wrench 查询，但已有真机记录证明 SDK 原生阻塞仍可能冻结同进程 Python 执行，进程隔离诊断原型尚未接入此路径。CSV 对不可用值留空并保存 `valid/invalid_reason`，不会伪造零。execute 的 `metadata.json` 固化完整 safety snapshot、frame/anchor/config 路径、轨迹生成审计、reference SHA、`L1/L2`、live preflight 与 execution result；结束时再记录 host 观察到的各流平均发布率。125 Hz state 和 50 Hz wrench 是配置目标；现有并发长测失败，不能将这些目标写成已达到的持续采集性能。

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

必须同时通过：显式 execute/enable、精确 operator confirmation、本机 RT 网卡 IP、SDK connected、frame/anchor/safety reviewed、runtime↔anchor↔config 的 model/serial/controller/payload/软限位一致、anchor↔config 的 tool/workpiece 声明一致且名称存在于 SDK available lists、collision 查询有效且未触发、起点一致、pinned slow candidate 白名单、release manifest 的单独 first-trial approval、C2/ROM/FK/闭合/有限性、由 xyz/time 重算而非信任 CSV 声明的速度/加速度、workspace、state/wrench thread 和新鲜度、力/力矩阈值、reviewed RT 配置与 logger healthy。当前 release approval=false，因此 execute 在连接前 fail closed。live preflight 同时绑定 exact trajectory digest 和完整 safety digest；offline preflight、手工 dataclass、事后改表或更换 safety snapshot 都不能交给 executor。SDK 不能证明当前 HMI 激活的是哪个 tool/workobject，必须由操作员另行审核。attach 后以及首个 hold 紧邻启动前会再次检查 collision、identity/payload/软限位/current joint、idle、stream 和 anchor；首个 RT hold target 及所有后续 command 都必须在审核的 lateness 内完成 flush + `fsync`，否则不会开始/下发并进入 `request_stop(reason)`。

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

该入口复用现有 approved-ROM IK、离线导数、`StateHistoryBuffer` 时间匹配和五参数 `least_squares` estimator，并在 JSON/metrics 中记录源 episode 与本次辨识的 Git commit。它与当前 V2 离线个体化共用估计器，但不会自动生成 V3 试验、验证真实 E2 指标或启动 BO；真实 recorded-data 到当前 observation/payload 的转换仍是独立待完成工作。

## 文档与证据边界

- [PROJECT_AUDIT.md](PROJECT_AUDIT.md)：清理前仓库、数据和 SDK 审计。
- [CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md)：2026-08 的机器人栈模块和线程/数据结构；其中尚无 Windows 验证的描述需结合本页列出的后续真机诊断阅读。
- [FINAL_PERSONALIZATION_RESEARCH_MAINLINE_V1.md](FINAL_PERSONALIZATION_RESEARCH_MAINLINE_V1.md)：冻结 V3/E2 结论与真实测量主线的研究作用域。
- [MODEL_INFORMED_BO_ARCHITECTURE_V2.md](MODEL_INFORMED_BO_ARCHITECTURE_V2.md)：当前时序辨识、E0/E2、残差 GP 和 EI 的离线集成，替代旧审查中对应的未实现状态。
- [E3 候选族报告](outputs/e3_candidate_comparison/REPORT.md)与[E3 低预算报告](outputs/e3_low_budget/REPORT.md)：后续独立三参数探索及 BO/Greedy 结果，不覆盖冻结 V3/E2 结果。
- [REAL_MEASUREMENT_VALIDATION_ANALYSIS_V1.md](REAL_MEASUREMENT_VALIDATION_ANALYSIS_V1.md)：已有真实测量离线分析接口与结论边界。
- [REAL_ROBOT_EXPERIMENT.md](REAL_ROBOT_EXPERIMENT.md)：首次真机分阶段 checklist 与 release gate。
- [CODE_CLEANUP_REPORT.md](CODE_CLEANUP_REPORT.md)：删除、保留、测试与未动用户数据。
