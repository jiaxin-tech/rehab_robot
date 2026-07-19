# 康复机器人控制系统

本仓库采集 ROKAE xCoreSDK 机器人状态与机器人控制器内部的力/力矩估计；**不使用外接 UDP/ATI 六维力传感器**。数据层以可追溯的状态快照为单位，供离线回放、ComfortNet、切向 PINN 与路径切向 MPC 使用。

## 已确认的数据链路

```text
xCore RT state packet ── pose/q/keypads ──┐
                                           ├─ RobotStateSample ── schema v3 CSV
forceControl().getEndTorque() ── wrench ──┘       │
       (独立、带主机时间戳的 SDK 查询)              ├─ ComfortNet
                                                   └─ tangential PINN / MPC
```

`xCoreSDK_python` v0.7.0 的 `RtSupportedFields` 只公开：TCP 位姿、关节位置、按键与外部轴字段。项目 RT 线程订阅 `tcpPoseAbc_m`、`jointPos_m`、`keypads`；其中 TCP 位姿是 **base 系**、单位 m/rad。该 RT 流没有公开的设备时间戳、关节速度、关节力矩、笛卡尔 wrench 或碰撞字段。

因此：

- TCP/关节位置来自同一个接收状态包；SDK 未承诺硬件级严格同步。
- TCP/关节速度是相邻 RT 包按主机单调时间的数值差分，不是 SDK 原始速度；RPY 差分不应解释为严格物理角速度。
- `forceControl().getEndTorque()` 是单独的控制器查询。它返回 `joint_measured_torque_nm`、`joint_external_torque_nm`（均为关节空间 N·m）、`cartesian_force_raw_n`（N）和 `cartesian_torque_raw_nm`（N·m）。
- `joint_measured_torque_nm` 不能当作人体交互力；`joint_external_torque_nm` 是控制器模型估计的外部关节力矩，仍依赖正确的工具/负载和模型。
- Cartesian 结果可请求 `world`、`flange` 或 `tool` 表达。默认请求 `world`；SDK 文档没有证明其补偿内容、wrench 参考点或与 RT 位姿的天然同步关系。

所以本实现应准确称为：**同一机器人控制器的数据、位姿/关节位置同一 RT 接收包，wrench 为独立且带时间戳的 `getEndTorque()` 查询；不是完整硬件同一时刻同步采集。**

## 快照、时间和坐标系

`collection/state.py` 定义不可变的 `KinematicStateFrame`、`InternalWrenchFrame` 和 `RobotStateSample`。每行保存状态/力的 host monotonic 时间、年龄、内部 skew、`valid` 和 `invalid_reason`；SDK 不提供的值写空，绝不填零。

算法坐标系是 `base`。默认 raw wrench 是 `world`，同时保留 raw、软件 bias、corrected 和 base-expression 字段。`baseFrame()` 仅用于 world→base 的表达旋转；在 `BASE_WRENCH_ROTATION_VERIFIED=True` 前，样本会记录为待验证且不能被 PINN/ComfortNet/MPC 当作有效 base 数据。完整 wrench 的 `tau_b = R tau_a + p × (R F_a)` 数学辅助函数存在于 `collection/state.py`，但除非已验证参考点偏移，采集代码只声明“旋转表达”，不声称做了力矩平移。

默认有效性门限在 `config/settings.py`：50 Hz、状态/力最大年龄 50 ms、内部 skew 最大 20 ms。未来时间、断流、状态流异常、碰撞、控制器错误、未完成软件 bias 或不可用 base wrench 都会产生无效行并保留原因。

## schema v3 CSV

每个 session 目录形如：

```text
data/<subject_id>/<session_id>/
  metadata.json
  episode_0001.csv
  episode_0001.json
```

CSV 所有算法量使用 SI。字段按以下组保存：

- 元数据与时效：`sample_index`、`sample_time_s`、`robot_state_time_s`、`pose_time_s`、`joint_time_s`、`force_time_s`、`force_query_*`、age/skew、`valid`、`invalid_reason`。
- base 位姿/速度/估算加速度：`x_m...rz_rad`、`vx_mps...wz_radps`、`ax_est_mps2...`、来源字段。
- 关节：`q*_rad`、`dq*_radps`、`joint_measured_torque_*_nm`、`joint_external_torque_*_nm`。
- wrench：raw/bias/corrected/base 的 `fx/fy/fz` 和 `tx/ty/tz`，加 `raw_force_frame`、`base_wrench_transform_kind`、`force_source`。
- 轨迹：`trajectory_s`、`trajectory_arc_length_m`、切向量、`force_tangent_n`、`velocity_tangent_mps`、`acceleration_tangent_mps2`。
- 状态与标签：operation/collision/error、`force_estimate_valid`、comfort/pain 标签。

无效行默认仍写入 CSV，便于离线审计；训练加载器只接受 schema v3、base/SI、`valid=1`、已验证 base wrench 的有限值行。

## 安全与首次真机条件

内部力估计不是独立硬件安全传感器。`SafetyGuard` 检查状态/力时效、笛卡尔力/力矩、关节外力矩、TCP 速度、单步目标位移、工作空间、关节软限位余量、碰撞与控制器异常，并在触发时调用机器人 stop。

默认配置刻意 fail-closed。运行 `run_collection.py` 或 `run_control.py` 前必须在 `config/settings.py` 填写并复核：

```python
WORKSPACE_MIN_M = [...]
WORKSPACE_MAX_M = [...]
CONTROLLER_COLLISION_CONFIGURATION_CONFIRMED = True  # 仅在 HMI/控制器碰撞检测已确认后
ROBOT_LOCAL_IP = "..."  # 连续实时轨迹/MPC 必填
```

控制器软限位由 SDK `getSoftLimit()` 读取；读取失败会拒绝人体采集/控制。适配器还记录 SDK Toolset 的 load、TCP/ref 与可用工具/工件名称，但该接口不能证明独立 HMI/RL 工程的 active tool/workobject，故仍需操作员核对。不要自动调用 `calibrateForceSensor()`；默认流程只做会话级的软件 reference bias。

首次无人空载步骤：

1. 在 HMI 配置并复核工具、负载、重心、工作空间、软限位和碰撞检测。
2. 先运行 `python scripts/check_internal_force.py --duration 10`；不运动，仅记录 raw 内部 wrench 的频率、漂移和异常。
3. 在固定参考姿态、idle、低速度、无人接触时运行带 `--software-bias` 的诊断，确认会话软件 bias 后的稳定性。
4. 仅当 base/world 验证完成后，才将 `BASE_WRENCH_ROTATION_VERIFIED=True`，再开始有效 base-frame 数据采集。

首次轻推方向验证：

1. 空载和软件 bias 完成后，在参考姿态对一个已知 base 轴施加很小、缓慢的手推。
2. 对照 raw world 与 rotated base 记录，核对力正负、轴向、量级，以及力矩是否只按同一参考点旋转。
3. 沿至少两个不共线方向重复；确认 `baseFrame()` 的方向、Euler 约定、工具/负载和 `getEndTorque()` 的补偿行为。
4. 在记录验证证据后才启用 `BASE_WRENCH_ROTATION_VERIFIED`。若力矩作用点/平移语义未获厂商确认，仍不得把旋转结果当完整 wrench 变换。

## 运行

环境要求：Windows 10/11 64 位、CPython 3.12 64 位、xCoreSDK 0.7.0 与 xCore 控制器 ≥3.2。仓库中 `.pyd/.dll` 为 Windows 二进制，不能在 macOS/Linux 真机加载。

```powershell
python -m pip install -r requirements.txt

# ROKAE 真机诊断（全程只读：不发送上电、运动、拖动、标定或 stop 命令）
# 1. 先确认 RT 状态流的更新周期、age、丢帧与异常
python scripts/check_rt_state_timing.py --duration 10

# 2. 再确认直接 getEndTorque() 查询能否稳定满足 50 Hz（20 ms deadline）
python scripts/check_wrench_query_timing.py --duration 10 --target-hz 50

# 3. 用项目的统一快照路径审计状态/wrench age、skew 与无效原因
python scripts/check_snapshot_alignment.py --duration 10
python scripts/check_snapshot_alignment.py --duration 10 --software-bias --confirm-unloaded

# 4. 完成软件 bias 后，对 X/Y/Z 分别做一次小而慢的人工正向轻推。
#    脚本只记录主轴、符号和 cross-axis ratio，绝不自动修改验证开关。
python scripts/check_wrench_frame_rotation.py --direction X --confirm-unloaded
python scripts/check_wrench_frame_rotation.py --direction Y --confirm-unloaded
python scripts/check_wrench_frame_rotation.py --direction Z --confirm-unloaded

# 5. 操作员经 HMI/外部方式安排多个静止姿态后，检查软件 bias 的残差是否随姿态变化。
#    该结果不能用来断言 SDK 是否已完成重力补偿。
python scripts/check_wrench_pose_dependence.py --poses 3 --confirm-unloaded

# 旧的单通道内部 wrench 快速查看（不上电、不运动）
python scripts/check_internal_force.py --duration 10
python scripts/check_internal_force.py --duration 10 --software-bias

# 配置完安全边界和控制器碰撞检测后，采集
python scripts/run_collection.py --subject subject_001 --session session_01

# 训练（默认切向 PINN；--cartesian 仅用于明确验证的 3D 实验）
python scripts/train_comfort.py --data-dir data/subject_001
python scripts/train_pinn.py --data-dir data/subject_001

# 仅在 base wrench 方向验证后运行
python scripts/run_control.py --subject subject_001
```

上述诊断默认将逐样本 CSV 与含摘要的 JSON 写入 `diagnostics/`，可用
`--output-dir` 改写。所有采样循环使用 `time.perf_counter_ns()` 统计本机
周期/查询时长；样本 age 与 skew 则沿用项目的 host-monotonic 时间戳定义。
手推与多姿态工具不会调用运动接口，姿态调整必须由训练合格的操作员在
控制器/HMI 的既定安全流程中完成。

采集顺序为：连接与 SDK/配置核对 → 参考点到位/稳定 → 人工无接触确认 → 软件 bias → 独立 50 Hz 采样线程 → 连续轨迹和静止尾段 → 原子写 CSV/episode JSON。运动完成由状态、到位误差和稳定性验证，不依赖固定 `sleep(0.5)` 假设。

## 离线验证

```bash
python3 -B -m pytest -q -p no:cacheprovider
```

测试使用 fake ROKAE，不连接机械臂，覆盖 SI 边界、快照时效/skew、bias、wrench 旋转/完整数学变换、切向/回程轨迹、CSV/异常元数据、采样线程、状态流异常和碰撞安全触发。
