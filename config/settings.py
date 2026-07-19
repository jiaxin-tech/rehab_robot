# config/settings.py
# 全局配置，所有模块从这里读参数，不要在各文件里硬编码

# ── 全局单位与坐标约定 ──────────────────────────────
# 项目内部一律使用 SI：m、s、rad、kg、N、N·m。
# xCore realtime ``tcpPoseAbc_m`` 明确是相对于机器人 *base* 的位姿。
# PINN、MPC 和轨迹也统一以 base 为算法坐标系。getEndTorque 不支持直接
# 请求 base wrench，因此原始力保留为 world，并单独记录 rotation_only
# 的 world -> base 坐标表达转换；其方向和力矩参考点必须按首次真机流程验证。
DATA_SCHEMA_VERSION = 3
CONTROL_FRAME       = "base"
# 保留旧的公开名字，所有新代码优先使用 CONTROL_FRAME。
PROJECT_FRAME       = CONTROL_FRAME
SI_UNITS            = "m,s,rad,kg,N,Nm"

# ── 硬件网络 ──────────────────────────────────────────
ROBOT_IP          = "192.168.50.103"
# xCoreSDK 0.7.0 Windows 配置。只做 NRT 慢扫时可留空；运行 50 Hz
# 连续激励采集或 run_control.py 时必须填写本机同网段的网卡 IP。
ROBOT_LOCAL_IP    = ""
ROBOT_CLASS       = "xMateRobot"   # 6轴协作机器人（xMate CR/SR/ER系列）
ROBOT_STATE_MS    = 8               # SDK支持 1/2/4/8/1000 ms
ROBOT_MAX_LINEAR_SPEED_M_S = 1.0     # 100%对应的末端线速度 (m/s)
ROBOT_CMD_CACHE   = 1               # 限制未规划路径点，避免在线控制指令堆积
ROBOT_RT_NETWORK_TOLERANCE = 20      # 实时网络丢包/延迟容忍百分比 (0~100)
ROBOT_RT_FILTER_HZ         = 50.0    # 实时位置指令低通截止频率 (1~1000 Hz)

# 珞石内置关节力矩/动力学估计。没有外接 UDP/ATI 六维力传感器。
# getEndTorque(v0.7.0) 文档仅支持 world/flange/tool；采集始终请求 world。
ROBOT_FORCE_SOURCE             = "rokae_force_control_get_end_torque"
ROBOT_FORCE_RAW_FRAME          = "world"
# 兼容旧调用方；它表示 raw frame，不是算法 frame。
ROBOT_FORCE_FRAME              = ROBOT_FORCE_RAW_FRAME
ROBOT_FORCE_HZ                 = 50       # 独立 SDK wrench 更新线程频率
ROBOT_FORCE_BIAS_SAMPLES       = 50       # 通常由 FORCE_BIAS_DURATION_S 覆盖
ROBOT_FORCE_STALE_S            = 0.05     # 力估计超过此时间必须失效
FORCE_BIAS_DURATION_S          = 1.0
FORCE_BIAS_REQUIRE_CONFIRMATION = True
BASE_WRENCH_TRANSFORM_KIND     = "rotation_only"
# Keep false until the documented empty-load and known-direction checks have
# verified the baseFrame/Euler direction on this exact robot/tool setup.
BASE_WRENCH_ROTATION_VERIFIED  = False

# ── 采集参数 ──────────────────────────────────────────
COLLECT_HZ        = 50              # 数据存储频率
COLLECT_DT        = 1.0 / COLLECT_HZ
MAX_ROBOT_STATE_AGE_S = 0.05         # 最新实时状态距采样时刻的最大年龄
MAX_FORCE_SAMPLE_AGE_S = 0.05        # 最新 getEndTorque 样本最大年龄
MAX_INTERNAL_STATE_SKEW_S = 0.02     # pose/joint 与 wrench 的最大内部偏差
WRITE_INVALID_SAMPLES = True          # 离线采集保留 valid=0 行，不伪造有效数据
POST_MOTION_RECORD_S = 1.0
ARRIVAL_POSITION_TOLERANCE_M = 0.002
STABLE_TCP_SPEED_MPS = 0.005
STABLE_DURATION_S = 0.5

# ── 安全参数 ──────────────────────────────────────────
MAX_FORCE_N       = 30.0            # 去偏后的 base 合力上限 (N)，真机前复核
MAX_CARTESIAN_TORQUE_NM = 8.0       # 去偏后的 base 力矩上限 (N·m)，真机前复核
# 各关节外部力矩阈值(N·m)。这些是项目安全配置，不是 SDK 默认值；必须在
# 首次人体实验前按机器人型号/工具/人群复核。
MAX_JOINT_EXTERNAL_TORQUE_NM = [20.0, 20.0, 20.0, 12.0, 12.0, 8.0]
MAX_TCP_SPEED_MPS = 0.20
MAX_TARGET_STEP_M = 0.005
# None 表示尚未配置该工作空间边界；SafetyGuard 会在 metadata 中标出。
WORKSPACE_MIN_M = None
WORKSPACE_MAX_M = None
# Human-facing collection/control is fail-closed until a site-specific Cartesian
# workspace is configured.  The standalone force diagnostic does not move the
# robot and is intentionally unaffected.
REQUIRE_WORKSPACE_LIMITS = True
JOINT_SOFT_LIMIT_MARGIN_RAD = 0.05
REQUIRE_JOINT_SOFT_LIMITS = True
# xCoreSDK exposes a safety-event query but this project deliberately does not
# invent model-specific collision thresholds. Confirm controller/HMI collision
# detection before enabling human-facing collection or control.
CONTROLLER_COLLISION_CONFIGURATION_CONFIRMED = False
REQUIRE_COLLISION_STATE_QUERY = True
SAFETY_CHECK_HZ   = 200            # 安全监测频率
INIT_SPEED_RATIO  = 5               # 初始速度比例 (1~100)

# ── episode / 可追溯性 ─────────────────────────────
# 以下参数必须与控制器已设置的 toolset/load 一致；代码不会替用户写入控制器。
TRAJECTORY_NAME = "rehab_xz_arc"
TRAJECTORY_VERSION = "1"
TOOL_NAME = None
WORKPIECE_NAME = None
PAYLOAD_MASS_KG = None
PAYLOAD_COM_M = None

# ── 轨迹参数（需要标定后修改）──────────────────────────
# 关节旋转中心坐标 (m)，标定脚本会自动输出
JOINT_CENTER      = [0.300, -0.200, 0.350]
JOINT_RADIUS      = 0.250           # 末端到旋转中心距离 (m)
JOINT_ANGLE_MIN   = -0.6            # 关节角度下限 (rad)
JOINT_ANGLE_MAX   =  0.6            # 关节角度上限 (rad)
JOINT_NEUTRAL     =  0.0            # 中立位角度 (rad)

# 激励轨迹的正弦参数 [弧长幅度(m), 角频率(rad/s)]
EXCITATION_PARAMS = [
    (0.040, 0.5),   # 慢速大幅
    (0.020, 1.5),   # 中速中幅
    (0.010, 3.0),   # 快速小幅
]
EXCITATION_DURATION = 10.0          # 每段激励时长 (s)

# ── 触觉传感器（未购入时保持None，接口预留）────────────
TACTILE_IP         = None           # 购入后填写，如 "192.168.1.2"
TACTILE_PORT       = 50000          # 按实际修改
TACTILE_DIM        = 16            # 触觉传感器输出维度，按实际修改

# ── 舒适度网络 ────────────────────────────────────────
# input_mode 三选一:
#   "force"         → 只用力传感器   输入维度 = 9  (Fx,Fy,Fz, x,y,z, vx,vy,vz)
#   "tactile"       → 只用触觉传感器 输入维度 = 3 + TACTILE_DIM (x,y,z + tactile)
#   "force+tactile" → 两者拼接      输入维度 = 9 + TACTILE_DIM
COMFORT_INPUT_MODE = "force"        # 触觉传感器到位后改为 "force+tactile"
COMFORT_INPUT_DIM  = 9              # 会被 comfort_net.py 根据mode自动计算，这里是默认值
COMFORT_HIDDEN     = [64, 32]
COMFORT_LR         = 1e-3
COMFORT_EPOCHS     = 200
COMFORT_BATCH      = 64
COMFORT_MODEL_PATH = "models/comfort_net.pth"
COMFORT_THRESHOLD  = 0.5            # MPC硬约束下限

# ── PINN ──────────────────────────────────────────────
PINN_HIDDEN_LAYERS = [64, 64, 64]
PINN_LR            = 1e-3
PINN_EPOCHS        = 2000
PINN_LAMBDA        = 0.1            # 物理损失权重
# M/B/K 初始猜测值（量级对了就行）
PINN_M_INIT        = 1.0            # kg
PINN_B_INIT        = 0.5            # N·s/m
PINN_K_INIT        = 10.0           # N/m

# ── MPC ───────────────────────────────────────────────
MPC_HORIZON        = 20             # 预测步数
MPC_DT             = 0.02           # 控制周期 (s) = 50Hz
MPC_W_TRACKING     = 1.0            # 轨迹跟踪权重
MPC_W_COMFORT      = 2.0            # 舒适度权重
MPC_W_FORCE        = 0.5            # 受力最小化权重
MPC_MAX_ACCEL_M_S2 = 0.5            # 末端线加速度约束 (m/s²)
# 仅供旧的单轴 Cartesian MPC 兼容接口使用；默认康复控制使用路径弧长。
CONTROL_AXIS       = 2              # base 坐标轴：0=x, 1=y, 2=z

# ── 数据路径 ──────────────────────────────────────────
DATA_DIR           = "data"
LOG_DIR            = "logs"
