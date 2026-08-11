# config/settings.py
# 全局配置，所有模块从这里读参数，不要在各文件里硬编码

# ── 全局单位与坐标约定 ──────────────────────────────
# 项目内部一律使用 SI：m、s、rad、kg、N、N·m。
# xCore realtime ``tcpPoseAbc_m`` 明确是相对于机器人 *base* 的位姿。
# 实验采集与轨迹命令统一以 base 为机器人坐标系。getEndTorque 不支持直接
# 请求 base wrench，因此原始力保留为 world，并单独记录 rotation_only
# 的 world -> base 坐标表达转换；其方向和力矩参考点必须按首次真机流程验证。
DATA_SCHEMA_VERSION = 3
CONTROL_FRAME       = "base"
# 保留旧的公开名字，所有新代码优先使用 CONTROL_FRAME。
PROJECT_FRAME       = CONTROL_FRAME
SI_UNITS            = "m,s,rad,kg,N,Nm"

# ── 硬件网络 ──────────────────────────────────────────
ROBOT_IP          = "192.168.50.103"
# xCoreSDK 0.7.0 Windows 配置。只读状态/wrench 采集可留空；只有显式、
# 已批准的 realtime motion preparation 才需要本机同网段的网卡 IP。
ROBOT_LOCAL_IP    = ""
ROBOT_CLASS       = "xMateRobot"   # 6轴协作机器人（xMate CR/SR/ER系列）
ROBOT_STATE_MS    = 8               # SDK支持 1/2/4/8/1000 ms
ROBOT_MAX_LINEAR_SPEED_M_S = 1.0     # 100%对应的末端线速度 (m/s)
ROBOT_CMD_CACHE   = 1               # 限制未规划路径点，避免在线控制指令堆积
# 以下两个旧 wrapper 默认仅供兼容/诊断；当前 gated executor 不把它们
# 当作批准值。真实执行要求 experiment_safety.json 中独立填写并审核 RT
# network tolerance，并把 reviewed_rt_filter_hz 显式传给 SDK。
ROBOT_RT_NETWORK_TOLERANCE = 20      # legacy enable_realtime compatibility only
ROBOT_RT_FILTER_HZ         = 50.0    # legacy enable_realtime compatibility only

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
TOOL_NAME = None
WORKPIECE_NAME = None
PAYLOAD_MASS_KG = None
PAYLOAD_COM_M = None

# ── 数据路径 ──────────────────────────────────────────
DATA_DIR           = "data"
LOG_DIR            = "logs"
