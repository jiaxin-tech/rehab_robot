# config/settings.py
# 全局配置，所有模块从这里读参数，不要在各文件里硬编码

# ── 硬件网络 ──────────────────────────────────────────
ROBOT_IP          = "192.168.50.102"
ROBOT_DASH_PORT   = 29999
ROBOT_MOVE_PORT   = 30003
ROBOT_FEED_PORT   = 30004

SENSOR_IP         = "192.168.50.200"   # ATI力传感器，按实际修改
SENSOR_PORT       = 49152
SENSOR_HZ         = 100             # 1000 / SPEED(=10)
SENSOR_FILTER     = 4               # 15Hz低通
FORCE_DIV         = 10000.0
TORQUE_DIV        = 100000.0

# ── 采集参数 ──────────────────────────────────────────
COLLECT_HZ        = 50              # 数据存储频率
COLLECT_DT        = 1.0 / COLLECT_HZ

# ── 安全参数 ──────────────────────────────────────────
MAX_FORCE_N       = 30.0            # 合力上限 (N)，超过立刻停
SAFETY_CHECK_HZ   = 200            # 安全监测频率
INIT_SPEED_RATIO  = 5               # 初始速度比例 (1~100)

# 真实机器人康复轨迹安全边界：用于拒绝越过人体标定活动范围、过快往复等指令。
# 明显危险样本请用 synthetic_risk 离线生成，不下发到机器人。
REHAB_MAX_REAL_RANGE_SCALE     = 0.95
REHAB_MAX_REAL_CYCLES_PER_SEC  = 0.30
REHAB_MIN_REAL_DURATION        = 8.0

# ── 轨迹参数（需要标定后修改）──────────────────────────
# 关节旋转中心坐标 (mm)，标定脚本会自动写入
JOINT_CENTER      =[-133.82024511942845, -666.6341456164093, 54.12649990352616]
JOINT_RADIUS      = 298.7           # 末端到旋转中心距离 (mm)
JOINT_ANGLE_MIN   =0.2221          # 关节角度下限 (rad)
JOINT_ANGLE_MAX   = 1.3218            # 关节角度上限 (rad)
JOINT_NEUTRAL     = 0.9510            # 中立位角度 (rad)

# 夹爪向下时的末端姿态 [rx, ry, rz] (deg)。
# None 表示运行脚本启动时读取当前姿态，并在整条轨迹中锁定不变。
TOOL_DOWN_ORIENTATION = None

# 激励轨迹的正弦参数 [幅度(mm), 频率(rad/s)]
EXCITATION_PARAMS = [
    (40, 0.5),   # 慢速大幅
    (20, 1.5),   # 中速中幅
    (10, 3.0),   # 快速小幅
]
EXCITATION_DURATION = 5.0          # 每段激励时长 (s)

# 康复轨迹参数：用于真实康复动作和舒适度判断
REHAB_DURATION      = 20.0          # 每段康复轨迹时长 (s)
REHAB_CYCLES        = 3.0           # 每段内往复次数
REHAB_RANGE_SCALE   = 0.8           # 使用标定活动范围的比例，避免贴近极限
CONTROL_TRAJECTORY_REPEATS = 2      # control中同一条康复轨迹连续跑几次
REHAB_VARIANTS      = [
    {"name": "small_slow",  "range_scale": 0.50, "cycles": 2.0, "duration": 24.0},
    {"name": "medium",      "range_scale": 0.70, "cycles": 3.0, "duration": 20.0},
    {"name": "large_slow",  "range_scale": 0.90, "cycles": 2.0, "duration": 24.0},
    {"name": "small_fast",  "range_scale": 0.50, "cycles": 4.0, "duration": 16.0},
    {"name": "large_fast",  "range_scale": 0.90, "cycles": 4.0, "duration": 16.0},
]

# 真实可采集的“更容易不适”轨迹：仍在安全边界内，采集后由受试者/操作者标注。
REHAB_UNCOMFORTABLE_VARIANTS = [
    {"name": "safe_near_limit_slow", "range_scale": 0.95, "cycles": 2.5, "duration": 22.0},
    {"name": "safe_near_limit_fast", "range_scale": 0.90, "cycles": 5.0, "duration": 18.0},
    {"name": "safe_midrange_fast",   "range_scale": 0.70, "cycles": 4.5, "duration": 15.0},
]

# 离线合成的危险/明显不适负样本：只写CSV，不会控制机器人，comfort自动标为2。
SYNTHETIC_RISK_VARIANTS = [
    {
        "name": "unsafe_sim_range_over_joint",
        "range_scale": 1.25,
        "cycles": 3.0,
        "duration": 16.0,
        "force_n": 32.0,
        "stiffness_scale": 4.0,
        "damping_scale": 3.0,
    },
    {
        "name": "unsafe_sim_high_stiffness",
        "range_scale": 0.85,
        "cycles": 4.0,
        "duration": 16.0,
        "force_n": 36.0,
        "stiffness_scale": 8.0,
        "damping_scale": 4.0,
    },
    {
        "name": "unsafe_sim_high_accel",
        "range_scale": 0.75,
        "cycles": 8.0,
        "duration": 8.0,
        "force_n": 28.0,
        "stiffness_scale": 5.0,
        "damping_scale": 5.0,
    },
]

# ── 触觉传感器（未购入时保持None，接口预留）────────────
TACTILE_IP         = None           # 购入后填写，如 "192.168.1.2"
TACTILE_PORT       = 50000          # 按实际修改
TACTILE_DIM        = 16            # 触觉传感器输出维度，按实际修改

# ── 舒适度网络 ────────────────────────────────────────
# 默认输入：PINN每帧辨识参数 + 力数据
#   "pinn_force"    → [Mx,My,Mz,Bx,By,Bz,Kx,Ky,Kz,Fx,Fy,Fz] 维度=12
COMFORT_INPUT_MODE = "pinn_force"
COMFORT_INPUT_DIM  = 12             # 会被 comfort_net.py 根据mode自动计算，这里是默认值
COMFORT_HIDDEN     = [64, 32]
COMFORT_LR         = 1e-3
COMFORT_EPOCHS     = 100
COMFORT_BATCH      = 64
COMFORT_MODEL_PATH = "models/comfort_net.pth"
COMFORT_THRESHOLD  = 0.5            # MPC硬约束下限

# ── PINN ──────────────────────────────────────────────
PINN_HIDDEN_LAYERS = [64, 64, 64]
PINN_LR            = 1e-3
PINN_EPOCHS        = 2000
PINN_LAMBDA        = 0.1            # 物理损失权重
PINN_ONLINE_ENABLED = True         # 控制阶段是否后台更新PINN参数；可用命令行 --online-pinn/--no-online-pinn 覆盖
PINN_ONLINE_UPDATE_EVERY = 100      # 每隔多少控制步尝试启动一次后台更新
PINN_ONLINE_MIN_SAMPLES  = 50       # 后台更新所需最少窗口帧数
PINN_ONLINE_EPOCHS       = 300      # 后台PINN单次训练轮数
# M/B/K 初始猜测值：项目内部统一使用mm
PINN_M_INIT        = 0.001          # N·s²/mm，约等于 1 kg
PINN_B_INIT        = 0.0005         # N·s/mm，约等于 0.5 N·s/m
PINN_K_INIT        = 0.01           # N/mm，约等于 10 N/m

# ── MPC ───────────────────────────────────────────────
MPC_DIM            = 3              # MPC任务空间维度，当前使用xyz三维
MPC_HORIZON        = 8              # 预测步数；20步在Python/SciPy下容易超过50Hz控制周期
MPC_DT             = 0.03           # 控制周期 (s) ≈ 33Hz，匹配Dobot ServoP建议最小30ms周期
MPC_W_TRACKING     = 1.0            # 轨迹跟踪权重
MPC_W_POS          = 1.0            # 位置跟踪子权重
MPC_W_VEL          = 0.2            # 速度跟踪子权重
MPC_W_COMFORT      = 10.0          # 不舒适时的控制平滑权重；comfort优先时可调大
MPC_W_JERK         = 1.0            # jerk惩罚基础权重，抑制加速度突变
MPC_TRACKING_MIN_SCALE = 0.1        # comfort=0时tracking权重比例，越小越让位给舒适度
MPC_JERK_COMFORT_GAIN  = 10.0       # comfort低时jerk权重增益
MPC_A_MAX          = 500.0          # 最大加速度指令 (mm/s²)
MPC_POS_SCALE      = 100.0          # 位置误差归一化尺度 (mm)
MPC_VEL_SCALE      = 200.0          # 速度误差归一化尺度 (mm/s)

# ── 控制运行保护 ─────────────────────────────────────
CONTROL_START_TOLERANCE_MM = 2.0    # 控制前当前位置离轨迹起点超过该值则先MovL到起点
CONTROL_MAX_FEEDBACK_JUMP_MM = 80.0 # 单周期反馈跳变超过该值视为反馈异常并停机
CONTROL_MAX_TRACK_ERROR_MM = 200.0  # 实际位姿偏离当前目标过大时停机，防止反馈/控制发散
CONTROL_MAX_TARGET_STEP_MM = 100.0    # MPC单周期目标点最大位移，限制突变指令
CONTROL_MIN_TARGET_STEP_MM = 0.5    # comfort很低时单周期目标点最大位移下限
CONTROL_MIN_PROGRESS_SCALE = 0.05   # comfort很低时参考轨迹最慢推进比例
CONTROL_COMFORT_RECOVERY_POWER = 2.5 # 控制用comfort恢复到1的速度；越大恢复越快
CONTROL_COMFORT_SPEED_POWER = 2.0   # comfort低时速度/步长衰减强度；越大越保守
CONTROL_LOW_COMFORT_RESET_THRESHOLD = 0.35 # 低于该值时清空MPC热启动，避免沿旧速度滑行
CONTROL_COMFORT_FORCE_FLOOR_N = 12.0 # 合力低于该值时，避免控制comfort被模型瞬时0值锁死
CONTROL_COMFORT_LOW_FORCE_FLOOR = 0.35 # 低力时comfort_ctrl的最高下限，随力增大线性衰减
CONTROL_VEL_FILTER_ALPHA = 0.30     # 反馈速度低通系数，降低单帧抖动对MPC的影响
CONTROL_MAX_MPC_REF_DEVIATION_MM = 35.0 # MPC目标相对当前参考点最大偏离，防止跑飞

# ── 数据路径 ──────────────────────────────────────────
DATA_DIR           = "data"
LOG_DIR            = "logs"
