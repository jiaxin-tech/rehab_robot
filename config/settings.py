# config/settings.py
# 全局配置，所有模块从这里读参数，不要在各文件里硬编码

# ── 硬件网络 ──────────────────────────────────────────
ROBOT_IP          = "192.168.5.1"
ROBOT_DASH_PORT   = 29999
ROBOT_MOVE_PORT   = 30003
ROBOT_FEED_PORT   = 30004

SENSOR_IP         = "192.168.1.1"   # ATI力传感器，按实际修改
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

# ── 轨迹参数（需要标定后修改）──────────────────────────
# 关节旋转中心坐标 (mm)，标定脚本会自动写入
JOINT_CENTER      = [300.0, -200.0, 350.0]
JOINT_RADIUS      = 250.0           # 末端到旋转中心距离 (mm)
JOINT_ANGLE_MIN   = -0.6            # 关节角度下限 (rad)
JOINT_ANGLE_MAX   =  0.6            # 关节角度上限 (rad)
JOINT_NEUTRAL     =  0.0            # 中立位角度 (rad)

# 激励轨迹的正弦参数 [幅度(mm), 频率(rad/s)]
EXCITATION_PARAMS = [
    (40, 0.5),   # 慢速大幅
    (20, 1.5),   # 中速中幅
    (10, 3.0),   # 快速小幅
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

# ── 数据路径 ──────────────────────────────────────────
DATA_DIR           = "data"
LOG_DIR            = "logs"
