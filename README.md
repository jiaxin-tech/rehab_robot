# 康复机器人控制系统

本项目用于上肢康复机器人实验：同步采集珞石机械臂状态与内置关节力矩传感器估算的末端六维力，生成患者关节激励轨迹，训练舒适度网络和逆 PINN，并用 MPC 进行在线轨迹控制。

## 架构

```text
rehab_robot/
├── config/settings.py                 # IP、频率、阈值和模型参数
├── hardware/
│   ├── windows/
│   │   ├── rokae_xcore.py             # 项目接口 ↔ 珞石 xCoreSDK 适配层
│   │   ├── rokae_force_sensor.py      # 内置关节力矩 → 末端六维力适配层
│   │   └── xcoresdk/                   # Windows xCoreSDK 0.7.0 运行文件
│   │       ├── xCoreSDK_python*.pyd    # CPython 3.12 / Win64
│   │       ├── xCoreSDK.dll
│   │       └── xCoreSDK_python/        # 类型声明与实时控制子模块
├── collection/
│   ├── safety_guard.py                # 独立力阈值安全停止
│   ├── trajectory.py                  # 标定、慢扫和激励轨迹
│   └── collector.py                   # 机械臂/力数据同步写入 CSV
├── models/
│   ├── comfort_net.py                 # 舒适度模型
│   └── pinn.py                        # 患者 M/B/K 在线辨识
├── control/mpc_controller.py          # MPC 控制器
└── scripts/                           # 采集、训练和控制入口
```

数据流：

```text
xCoreSDK状态 + 内置力估计 → DataCollector → episode CSV
                                        ├→ ComfortNet
                                        └→ inverse PINN → M/B/K

在线状态 + 力 → PINN + ComfortNet → MPC(50 Hz) → 最新笛卡尔目标
                                                    ↓
                                 xCoreSDK 1 ms 实时回调 → 机械臂
```

## xCoreSDK 环境

- Windows 10/11 64 位
- 64 位 CPython 3.12（仓库中的 `.pyd` 是 `cp312-win_amd64` 构建）
- xCore 控制器 V3.2 或更高版本
- xCoreSDK 0.7.0 Windows 运行文件位于 `hardware/windows/xcoresdk/`

适配层对上层统一使用以下单位：位置 mm、姿态 deg、关节角 deg、线速度 mm/s。调用 SDK 时会自动转换为其要求的 m 和 rad。

适配层启动时会读取 `BaseRobot.sdkVersion()` 并强制校验为 `0.7.0`，同时检查控制器报告的是 6 轴机器人。版本或机型不匹配时会在上电前终止。

在 `config/settings.py` 中至少确认：

```python
ROBOT_IP = "192.168.50.102"  # 珞石控制器 IP
ROBOT_LOCAL_IP = "192.168.50.10"  # 实时控制必填：本机同网段网卡 IP
ROBOT_CLASS = "xMateRobot"   # 当前项目按 6 轴协作机器人处理
ROBOT_CMD_CACHE = 1           # 避免在线路径命令积压
ROBOT_RT_NETWORK_TOLERANCE = 20 # SDK 实时网络容忍百分比
ROBOT_RT_FILTER_HZ = 50.0       # 实时位置指令滤波截止频率
ROBOT_FORCE_FRAME = "tool"    # 末端力输出坐标系
ROBOT_FORCE_HZ = 50            # getEndTorque 后台读取频率
```

机械臂负载、重心和工具坐标必须在 xCore 控制器工具组中正确配置，否则控制器由关节力矩推算出的末端六维力会包含明显的重力补偿误差。程序启动时只做当前会话的软件去零，不会自动执行硬件力传感器标定。

## 使用

```powershell
python -m pip install -r requirements.txt

# 0. 不上电、不运动，检查内置力数据和实际更新频率
python scripts/check_internal_force.py --duration 10

# 可选：同时检查软件去零后的漂移
python scripts/check_internal_force.py --duration 10 --software-bias

# 1. 采集数据
python scripts/run_collection.py --subject subject_001 --session session_01

# 2. 训练舒适度网络
python scripts/train_comfort.py --data-dir data/subject_001

# 3. 离线验证 PINN
python scripts/train_pinn.py --data-dir data/subject_001

# 4. 运行 PINN + MPC 控制
python scripts/run_control.py --subject subject_001
```

`run_collection.py` 的拖拽标定和慢速扫描使用 `NrtCommandMode + MoveLCommand`，50 Hz 连续激励阶段会切换到实时模式；`run_control.py` 也先用 NRT 安全到达参考起点，再让在线闭环使用 0.7.0 的 `RtCommandMode + cartesianPosition`。SDK 按控制器要求以 1 ms 调用回调函数，采集/MPC 主循环每 20 ms 更新一次回调所保持的最新笛卡尔目标。不要把 50 Hz 主循环直接改成每 20 ms 调用 `sendCommand()`；官方接口要求直接发送实时命令时必须维持 1 ms 间隔。

内部六维力来自 `getEndTorque()` 的模型估算，不是独立法兰六维力传感器的原始测量。首次用于人体实验前，应在空载和已知载荷下验证零漂、方向、幅值误差以及 50 Hz 连续读取稳定性。
