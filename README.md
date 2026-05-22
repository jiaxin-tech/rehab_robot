# 康复机器人控制系统

## 工程结构

```
rehab_robot/
├── config/
│   └── settings.py          # 全局配置（IP、频率、安全阈值等）
│
├── hardware/
│   ├── dobot_cr5.py         # Dobot CR5 TCP接口
│   └── force_sensor.py      # ATI力传感器 UDP接口
│
├── collection/
│   ├── safety_guard.py      # 安全守卫线程
│   ├── trajectory.py        # 激励轨迹生成
│   └── collector.py         # 数据采集主逻辑
│
├── models/
│   ├── comfort_net.py       # 舒适度神经网络（定义+训练+推理）
│   └── pinn.py              # Inverse PINN（定义+训练+在线辨识）
│
├── control/
│   └── mpc_controller.py    # MPC多目标控制器
│
├── utils/
│   ├── signal_processing.py # 滤波、微分等信号处理工具
│   └── logger.py            # 统一日志
│
├── scripts/
│   ├── run_collection.py    # 入口：数据采集
│   ├── train_comfort.py     # 入口：训练舒适度网络
│   ├── train_pinn.py        # 入口：训练/验证PINN
│   └── run_control.py       # 入口：运行完整控制系统
│
├── data/                    # 采集的CSV数据（git ignore）
└── logs/                    # 运行日志（git ignore）
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 采集数据
# 只采 PINN 数据：慢扫 + 激励，不标舒适度
python3 scripts/run_collection.py --subject subject_001 --session pinn_01 --collect-kind pinn
# 只采舒适度数据：跑康复轨迹，每段结束后让你标舒适度
python3 scripts/run_collection.py --subject subject_001 --session comfort_01 --collect-kind comfort --rehab-episodes 5
# 两者都采
python3 scripts/run_collection.py --subject subject_001 --session session_01 --collect-kind both

# 2. 训练舒适度网络
python scripts/train_comfort.py --data-dir data/subject_001

# 3. 验证PINN（用采集数据离线验证）
python scripts/train_pinn.py --data-dir data/subject_001

# 4. 运行完整控制系统
python scripts/run_control.py --subject subject_001
```

## 数据流

```
传感器 → collector → CSV文件
                         ↓
              train_comfort.py → comfort_net.pth
              train_pinn.py    → 验证辨识精度

实时控制：
传感器 → PINN(在线辨识M/B/K) ──→ MPC → 运动指令
       → ComfortNet(实时评分) ──→ MPC
```
