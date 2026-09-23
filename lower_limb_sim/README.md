# 下肢力学、轨迹与离线实验

本包提供仰卧位髋膝康复的二维力学基础、参考轨迹处理、离线辨识及仿真实验。整个项目的当前研究结论与入口见 [研究指南](../docs/RESEARCH.md)，安装和核心回归见 [开始使用](../docs/GETTING_STARTED.md)。

模型始终采用 `theta_shank = q_hip - q_knee`。`L1` 为髋到膝长度，`L2` 为膝到束带等效牵引点的距离；`L2` 不代表完整小腿长度或小腿质心距离。

| 需要查看的内容 | 主要模块 |
|---|---|
| 运动学、Jacobian、全动力学与力映射 | [kinematics.py](kinematics.py)、[jacobian.py](jacobian.py)、[full_dynamics.py](full_dynamics.py)、[force_mapping.py](force_mapping.py) |
| 五参数时序辨识 | [parameter_estimator.py](parameter_estimator.py)；当前灰箱 GP/EI 集成位于 [personalization](../personalization/) |
| 共享 E0/E2 指标 | [mechanical_endpoints.py](mechanical_endpoints.py) |
| 冻结参考轨迹与机器人导出 | [reference_release.py](reference_release.py)、[reference_measured_asymmetric.py](reference_measured_asymmetric.py)、[run_robot_trajectory_export.py](run_robot_trajectory_export.py) |
| 五腿 V3/E2 仿真与必要性分析 | [five_leg_mujoco_v1](five_leg_mujoco_v1/) |
| 冻结低预算算法比较 | [frozen_multi_leg_algorithm_benchmark_v1](frozen_multi_leg_algorithm_benchmark_v1/) |
| 后续独立三参数 E3 探索 | [e3_candidate_comparison](e3_candidate_comparison/)、[e3_low_budget](e3_low_budget/) |
| 离线视频展示 | [visualization](visualization/)；命令见 [视频指南](../docs/visualization/README.md) |

`run_*.py`、P2、连续参考邻域及 `formal_artifacts/` 还保留早期研究阶段。它们各自的参数域、指标与冻结结果不能互相替代。原有 Stage 1–6 长篇说明已完整归档为 [下肢建模与参考轨迹开发手册](../docs/history/LOWER_LIMB_STAGES_1_TO_6.md)，其中保留原命令、公式及阶段边界；历史命令可能依赖单独保存的大型数据。

机器人操作和真实测量入口见 [机器人操作指南](../docs/ROBOT_OPERATIONS.md)。离线仿真结果不直接批准真实轨迹执行。
