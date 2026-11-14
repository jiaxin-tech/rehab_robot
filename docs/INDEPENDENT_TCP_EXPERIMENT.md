# 独立 TCP 实验入口

2026-11-12：用户已确定优先实际康复条件，正式流程改为机械臂位姿与受力驱动 TCP 候选优化。完整操作、目标函数、候选、重复与证据文件见 [TCP_FORCE_EXPERIMENT.md](TCP_FORCE_EXPERIMENT.md)。

默认 trajectory.mode=independent_tcp、training.mode=tcp_force。规划和训练不需要患者姿态、髋中心、连接偏移或腿长，不使用假腿角度/力矩；台架传感器仅作验证和保护，validation.capture_dummy_leg=false 可运行仅机器人流程。

算法轮间更新，仍使用既有实时笛卡尔位置执行链路。旧原候选 FK/IK/JᵀF 与六参数方法只作离线对照，不通过猜测几何使检查通过。设计依据见 [FORCE_DRIVEN_DESIGN_REVIEW.md](FORCE_DRIVEN_DESIGN_REVIEW.md)。
