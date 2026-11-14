# 正式 TCP 受力方法与离线对照审核

2026-11-12。用户已确认优先实际康复条件：机械臂位姿与受力直接驱动 TCP 候选决策，原六参数只作离线对照。没有提供论文原稿，不编造原文或真机结果。

正式入口为 lab_session.py → tcp_force_training.py。它复用独立 TCP 规划、MaternGP、期望改进和模型约束评分，不调用六参数拟合、患者 IK/JᵀF 或假腿传感器。输出名为 load_score、force_peak_fraction、torque_peak_fraction，明确区别原 E3/E2 和生理参数。

| 审核项目 | 当前行为 |
| --- | --- |
| 未知患者姿态/几何 | 不作为正式配置输入；删除相关配置的合成测试仍能完成流程 |
| 假腿传感器 | 仅作可选台架验证与保护，关闭后 observe/prepare/run/train/complete 不依赖 ADS |
| 任务完成 | 限制高度、位移和时间，核对实际 TCP；低力但不完成行程会拒绝 |
| 受力指标 | 力和力矩移动分支 RMS 等权归一化，全周期峰值约束，失效数据不当作零 |
| 候选 | 有限目录，使用实际延时后的位移/时间特征；默认三种相同空间端点的节奏 |
| 决策 | 现有 GP 和 EI/可行性工具，保留预算、无可行候选、域耗尽和改善不足停止 |
| 重复 | 正式方法保存相同 ID 的真实重复轮次并更新，噪声/依赖局限明确记录 |
| 预测验证 | 运动前冻结下一轮预测，运动后计算误差 |
| 证据 | 配置、任务、原始机器人数据和代码哈希，输入变化时拒绝旧推荐 |

原六参数核心 config.py、surrogate.py、policy.py、coupled_model.py、coordinate_consistent_model.py、coordinates.py 无改动，原重复限制等行为保留。旧几何适配与 worker 留作有条件的离线对照，不作为临床主线，也不重定义六参数含义。

机器人外力补偿、受力坐标与参考点、无接触偏置、工具负载和时效仍需实验核对。当前 force_training 的补偿为 false、偏置为 null，正式带载学习会报告阻断；不会要求患者髋中心等参数或自动填零。空载不训练。

GP 噪声、可行性门限和目标权重是工程定义，不是已校准临床安全概率。患者姿态变化、跨轮漂移和重复依赖尚未建模，机器人任务完成不能证明患者关节活动或疗效。

测试仅使用合成数据与离线故障注入，没有运行真机、改 SDK 二进制或改变原 results/。软件回归不证明受力优化收益。用户未同步的新实验不作否定判断。

操作见 [TCP_FORCE_EXPERIMENT.md](TCP_FORCE_EXPERIMENT.md)，论文架构见 [PAPER_ARCHITECTURE.md](PAPER_ARCHITECTURE.md)，复算证据见 [PAPER_EVIDENCE_SNAPSHOT.json](PAPER_EVIDENCE_SNAPSHOT.json)。
