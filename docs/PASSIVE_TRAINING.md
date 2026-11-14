# 原六参数条件观测适配：仅作离线对照

用户已确定正式方法直接使用机械臂位姿与受力优化 TCP，操作见 [TCP_FORCE_EXPERIMENT.md](TCP_FORCE_EXPERIMENT.md)。下述几何 IK/JᵀF 接口只作有条件的离线对照，不能作为当前正式方案，公开运动入口已拒绝它的带载启动。不得用假设几何或假腿传感器绕过；取舍依据见[设计审核](FORCE_DRIVEN_DESIGN_REVIEW.md)。

更新：2026-11-12。假腿模拟被牵引的人腿，编码器和关节力矩仅用于台架验证、独立对照与装置保护。正式辨识不读取 leg/leg.csv，也不把假腿已知刚度、阻尼、质量真值作为模型输入。

此前直接使用假腿 q/dq/ddq 和关节力矩的接法不符合这个实验目标，已从正式训练入口移除。旧训练结果不能继续驱动正式候选推荐。六参数模型、原端点公式、残差 GP、约束和 Greedy 核心保持不变。

## 可复用的原观测关系

用户确认 TCP 与腿上牵引点是几何关系已知且不变的固定连接。measurement_validation/robot_observation.py 因此复用已有代码：

1. 机械臂实测 TCP 位姿，加上已标定的 TCP 到牵引点偏移，得到实际牵引点位置。
2. 转换到以髋中心为原点的下肢平面坐标系，调用 lower_limb_sim/kinematics.py 中已有 inverse_kinematics，重建模型姿态 q。
3. 按既有三阶 Savitzky–Golay 方法估计 dq、ddq；不读取候选名义角度作为实际姿态。
4. 对机械臂外力进行已标定的坐标、符号和空载偏置处理，调用 lower_limb_sim/observation_model.py 已有的 joint_torque_from_endpoint_force，即 tau_obs=J(q).T@[Fx,Fz]。
5. 把模型重建的 q/dq/ddq 和等效广义力矩输入原六参数训练器，再走物理端点预测、残差 GP、原约束和 Greedy。

这里 tau_hip_nm、tau_knee_nm 是原训练器所需的数据列名，数值来自机械臂观测与已有测量方程，不能称作直接测得的人体关节力矩。六参数描述被牵引腿的有效力学特性，机械臂提供观测。

parameter_estimator.py 中的旧五参数估计器不参与当前训练；本次只复用原 IK、Jacobian 和力观测方程，继续调用既有六参数/残差 GP/Greedy，不增加新的优化算法。

## 物理适用范围与标定

固定连接仍不足以无条件辨识。当前已有观测模型还要求：髋中心在实验中固定、两连杆有效长度及连接偏移已知、运动近似处于矢状面、逆运动学分支可用，并且牵引点近似只传递力。

机械臂六维力矩先移到连接点，检查剩余自由力矩。超过已标定阈值时拒绝训练；不忽略力矩，也不擅自新增全空间六维力观测模型。出平面运动/受力、不可达姿态、Jacobian 接近奇异、数据缺口、坐标不符也会拒绝。

SDK 返回的外力补偿状态和参考点不能仅凭接口名推定，必须核对工具、负载、坐标、参考点和作用/反作用符号。空载偏置应在无腿接触载荷时确定，不能把已经连接并受载的腿静态作用力清零当作校准。一个常量偏置不代表完成了动态工具补偿。

config/experiment.json 的 robot_observation 保存实际标定输入：

| 配置 | 含义 |
| --- | --- |
| hip_center_in_base_m | 髋中心在机器人基座中的位置 |
| tcp_to_pull_point_in_tcp_m | TCP 原点到腿上等效牵引点的固定偏移，表达在 TCP 坐标系 |
| wrench_origin_in_tcp_m | SDK 受力参考点相对 TCP 的位置，用于力矩移点 |
| L1_m、L2_m | 髋到膝、膝到等效牵引点的有效长度；不是刚度、阻尼等待辨识真值 |
| wrench_frame_type、rotation_base_from_wrench | 原力数据坐标系及其到基座的旋转 |
| force_robot_on_leg_sign | 将 SDK 力转换为机器人施加在腿上的力的符号 |
| unloaded_wrench_bias、compensation_verified | 无接触偏置和已确认的外力/工具补偿条件 |
| state_time_offset_s、wrench_time_offset_s | 实测状态和力记录时间的标定修正 |
| max_out_of_plane_position_m、max_out_of_plane_force_n、max_connection_moment_nm | 已有平面点力模型的适用误差预算 |

analysis 仍提供 max_gap_s、max_read_duration_s、max_dispatch_time_error_s、derivative_rate_hz、derivative_window_s。旧假腿 sensor_receive_delay_s、参考角跟踪门限不参与正式机器人观测训练。新标定缺失时不猜零、不套用假腿编码器或力矩。

## 原候选和 TCP 轨迹

trajectory.mode=original_candidate 仍按原算法选择候选 ID，读取其 CSV，使用已有 build_start_anchored_relative_trajectory 做 FK 相对位移转换。它不依赖实测腿角度，无需额外 beta 到 TCP 对应表。用于规划的名义候选与用于辨识的机器人实测姿态分开，不能把计划当观测。

prepare 原样复制 original_candidate.csv，保存 tcp_trajectory.csv、tcp_plan.json、配置和来源哈希，不改变原候选曲线、相位或时间。原候选到 TCP 的已有转换不意味着新的 0.9 m 高度目标已经满足。

## 历史接口与离线结果

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --config config/experiment.json --session results/original_01
.\.venv\Scripts\python.exe -m scripts.lab_session run --live --session results/original_01
.\.venv\Scripts\python.exe -m scripts.lab_session train --training-mode passive --session results/original_01
.\.venv\Scripts\python.exe -m scripts.lab_session repeat --source results/original_01 --trained --session results/original_02
~~~

上面的 prepare/run 示例属于改为离线对照前的历史接口，不应作为新的正式实验操作。当前带载 run 会拒绝 original_candidate 路径；离线 train 可处理满足原条件的已有记录。新正式方法的 run 自动训练和 repeat 使用直接 TCP 受力接口。

original_measured.csv 保存机器人观测重建的状态、力与等效力矩，observation_source=robot_tcp_point_force_v1。original_joint_metrics.json 标明 dummy_leg_data_used=false 和测量模型限制。original_training_request.json、original_training.json、training.log 保存原算法输入与结果。训练输入哈希不包含 leg/leg.csv；旧的假腿传感器训练结果会被正式推荐入口拒绝。

E3、E2、peak_ratio 仍按原公式计算，输入换为原观测方程重建的等效关节力矩，分母固定为首轮实测参考。不能将这些指标写成直接人体关节力矩实测值。

原算法的重复候选、历史追加、STOP 和 DOMAIN_EXHAUSTED 限制保留。单参考自身归一化为 1 是定义，不是优化收益。

## 当前完成程度

软件已接通机器人观测适配与原六参数训练器，合成测试验证不读取假腿数据、机器人受力影响观测、坐标/符号/力矩移点以及无效条件拦截。原核心文件没有修改。

本离线对照的 robot_observation 标定未填写，原参考仍是 24 s/401 点，旧起点约 0.5282 m，原目录仅一个参考。这些数据不属于当前正式默认轨迹；正式独立 TCP 已使用 0.9 m 目标及三种节奏，详见新指南。两种方法均不能在没有真实实验时宣称辨识或优化收益。

完整操作见 [INDEPENDENT_TCP_EXPERIMENT.md](INDEPENDENT_TCP_EXPERIMENT.md)，论文说明见 [PAPER_ARCHITECTURE.md](PAPER_ARCHITECTURE.md)，审核见 [PAPER_CODE_AUDIT.md](PAPER_CODE_AUDIT.md)。
