# 独立 TCP 牵引实验

当前主流程：独立设计 TCP 轨迹 → 机器人执行 → 假腿被动运动 → 记录实测角度、力矩和机器人力反馈。

完整接口顺序是：prepare 生成并冻结轨迹，run --live 执行一次，analyze 生成被动响应数据，repeat --source 用同一轨迹方案重新捕获锚点并建立重复 session，complete 在证据齐全后封存最终状态。

config/experiment.json 中 trajectory.mode 为 independent_tcp。该模式不读取髋膝参考曲线来生成 TCP 位置，不调整髋幅值或膝相位，不要求假腿跟踪目标关节角。旧联动规划仅保留用于历史结果和回归测试；旧 session 不能直接在新模式下运行。

## 轨迹参数

| 配置 | 当前值 | 含义 |
| --- | --- | --- |
| target_peak_tcp_z_base_m | 0.9 | 机器人基座坐标绝对 Z 峰值，单位 m |
| horizontal_displacement_m | -0.2 | 康复 X 方向相对位移，单位 m |
| lift_duration_s | 80 | 上抬段初始时间，单位 s |
| return_duration_s | 80 | 返回段初始时间，单位 s |
| start_hold_s | 2 | 起点保持时间，单位 s |
| peak_hold_s | 2 | 最高点保持时间，单位 s |
| sample_period_s | 0.1 | 规划目标点的最大时间间隔，单位 s |

0.9 m 沿用现有高度目标；-0.2 m 水平位移和上述时间是新增候选参数，尚未经真机验证。当前康复 X 对应基座 -Y，因此 -0.2 m 对应基座 +Y 方向移动 0.2 m。不要把绝对峰值 0.9 m 理解为从起点再上抬 0.9 m。

起点来自每次准备时捕获的真实 TCP。轨迹从起点沿斜直线到达最高点，保持后沿原路返回；TCP 姿态固定。每段采用七次时间函数 s(u)=35u^4-84u^5+70u^6-20u^7，解析曲线在两端速度、加速度和加加速度为零。SDK 仍在规划点之间做 1 ms 线性位置插值，执行曲线的导数连续性不能等同于解析曲线。

空载和连接假腿使用相同规划器、相同配置和同一套保守运动限制：取 motion 与 empty_load 速度、加速度限制的较小值。必要时自动延长上抬、返回时间，不改变路径；不同起点对应不同抬升距离。当前起点不触发延时的情况下总时间为 164 s。

## 现场操作

先确认 TCP、工具负载、坐标方向、工作空间和假腿安装状态。修改参数后必须新建 session，不能修改旧预览 CSV 后直接执行。

先准备空载，保留当前工具并拆除假腿：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --config config/experiment.json --empty-load --session results/tcp_empty_20261111_01
~~~

检查 round_status.json 为 PREPARED 且 blockers 为空；检查 tcp_plan.json、tcp_safety.json 和 preview/ 的 XYZ、速度、加速度及 3D 路径。逐点核对现场空间，而不只检查 Z 峰值。确认后显式执行：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session run --empty-load --live --session results/tcp_empty_20261111_01
~~~

核对 robot/robot_state.csv 中的实际位移、最高点和回程，不能仅凭 EMPTY_LOAD_COMPLETE 判定实体跟随成功。新轨迹验证通过后，安装假腿，再新建正式会话：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --config config/experiment.json --session results/tcp_passive_20261111_01
.\.venv\Scripts\python.exe -m scripts.lab_session run --live --session results/tcp_passive_20261111_01
~~~

安装后起点、路径和受力条件会改变，正式会话仍需重新核对预览。新模式不复用旧空载缩放系数或 linked_joint_reference.csv。准备后配置发生任何改变都会阻止执行，要求重新准备。

## 假腿保护与采集

假腿驱动应保持未使能、无故障。保留编码器物理角度范围、状态/力数据有效性、采集新鲜度、力限、力矩限、碰撞及实时停止保护。max_start_angle_error_deg 仅用于检测准备后、运动前假腿是否被移动；运动后不检查参考角跟踪误差。max_tracking_error_deg 不参与本模式。

正式实验同步记录机器人状态、六维力数据以及假腿编码器和力矩。leg_start_origin.json 只记录初始状态，不改变全局标定零点，也不参与 TCP 路径生成。锚点文件中的旧关节参考字段为兼容存储占位，本模式不使用。

## 实测响应分析

采集正常关闭后自动生成：

- passive_response.csv：标定后的实测髋膝角、原始关节力矩、匹配的机器人 TCP 位置及六维力数据。
- passive_response_summary.json：实测范围、匹配数量、时间容差和分析限制。

仅在时间容差内匹配有效机器人记录；跨大间隙不插值，未匹配记录保留为空值。机器人力查询时间采用主机查询起止中点近似，不能视为传感器硬件同步时间。保留 quality 的采样间隙等质量检查；该结果是描述性分析，不自动宣称可用于动力学辨识。

机器人 TCP 力向量模长不等于绑带纯牵引力；没有独立张力测量或经过验证的坐标、重力和工具补偿时，不将其标为绑带张力。关节力矩不默认减去静态偏置。

需要重新生成描述性分析时：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session analyze --session results/tcp_passive_20261111_01
~~~

analyze 只重新分析实测响应，不运行旧关节参考优化算法。train 作为旧命令别名保留；training.enabled 保持 false。


## 接口边界

当前模式不调用 experiment_algorithm 的六参数关节参考训练。该算法要求假腿实际角度跟踪预设髋膝参考曲线，而独立 TCP 实验的髋膝角是被动响应，二者不满足同一训练前提。历史主动关节参考实验仍可使用旧配置和 train 入口，但必须有有效 measured.csv、跟踪误差和动力学质量证据。

重复实验：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session repeat --source results/tcp_passive_20261111_01 --session results/tcp_passive_20261111_02
.\.venv\Scripts\python.exe -m scripts.lab_session repeat --source results/tcp_passive_20261111_01 --session results/tcp_passive_20261111_03 --live
~~~

repeat 以成功 session 为源，重新捕获 TCP 锚点，冻结相同 TCP 参数，并写入 parent_session 与 repeat_index。它不会复制旧的机器人状态、力数据、假腿 CSV 或旧锚点。

完成实验：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session complete --session results/tcp_passive_20261111_02
~~~

complete 是离线封存动作，会检查执行成功、实时清理、轨迹命令、机器人状态和响应证据，然后把 round_status.json 从 CAPTURE_COMPLETE 或 EMPTY_LOAD_COMPLETE 提升为 COMPLETE。它不会重新发送运动命令。
