# 机械臂受力驱动的下肢牵引实验

2026-11-12：正式主线根据机械臂实测 TCP 和六维受力评价、选择下一 TCP 轨迹，不要求患者姿态、髋中心、连接偏移或腿长，不读取假腿角度或关节力矩。原六参数/物理先验/残差 GP 只保留为离线对照。

流程：独立 TCP 候选 → ROKAE 执行 → 审核实际任务完成与数据 → 直接受力指标 → 复用已有 GP/受约束候选决策 → 下一轮。学习发生在轮间，SDK 仍为实时笛卡尔位置控制。

详细操作、指标、默认候选和配置见 [TCP 受力实验指南](docs/TCP_FORCE_EXPERIMENT.md)。论文与证据见 [论文架构](docs/PAPER_ARCHITECTURE.md)、[代码审核](docs/PAPER_CODE_AUDIT.md)。

默认比较相同空间端点下的三种节奏：80/80、100/80、80/100 秒上抬/返回，起点和最高点各保持 2 秒。Z 目标沿用 0.9 m，保守速度/加速度限制保留。接口支持在明确任务区间内比较水平位移，默认固定 -0.2 m。规定行程、时长和实际跟随检查防止以少运动换低受力。

trajectory.mode=independent_tcp、training.mode=tcp_force。force_training 的机器人无接触偏置和补偿条件尚需现场核对，默认 null/false 会阻止正式带载学习；这些不是患者几何参数。空载不训练。validation.capture_dummy_leg=true 用于台架验证及保护；false 时观测、运动、训练和完成检查均不依赖假腿/ADS。

基础命令：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session check --config config/experiment.json
.\.venv\Scripts\python.exe -m scripts.lab_session observe --session results/observe_01 --duration 30
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --session results/force_01
.\.venv\Scripts\python.exe -m scripts.lab_session run --session results/force_01 --live
.\.venv\Scripts\python.exe -m scripts.lab_session repeat --source results/force_01 --trained --session results/force_02
~~~

prepare 不运动；先核对阻断项与预览。run 会运动并在结束后自动训练；repeat --trained 只准备下一轮。先完成短诊断和空载，再按现场条件进行带载实验。SDK 与停止流程见 [真机操作手册](docs/REAL_ROBOT_RUNBOOK.md)。

离线回归：

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_mainline.py tests/test_original_training_adapter.py tests/test_tcp_force_training.py tests/test_realtime_link_diagnostic.py tests/test_restore_session_tcp.py --junitxml=.cache/paper_audit_tests.xml
~~~

软件测试通过不代表真机优化或临床效果。机器人任务完成不等于患者关节活动范围已验证。当前没有新的多轮真机受力优化结果，不虚构收益。

其他说明：[实验指南](docs/EXPERIMENT.md)、[代码地图](docs/CODE_MAP.md)、[设计取舍](docs/FORCE_DRIVEN_DESIGN_REVIEW.md)、[原六参数条件适配记录](docs/PASSIVE_TRAINING.md)。
