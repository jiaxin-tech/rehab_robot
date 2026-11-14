# 面向机械负荷降低的单臂假腿轨迹适配

2026-11-14。当前分支整合自 `cleanup/unified-dummy-leg-mainline`，保留机械臂实测 TCP 和六维受力驱动的正式方法。论文研究受控机械条件下的轨迹适配，以降低机械负荷作为舒适度相关设计目标；没有人体舒适评价或临床效果结论。

**从[当前论文与实验工作流](docs/CURRENT_WORKFLOW.md)进入。** 旧电脑先按[本地检查与数据回传命令](docs/MANUAL_LAB_COMMANDS_2026-11-14.md)完成假腿诊断；分支来源见[整合记录](docs/BRANCH_INTEGRATION_2026-11-14.md)，完整机器人实验见[TCP受力实验指南](docs/TCP_FORCE_EXPERIMENT.md)。

唯一正式流程为：独立 TCP 候选 → ROKAE 执行 → 实际任务与原始数据审核 → 直接受力指标 → 有限候选决策 → 下一轮重新准备并显式启动。学习发生在轮间，SDK仍执行实时笛卡尔位置控制。训练不需要患者姿态、髋中心、连接偏移或腿长，也不读取假腿角度/关节力矩；假腿数据可保留作独立台架验证和保护。

`config/experiment.json`仍是实际操作配置，本次不替换其现场安装或限值。其原三候选80/80、100/80、80/100秒对应164/184/184秒总周期，只作不同节奏的先导，不能称相同总时长比较。

新增[论文配置模板](config/paper_personalized_load.template.json)提出80/80、70/90、90/70秒的同空间任务、164秒固定周期比较，**仅供现场审核**。模板的安装/机械条件身份未填写，补偿和偏置仍为false/null，训练未启用。保守重定时若使实际候选周期改变，本批固定周期比较应被拒绝；不得放宽限速来通过。0.9m目标、-0.2m水平位移与原限值均仅沿用配置，不是新的现场适用性证明。

个性化的验证必须包含开发条件中冻结的公共候选、等成本简单选择和独立复测。`TCP_FORCE_GP_EI`是待验证方法，不能把GP/EI名称或相对原参考的改善当成个性化创新。六参数/物理先验/残差GP和旧E2/E3仅作有条件的离线对照。

基础命令：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session check --config config/experiment.json
.\.venv\Scripts\python.exe -m scripts.lab_session observe --config config/experiment.json --session results/observe_01 --duration 30
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --config config/experiment.json --session results/force_01
.\.venv\Scripts\python.exe -m scripts.lab_session run --session results/force_01 --live
.\.venv\Scripts\python.exe -m scripts.lab_session repeat --source results/force_01 --trained --session results/force_02
~~~

observe访问设备，prepare读取起点但不运动；先核对阻断项和完整预览。run会运动，训练是否执行取决于已审核配置；repeat --trained只准备下一轮。先完成短诊断与空载，再按现场条件进行带载。SDK与停止流程见[真机操作手册](docs/REAL_ROBOT_RUNBOOK.md)。模板不能直接作为已审核运动配置。

离线回归：

~~~powershell
.\.venv\Scripts\python.exe -m pytest -c pytest-core.ini -q --junitxml=.cache/paper_branch_tests.xml
~~~

软件测试通过不代表真机优化或临床效果。当前没有完整带假腿多轮优化结果。原始采样点不是独立试验，假腿机械配置不是不同患者，机器人任务完成也不等于肢体关节活动范围已验证。实验室硬截止为2027年1月15日，按每周两次、每次三至四小时的单人现场时间收敛。

其他说明：[实验指南](docs/EXPERIMENT.md)、[代码地图](docs/CODE_MAP.md)、[设计取舍](docs/FORCE_DRIVEN_DESIGN_REVIEW.md)、[原六参数条件适配记录](docs/PASSIVE_TRAINING.md)。
