# 实验指南

2026-11-14：正式流程为机械臂位姿/受力驱动TCP候选优化，原六参数只作离线对照。后续操作、前提、结果判据与回传命令统一见[实验操作总手册](NEXT_LAB_OPERATIONS.md)，算法说明见[TCP受力方法](TCP_FORCE_EXPERIMENT.md)。

Windows x64、Python 3.12、仓库自带 xCoreSDK 0.7.0。在仓库根目录 PowerShell 安装：

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win-py312.lock.txt
$env:PYTHONUTF8 = '1'
~~~

默认现场配置为config/experiment.json；正式研究按总手册建立独立配置副本并绑定各轮，原始配置保留。正式主线使用trajectory、force_training、robot、motion、empty_load、frame和acquisition。frame是机器人任务方向，不需要患者髋中心或腿长。旧robot_observation、analysis和原training.candidates留给六参数离线对照，不是正式训练必填项。

validation.capture_dummy_leg=true 时，台架另需 ADS、假腿标定和状态保护；false 时仅使用机器人。所有正式训练都不读取假腿传感器。机器人 TCP/工具负载、受力坐标与补偿、无接触偏置、工作空间和数据时效仍必须核对。

接口顺序：check → observe → 短诊断与空载 → prepare → 核对预览和 blockers → run --live（结束后自动训练）→ repeat --trained → 下一轮 → complete。每轮新目录；配置、安装或起点变化则重新准备。

analyze重算已完成记录的直接受力响应；离线核查不解除失败账本的硬件续跑限制。repeat --trained准备下一自适应候选，repeat --evaluation准备独立评价且不更新GP；不带两者的精确重复会被拒绝。STOP不绕过。完整带载TCP回合会自动训练，training.enabled=false不是关闭它的开关。

原始数据与模型保存于 robot/、dispatch.csv、tcp_force_binding.json、tcp_force_training.json。可选 leg/ 和 passive_response.* 属于台架验证。当前方法不估计人体关节力矩或六参数，不把受力模长称为绑带纯张力，不把机器人任务完成等同于患者关节训练效果。

源码和论文见 [PAPER_ARCHITECTURE.md](PAPER_ARCHITECTURE.md)、[PAPER_CODE_AUDIT.md](PAPER_CODE_AUDIT.md)、[CODE_MAP.md](CODE_MAP.md)。SDK 诊断、停止与历史故障见 [REAL_ROBOT_RUNBOOK.md](REAL_ROBOT_RUNBOOK.md)、[EMPTY_LOAD_LAB_CHECKLIST.md](EMPTY_LOAD_LAB_CHECKLIST.md)；其中旧几何规划说明不适用于当前主线。
