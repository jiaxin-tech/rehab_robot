# 实验指南

## 安装与参数

Windows x64、Python 3.12，在仓库根目录运行：

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win-py312.lock.txt
$env:PYTHONUTF8 = '1'
~~~

只改 [experiment.json](../config/experiment.json)。机器人 SDK 已随仓库保存；leg.ads_dll_path 填实验电脑的真实 TwinCAT DLL 路径，确认 ADS 路由可读。

| 参数组 | 内容 |
| --- | --- |
| robot、leg | 网络、设备与假腿采集；observe 首先需要这些参数 |
| calibration、frame | 编码器零位、角度方向、范围和安装坐标 |
| motion、acquisition | 运动限值、采集频率、时间预算与起始角容差 |
| analysis | 训练时使用的时延、导数窗口、数据间隔和跟踪误差 |
| training | 默认关闭自动训练；候选轨迹和训练超时 |

null 是待填写参数。observe 不要求填写运动参数；run 需要完整物理限值；analysis 可在拿到数据后、训练前填写。安装或零位改变后更新标定。没有签认、发布审批或 SHA/hash 检查。

## 最短操作

**先拿联调数据**：同步读取机器人状态、力与假腿，不发送上电、模式切换或运动命令。厂商连接/断开仍可能影响控制器会话。保持现场装置静止，假腿状态与 leg.guards 一致。

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session observe --session results/observe_001 --duration 10
~~~

observe 的完成只表示观测结束；它没有轨迹发送记录，不能直接作为运动辨识回合训练。故障会中止采集并记录原因。

**运动实验**：检查参数，读取起点并查看预览，核对现场负载、安装和假腿起始角，然后显式启动。

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session check
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --session results/round_001
.\.venv\Scripts\python.exe -m scripts.lab_session run --session results/round_001 --live
.\.venv\Scripts\python.exe -m scripts.lab_session train --session results/round_001
~~~

每轮换新会话名。run 会运动；Ctrl+C、数据过期、超限或故障进入停止与清理。模型建议不自动驱动机器人。training.enabled=true 时采集后自动训练，默认先保留原始数据。

## 结果与诊断

| 文件 | 用途 |
| --- | --- |
| report.md、round_status.json | 一份简报与一份完整状态；含失败原因、质量和训练结果 |
| leg/leg.csv、robot/ | 不同频率的原始数据分别保存 |
| anchor.json、preview/、dispatch.csv | 运动起点、预览和实际发送时刻；observe 不生成 |
| measured.csv、training.log | 对齐后的辨识数据与训练日志 |

时序问题用一个诊断命令，kind 可选 state、wrench、alignment：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.diagnose_robot state --duration 10
~~~

诊断只输出样本 CSV 和摘要 JSON，不在 JSON 中重复样本。静载、重复性和轨迹敏感性验证保留 scripts.run_real_measurement_validation_analysis，输出 validation_summary.json 和 validation_metrics.csv 两份文件。

完成的观测或运动会话可用 python -m utils.assets archive-session --session results/会话名 汇总到现有 data/ 数据包。原始日志保留。本机旧报告集中在 results/history.zip；不要使用历史报告中的旧命令和签认要求。

## 当前进展与下一步

截至2026年10月26日：真实假腿静止采集和机器人状态/力读取已完成基础验证。10月18日四次空载尝试未形成完整轨迹命令记录；仍缺完整真机轨迹、同步动态数据及实测辨识/优化结果。

当前 check 仍有8项物理参数待填，且本机配置的 ADS DLL 不存在。碰撞初始状态、偶发约240～312毫秒力查询阻塞和整链数据时效仍需现场复测。旧静止采集约487～488 Hz，最大接收间隔约39毫秒；不等于稳定无缺口的500 Hz。

下一次依次完成：短时同步观测 → 慢速空载轨迹 → 假腿联动与重复采集 → 时间对齐、辨识与候选比较。单参考自身归一化为1、单候选 DOMAIN_EXHAUSTED 和离线测试通过，均不能作为优化有效的结论。

修改代码后的统一验证：python -m pytest -c pytest-core.ini -q。架构和文件职责见[代码地图](CODE_MAP.md)。
