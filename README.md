# 假腿实验

个人实验主线：机器人与假腿采集 → 质量与时间对齐 → 六参数辨识 → 残差 GP Greedy。

只编辑 [config/experiment.json](config/experiment.json)。没有 SHA/hash 校验、签认表、放行文件或证据审计流程。

~~~powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m scripts.lab_session check
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --session results/round_001
.\.venv\Scripts\python.exe -m scripts.lab_session run --session results/round_001 --live
~~~

prepare 读取现场起点并生成预览；run --live 会实际运动。运行前填写缺失的物理参数，查看预览，并核对实际安装、负载和起点。限位、力阈值、过期数据停止和异常清理继续生效。

默认只采集，训练可随后单独运行：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session train --session results/round_001
~~~

| 入口 | 内容 |
| --- | --- |
| [实验指南](docs/EXPERIMENT.md) | 安装、参数和最短操作路径 |
| [代码地图](docs/CODE_MAP.md) | 架构、目录和每个文件的用途 |
| [主线测试](tests/test_mainline.py) | 离线回归，不连接真实设备 |
| [结果汇总](data/experiment_summary.csv) | source / metric / value 三列长表 |
| [实验数据包](data/experiment_data.zip) | 原始 CSV、结果表、参考轨迹和公共模型资产 |

本轮先看 results/round_001/report.md。质量、训练结果和失败原因集中在 round_status.json。当前候选域只有参考自身，DOMAIN_EXHAUSTED 是预期停止原因，不能据此声称优化有效。
