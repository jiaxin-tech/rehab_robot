# 假腿采集与 Greedy 轨迹更新

本仓库统一原机器人仓库和“康复”研究工作区，保留一条实验主线：假腿与机器人采集 → 质量检查及时间对齐 → 六参数辨识 → 残差 GP 均值 Greedy 更新。

| 入口 | 内容 |
| --- | --- |
| [操作与代码地图](docs/CODE_MAP.md) | 最短操作路径、架构图、每个目录和文件的作用 |
| [实验指南](docs/EXPERIMENT.md) | 安装、离线检查、现场流程和追溯 |
| [统一配置](config/experiment.json) | 安全、采集、标定、分析与训练的唯一可编辑配置 |
| [主线测试](tests/test_mainline.py) | 采集、安全门禁、异常停止、分析与训练回归 |
| [结果汇总](data/experiment_summary.csv) | 实验指标、验证结果及来源哈希 |
| [原始证据包](data/evidence.zip) | 去重后的原始数据、冻结轨迹、正式结果和复现材料 |

~~~powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest -c pytest-core.ini -q
.\.venv\Scripts\python.exe -m scripts.lab_session check
.\.venv\Scripts\python.exe -m utils.evidence verify
~~~

统一入口为 scripts.lab_session，轮次入口为 scripts.run_robot_leg_round，算法位于 experiment_algorithm。所有项目依赖随仓库保存，配置和证据的运行快照只生成到 .cache，不进入 Git。

当前现场审批尚未完成。默认保留只采集模式，开启训练和执行仍需实际配置与证据；软件验证不构成真机放行。参考自身只用于初始化，单候选时返回 DOMAIN_EXHAUSTED，不代表优化收益。

旧研究的代码、正式报告和协议集中保留于证据包，并可从 Git 基线 585e8064 追溯。缓存、临时测试产物及可再生仿真跟踪已退出主工作树。
