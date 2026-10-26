# 假腿实验

目标：获得可对齐的真实机器人与假腿数据，完成六参数辨识，再比较轨迹效果。

只编辑 [config/experiment.json](config/experiment.json)。先做不发送轨迹的同步观测：

~~~powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m scripts.lab_session observe --session results/observe_001 --duration 10
~~~

配置好实验电脑的机器人连接和 ADS DLL 后使用。运动实验依次执行 check → prepare → run --live，采集完成后再 train。

- [实验指南](docs/EXPERIMENT.md)：安装、最短操作、结果和当前进展。
- [代码地图](docs/CODE_MAP.md)：架构和逐文件用途。
- [主线测试](tests/test_mainline.py)：离线采集、设备适配、停止和算法验证。
- [历史汇总](data/experiment_summary.csv)与[数据包](data/experiment_data.zip)：原始记录、参考轨迹和模型。

每轮只看该会话的 report.md；机器可读结果集中在 round_status.json。当前只有一个参考候选，尚不能据此证明优化收益。
