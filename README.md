# 假腿实验仓库

2026-11-11 更新：当前默认使用独立 TCP 牵引轨迹，假腿被动响应，不要求跟踪髋膝参考角。新流程及完整操作见[独立 TCP 牵引实验](docs/INDEPENDENT_TCP_EXPERIMENT.md)。下文涉及 Z 单轴缩放、膝相位和髋幅值求解的内容仅适用于历史模式。

本仓库用于采集真实 ROKAE 机器人与假腿数据，完成六参数辨识、轨迹执行和结果分析。

## 真机入口

详细的现场操作顺序、SDK 控制链路、实时诊断、空载实验、假腿正式实验、停止流程和证据归档见：

[真机实验操作手册](docs/REAL_ROBOT_RUNBOOK.md)

基本流程是：

1. 修改并检查 config/experiment.json。
2. 用 observe 做不发送运动的同步观测。
3. 用 prepare 生成新 session 和预览。
4. 先完成 1 mm 实时链路诊断。
5. 再做空载实验。
6. 最后安装假腿并运行正式实验。

正式运行示例：

~~~powershell
$env:PYTHONUTF8 = '1'
python -m scripts.lab_session check --config config/experiment.json
python -m scripts.lab_session prepare --config config/experiment.json --session results/round_YYYYMMDD_01
python -m scripts.lab_session run --config config/experiment.json --session results/round_YYYYMMDD_01 --live
~~~

## 其他文档

- [实验指南](docs/EXPERIMENT.md)
- [空载实验清单](docs/EMPTY_LOAD_LAB_CHECKLIST.md)
- [代码地图](docs/CODE_MAP.md)
- [xCoreSDK Windows 运行时说明](hardware/windows/xcoresdk/README.md)

修改代码后，在连接真机前运行：

~~~powershell
python -m pytest -c pytest-core.ini -q
~~~
