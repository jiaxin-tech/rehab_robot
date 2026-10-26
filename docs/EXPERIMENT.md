# 假腿实验主线

唯一流程：采集 → 数据质量与时间对齐 → 六参数辨识 → 残差 GP 均值 Greedy 更新。

## 环境与离线检查

在仓库根目录运行，所有项目文件路径都相对仓库解析。

~~~powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest -c pytest-core.ini -q
.\.venv\Scripts\python.exe -m scripts.lab_session check
.\.venv\Scripts\python.exe -m utils.evidence verify
~~~

首次安装：使用 Windows x64 / Python 3.12 创建 .venv，安装 requirements-win-py312.lock.txt。机器人原生 SDK 使用 CPython 3.12；主线动力学需要 MuJoCo，所需 MyoSuite 公共模型资产已包含在证据包中，无需额外下载。现场 ADS DLL 需按实验电脑实际安装提供到配置指定位置 hardware/windows/TcAdsDll.dll。

## 一份配置

仅编辑 config/experiment.json。章节分别保留采集、关节标定、时间对齐、安全、坐标系、参考轨迹、训练和现场入口的独立语义。

旧命令中的 config/*.json 是兼容别名。代码从统一配置生成 .cache 内的运行快照；快照用于严格字段检查和哈希绑定，不是另一份可编辑配置。更改配置或源代码后，旧审批不能继续作为新运行的依据。

安全参数缺失、现场证据未通过或 reviewed=false 时，离线检查应报告阻塞。整理代码不会补造现场证据，也不会开启机器人运动。

## 实验顺序

1. 编辑统一配置，复核机器人身份、工具、负载、限位、床面坐标、ADS 映射、角度标定、同步预算与现场证据。
2. 运行 lab_session check，按输出补齐实际缺项。events、capture、observe 为现场诊断动作，执行时需要显式 --live。
3. 准备一次实验：python -m scripts.run_robot_leg_round prepare --config config/experiment.json --session results/round_001。该步骤读取当前位置和生成预览，不发送轨迹。
4. 现场核对安装、起点和预览后，运行 python -m scripts.run_robot_leg_round run --session results/round_001 --confirmation 'I CONFIRM REVIEWED ROBOT LEG ROUND'。此步骤会实际运动，仍受完整安全门禁约束。
5. 检查该轮 report.md 和 round_status.json。仅完整关闭 CSV、通过质量与对齐检查的数据可用于辨识。

trial_kind=dummy_leg_capture_only 只采集和分析；dummy_leg_reference 才进行算法更新。默认保留现场尚未批准的状态。

第一条参考轨迹只用于初始化。当前候选域若仍只有参考自身，算法应返回 DOMAIN_EXHAUSTED；它不能证明优化收益。模型建议不自动发送给机器人。

## 两份数据文件

- data/experiment_summary.csv：必要实测指标与离线验证结果的统一长表，保留来源逻辑路径和源文件 SHA-256。合成硬件测试与真实实验保留不同来源，不能混作实测证据。
- data/evidence.zip：按 SHA-256 去重保存原始 CSV、历史正式结果、协议、冻结输入、原代码和近期交付验证材料。index.json 在包内记录来源、原大小、哈希与未保留的可再生产物。

25 份微信目录的原始 CSV 已按既有 source_sha256.json 核验，原始字节存入 raw/ 逻辑命名空间。骨架源数据和冻结参考轨迹也保留原始字节。首次使用时，仅将必要参考输入与公共模型资产还原到可丢弃的 .cache/evidence 目录。

~~~powershell
.\.venv\Scripts\python.exe -m utils.evidence list --prefix raw/
.\.venv\Scripts\python.exe -m utils.evidence extract raw/static_zero_2026-10-07.csv --output .cache/static_zero.csv
~~~

每次新采集继续先写独立会话目录，保证日志关闭、故障记录、审批消费记录和停止顺序的语义。不可删除 results/commissioning_release_uses 中的单次许可消费记录。

完成并关闭的一轮可运行 python -m utils.evidence archive-session --session results/round_001，统一追加证据包和汇总表。该命令拒绝未完成会话和同会话内容覆盖，保留原日志与审批消费记录。

## 整理与追溯

合并前 Git 基线为 585e8064d16e4e0e8706861f5adbfd03c551c626。早期二维研究、MyoLeg 大规模比较、消融与渲染任务退出日常代码和测试；公共动力学依赖仍属于主线。

历史源码和科学材料可由证据包或上述 Git 基线查询。测试临时目录、缓存、可重建的逐步仿真轨迹及视频不属于有效实测证据，清理清单保留在包内。历史文档中的批准状态仅属于原配置、原代码和原场景。
