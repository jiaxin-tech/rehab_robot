# 机器人与假腿：一次真实采集和训练

本入口完成一条慢速参考轨迹的一轮实验。顺序是：只读获取机器人当前位置和预览；现场确认假腿起始姿态、连接及预览；真实 ADS 预采集；运动同时写入假腿 CSV；停止并关闭 CSV；质量检查与时间对齐；实测角度求速度和加速度；读取实测关节力矩；原版残差 GP 均值 Greedy 更新一次；生成报告。训练不改变正在执行的轨迹，后续建议不自动执行。

## 1. 离线检查

在 `D:\rehab_robot` 的 PowerShell 运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_robot_leg_round check --config config/robot_leg_round.json --output round_readiness.json
```

此命令不连接任何设备。`offline_config_valid` 仅说明配置检查结果；不证明路由、控制器或物理运动已经验证。当前配置保留缺失值，预期检查不通过。

要补齐的配置和事实：

| 文件 | 必须补齐的真实信息 |
| --- | --- |
| `config/twincat_leg_capture.json` | 实验电脑上的同位数 TcAdsDll.dll 绝对路径、ADS 路由与过程映像布局复核、超时及新数据预算、轴1/4允许的状态及实测力矩限值。6077符号未核实可保持false，程序保存原始位值，训练使用3C1A。 |
| `config/leg_joint_calibration.json` | 髋的独立验证、膝的角度标定、两个关节的安全角度范围、力矩相对于逆动力学的正负号。当前髋公式仅暂定，膝保留缺失。 |
| `config/leg_round_analysis.json` | 同主机接收时刻与运动发送时刻的延迟测量、允许间隔和读耗时、发送时序预算、求导窗口、起点和跟踪误差。单主机时钟不等于设备硬件同步。 |
| `config/experiment_safety.json`、`config/rehab_frame_config.json` | 机器人现有运动检查要求的限值、身份、工具、负载、工作空间、床面坐标等现场确认内容。 |
| `config/live_observation_acquisition_template.json` | 真实力采集连接、身份、采集稳定性和已验证预算。不能复制测试配置。 |
| `config/robot_leg_round.json` | 安装编号、负责人、操作员、工具及工件名称、机器人类、力采集频率、安全事件有效期、发布过期时间和七类现场证据。历史IP及身份需现场复核。 |

七类证据是控制器安全、采集稳定性、力语义、安装与停止、假腿安装、关节角标定、跨设备时序。配置里的证据路径是待填写位置；程序不创建或伪造证据。软件完成不会自动把 reviewed 改成 true。

现行固定参考的髋范围约28.91至112.03度，膝约18.32至124.79度。软件会将整段计划角度与现场确认的假腿范围比较；如果不适合安装条件，应先修订并重新评审轨迹，不能扩大限值来凑通过。

## 2. 现场只读准备

先人工确认假腿处于参考起始姿态，机器人当前位姿对应这一姿态。所有配置与证据补齐后：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_robot_leg_round prepare --config config/robot_leg_round.json --session results/robot_leg_round/site_round_001
```

此步骤连接机器人读取位姿，不发送运动指令。保存 `anchor.json`、`preview/`、`report.md`，状态为 `AWAITING_ONSITE_CONFIRMATION` 后检查预览和实际安装。

## 3. 现场确认并执行一次

下面命令会实际发运动指令。仅在该次安装和起点确认完成、现场操作员可立即停止运动时执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_robot_leg_round run --session results/robot_leg_round/site_round_001 --confirmation 'I CONFIRM REVIEWED ROBOT LEG ROUND'
```

同一目录只允许尝试一次；失败或配置变更后重新准备新目录。采集未就绪、状态故障、力矩或角度越界、采集过期会阻止或中止运动。断连和停止仍使用原有机器人停止链路。采集子进程必须确认 CSV 已关闭；未完整完成、无法确认关闭、时间缺口或对齐不合格的记录不进入训练。异常数据和错误原因保留。

## 4. 查看结果

先看该轮 `report.md` 和 `round_status.json`。原始假腿数据在 `leg/leg.csv`，机器人记录在 `robot/`，真实发送标记在 `dispatch.csv`。`quality.json` 列出缺失、时间间隔、重复与变化情况；`measured.csv` 保存经标定、对齐后的角度、速度、加速度和力矩；`training_request.json`、`model_update.json`、`training.log` 保存本轮输入、六参数拟合、GP预测和Greedy决策及来源版本。

E3仍为主目标，E2与峰值比用于筛选。第一轮参考与自身比较，三者为1，只用于初始化，不能作为效果证据。当前候选目录只有固定参考，原算法会报告 `DOMAIN_EXHAUSTED`；后续比较需要另外准备并审核真实候选轨迹。空载联调设置为 `empty_load` 时不会训练。无论何种决策，本程序都不自动发送下一条轨迹。

本地离线测试用合成数据替换硬件，训练与分析调用真实实现。测试通过不代表 ADS 路由、安全回调、设备同步和运动已在物理设备验证。
