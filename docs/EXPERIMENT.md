# 最短实验操作

## 安装

Windows x64 / Python 3.12，在仓库根目录运行：

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win-py312.lock.txt
$env:PYTHONUTF8 = '1'
~~~

机器人 SDK 随仓库保存。TwinCAT ADS DLL 按本机安装位置填写 leg.ads_dll_path。MuJoCo 公共模型资产随 data/experiment_data.zip 保存，使用时自动还原到 .cache/data，不需单独下载。

## 一份参数文件

编辑 [config/experiment.json](../config/experiment.json)，无需签名、审批、到期时间、源码哈希或证据文件。

| 参数组 | 何时需要调整 |
| --- | --- |
| robot | 机器人和本机 IP、机器人类型、碰撞反馈方式及时效 |
| motion | 速度、加速度、工作空间、力/力矩、起点误差、发送时效、RT 参数 |
| leg | ADS 地址、映像布局、采集频率、超时和假腿力矩保护 |
| calibration | 髋/膝编码器换算、零位、角度范围与力矩方向 |
| frame | 床面 X/Z 轴在机器人 Base 下的方向 |
| acquisition | 力采样频率、运动前后记录长度、假腿数据时效和起始角容差 |
| analysis | 只在分析/训练时使用：时延、数据间隔、导数窗口和跟踪误差 |
| training | 自动训练开关、训练超时、候选轨迹列表 |

null 表示尚无实际参数，不能当成零或自动放行。代码不会为未测量的物理限制猜数值。analysis 未填写不影响只采集；开启训练前再填写。calibration 中已有数值来自既有假腿标定，安装或零位改变后需更新。

## 四个动作

**1. 离线检查**，列出参数缺项，不连接设备：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session check
~~~

**2. 准备与预览**，读取机器人起点，生成 preview/ 中的轨迹 CSV 和图片，不发送轨迹：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session prepare --session results/round_001
~~~

查看预览，核对实际安装、负载、机器人起点与假腿起始角。控制器中的工具、负载及保护设置由现场操作者直接配置，不再要求填写签认表。修改配置后应重新查看预览。

**3. 运行**，显式 --live 才启动运动：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session run --session results/round_001 --live
~~~

每轮使用新会话名。失败时先查 report.md；避免直接重复发送运动。Ctrl+C、数据失效、超限或设备故障会进入停止与清理路径。控制器碰撞保护和急停继续保留。

**4. 离线辨识与训练**，对已经完整结束的采集执行：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.lab_session train --session results/round_001
~~~

training.enabled=false 为默认值；改为 true 会在采集结束后自动运行同样的分析和训练。分析失败不会删除原始记录。模型建议不会自动发送给机器人；单参考候选只用于初始化，不能证明优化收益。

## 看结果和保存

| 本轮文件 | 含义 |
| --- | --- |
| report.md | 简明状态、失败原因和模型建议 |
| round_status.json | 集中保存状态、采集质量、分析指标及模型结果 |
| anchor.json | 实际起点；不需要签认版本 |
| preview/ | 执行前轨迹和图片 |
| leg/leg.csv | 假腿原始采集 |
| robot/ | 四路机器人 CSV 和采集元数据，不同频率的数据分别保存 |
| dispatch.csv | 实际发送时刻，用于与假腿时间对齐 |
| measured.csv | 分析后重建的关节状态与力矩 |
| training.log | 无硬件训练子进程日志 |

长期只保留一张汇总表和一份普通 ZIP。完成后按需归档，不是执行前置步骤：

~~~powershell
.\.venv\Scripts\python.exe -m utils.assets archive-session --session results/round_001
.\.venv\Scripts\python.exe -m utils.assets list --prefix raw/
~~~

压缩包按普通路径存储数据，没有 SHA 清单或证据索引。归档保留本地原始日志；相同会话名不能重复追加。

代码架构与逐文件用途见 [代码地图](CODE_MAP.md)。修改代码后执行离线回归：

~~~powershell
.\.venv\Scripts\python.exe -m pytest -c pytest-core.ini -q
~~~
