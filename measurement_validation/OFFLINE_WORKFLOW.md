# 假腿 CSV 离线处理

以下命令在 `D:\rehab_robot` 的 PowerShell 中运行，不连接设备。原始 CSV 无需改列名。

## 1. 每次采完生成质量报告

```powershell
$data = '你的本次采集目录'
.\.venv\Scripts\python.exe -m measurement_validation.offline_leg quality $data --output-dir .\outputs\leg_quality
```

目录中只放采集 CSV。如果夹有变量字典等其他 CSV，工具会在汇总中记录导入错误，继续处理其余文件，并以退出码 1 提醒。也可在 `quality` 后明确列出多个 CSV 路径。

打开输出目录的 `summary.md` 看汇总；`summary.json` 和各文件 JSON 保存详细信息：

- 长间隔的前后 CSV 行号、相对时间和间隔长度，接收时钟重复行号。
- UTC / FILETIME 重复和倒退行号，单独于接收时钟统计。
- 空值、必要字段缺失、状态字、故障和错误码。
- 原始位置和力矩的均值、标准差、范围、线性变化率、首尾各 10% 样本均值差。
- 全段及每秒位置稳定性，帮助排除移动中的标定片段。

默认按 500 Hz 检查，超过 4 ms 列为长间隔。位置范围超过 50 count、斜率绝对值超过 20 count/s、有效时长不足窗口的 80% 会标记。首尾力矩差超过 0.2 Nm 会提示变化。它们是可修改的筛查阈值，不是设备精度或运动限值：

```powershell
.\.venv\Scripts\python.exe -m measurement_validation.offline_leg quality $data --output-dir .\outputs\leg_quality --expected-hz 500 --gap-factor 2 --window-s 1 --max-range 50 --max-slope 20 --torque-drift-nm 0.2
```

移动时力矩变化不能直接叫传感器漂移；静止片段仍需确认姿态和负载一致。时间间隔推算的缺失量不是设备精确丢帧数。CSV 同步标签仅作为声明保存。没有关节标定不能自动判为动力学数据。工具不自动去零、不修改原始力矩。

## 2. 保存和验证角度标定

`calibration/hip_candidate.json` 原样保存既有髋关节暂定结果及来源哈希：

```text
hip_flexion_deg = (-121084.4218312675 - axis1_position_raw) / 993.5884722910836
```

该文件用于离线计算，不是运动配置；膝关节没有比例。当前已知点在 `calibration/oct11_points.json`，用这条命令复现：

```powershell
.\.venv\Scripts\python.exe -m measurement_validation.offline_leg calibrate .\measurement_validation\calibration\oct11_points.json --data-dir $data --output .\outputs\leg_calibration.json
```

## 3. 新角度点来了怎么做

在点清单中添加条目，填实际文件名、关节、从第一条记录起算的静止区间和独立测得角度。区间为 `[start_s, end_s)`，单位秒。不要把移动过程与保持过程混在一起。

```json
{"id":"hip_new_check","joint":"hip","file":"新采集文件.csv","start_s":0,"end_s":1,"angle_deg":15,"role":"validation"}
```

这里的 15 只是填写格式示例，应换成实际测得值。`role` 决定该点的用途：

- `fit`：参与重新拟合。至少两个不同已知角度的稳定点才能求比例。
- `validation`：只验证，不参与拟合，优先用它检验新角度点。
- `return`：回程点；实际角度不知道就填 `angle_deg: null`，不会生成角度误差。

输出 `prediction_error_deg` 是既有髋公式的预测角度减实际角度；`refit_error_deg` 是新拟合公式下的差值。拟合点会标成 `in_sample`，不能把它的误差当独立精度。

同一往返配对的两个点填相同 `pair_id`，可另填 `direction` 和 `cycle_id` 作记录。工具输出第二点减第一点的计数差和有比例时的角度差。只有实测角度相同才能称同角度重复差；“接近零”不等于真实零点。两个点不足以验证线性或绝对精度。

不稳定片段保留在报告中但不参与拟合。标定段默认还要求至少 3 行、时长至少 0.8 秒、最大间隔不超过 0.01 秒；清单的 `stability.max_gap_s` 可修改。拟合和验证区间重叠、来源哈希不一致会报错。若膝关节没有两个有效已知角度点，`fits.knee` 始终为 `null`。

手机安装位置、零角度定义、是否重置手机零点目前未知，清单的 `setup` 保留为空；补充事实后再填写，不据此推测机械参数。
