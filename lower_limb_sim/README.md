# 二维下肢髋膝屈曲运动学

本模块建立仰卧位人体下肢的二维运动学模型，不是普通二维机械臂。髋关节
固定在 `(0, 0)`，大腿连接髋与膝，小腿连接膝与束缚带等效牵引点。`x`
轴沿床面由髋指向脚，`z` 轴竖直向上，床面为 `z = 0`。

第一阶段回答“腿能到哪里”：包含正逆运动学、关节空间遍历、工作空间图谱、
离散最近点查询、数据保存和可视化。第二阶段回答“一个虚拟受试者在每个
姿态下需要多大准静态牵引力”：只增加重力、线性被动刚度、Jacobian 和
关节力矩到牵引点力的离线映射。第三阶段回答“同一条连续路径走得快慢
不同时，动态力如何变化”：增加双连杆惯性耦合、科氏/离心、线性阻尼和
解析最小 jerk 测试轨迹。三个阶段都不接入真实机器人。
第四阶段反转第三阶段的计算方向：隐藏虚拟受试者真值，只观察运动状态和
束缚带二维力，以传统有界最小二乘估计五个动力学参数。第四阶段仍然是
纯软件虚拟受试者实验。

## 运动学约定

`q_hip` 是髋屈曲角，`q_knee` 是膝屈曲角，内部统一使用 rad，显示和图谱
辅助字段使用 deg。膝屈曲时小腿向大腿折叠，因此小腿绝对方向必须是：

```text
q_hip - q_knee
```

绝不能改成 `q_hip + q_knee`。几何公式为：

```text
x_knee = L1 cos(q_hip)
z_knee = L1 sin(q_hip)
x_pull = x_knee + L2 cos(q_hip - q_knee)
z_pull = z_knee + L2 sin(q_hip - q_knee)
```

正式 `ROM_PROTOCOL_V2` 参数为 `L1 = 0.42 m`、`L2 = 0.30 m`、髋屈曲
`0~120 deg`、膝屈曲 `5~145 deg`，关节网格步长为 `1 deg`。图谱保留
所有关节网格行，并用
`reachable` 标记满足 `x_pull >= 0`、`z_pull >= 0` 的行；绘图和查询只
使用这些有效行。

需要注意：膝关节位置只由 `q_hip` 和 `L1` 决定。固定髋角、只增加
`q_knee` 不会使膝点升高；实际“抬膝”是髋屈曲增大并通常伴随膝屈曲的
耦合动作。

## 生成图谱和图片

从仓库根目录运行：

```bash
python -m lower_limb_sim.workspace_atlas
```

该 main 示例一次生成：

```text
lower_limb_sim/formal_artifacts/rom_protocol_v2/workspace/
├── workspace_atlas.csv
├── workspace_atlas.npy
├── workspace_hip_angle.png
├── workspace_knee_angle.png
└── sample_postures.png
```

CSV 便于人工查看；NPY 是字段一致的 NumPy 结构化数组，可用下面方式快速
加载，不需要 pickle：

```python
import numpy as np

atlas = np.load(
    "lower_limb_sim/data/workspace/workspace_atlas.npy",
    allow_pickle=False,
)
print(atlas.dtype.names)
```

也可仅生成数据，或单独重画图片：

```bash
python -m lower_limb_sim.workspace_atlas --no-plots
python -m lower_limb_sim.visualize
```

## 查询牵引点

```bash
python -m lower_limb_sim.query 0.40 0.20
```

查询返回最近有效图谱点、髋膝角（`q_hip`、`q_knee` 为 rad，另含 deg
字段）和 `distance_error`。默认误差不超过 `0.02 m` 才返回
`reachable=true`；超过阈值仍返回最近图谱点供诊断，但
`reachable=false`。

Python 调用示例：

```python
from lower_limb_sim.query import query_position

result = query_position(0.40, 0.20)
print(result)
```

## 测试

```bash
python -m pytest -q lower_limb_sim
```

测试覆盖随机正逆运动学往返、解析/数值 Jacobian、耦合抬膝/牵引点回收、
床面以下姿态过滤、重力与刚度分项、奇异失效、力矩重构，以及 CSV/NPZ
保存；还覆盖质量矩阵对称/正定、静态一致性、动能非负、解析最小 jerk
端点与速度缩放、动态轨迹几何和完整逆动力学。

## 检查“抬膝并向髋部回收”

可在 `sample_postures.png` 中检查大腿是否随髋屈曲向上转动、小腿是否按
`q_hip - q_knee` 向大腿折叠。定量检查一组动作前后的姿态：

```python
import numpy as np
from lower_limb_sim.config import L1, L2
from lower_limb_sim.kinematics import forward_kinematics

q_hip = np.deg2rad([20.0, 70.0])
q_knee = np.deg2rad([20.0, 120.0])
_, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)

assert z_knee[1] > z_knee[0]   # 膝盖向上抬
assert x_pull[1] < x_pull[0]   # 牵引点沿床面方向向髋部回收
assert np.all(z_pull >= 0.0)   # 牵引点没有穿过床面
```

这里“沿床面向髋部回收”指 `x_pull` 减小，并不要求 `z_pull` 保持不变；
如果实际装置要求牵引点严格沿某一固定高度运动，那属于后续轨迹约束，而
不是当前工作空间几何本身。

## 第二阶段：虚拟受试者准静态力地图

准静态链路为：

```text
(x_pull, z_pull)
        ↓ inverse/atlas
(q_hip, q_knee)
        ↓ gravity + linear passive stiffness
(tau_hip, tau_knee)
        ↓ F = pinv(J.T) @ tau
(Fx_robot_on_leg, Fz_robot_on_leg)
```

这里必须区分四个长度：

- `L1`：髋到膝。
- `L2`：膝到束缚带等效牵引点，只用于运动学和 Jacobian。
- `com_thigh_m`：髋到大腿质心。
- `com_shank_m`：膝到小腿质心。

`L2` 不是小腿质心距离，也不默认等于完整小腿长度。所有内部量采用 SI：
m、kg、rad、N、N·m。

重力力矩来自虚拟腿段的重力势能。由于小腿绝对角为
`q_hip - q_knee`，膝重力项必须带负号：

```text
tau_gravity_knee =
    -mass_shank * com_shank * g * cos(q_hip - q_knee)
```

被动力矩采用第一版线性弹簧：

```text
tau_stiffness_hip  = k_hip  * (q_hip  - q0_hip)
tau_stiffness_knee = k_knee * (q_knee - q0_knee)
```

总需求力矩始终保留重力、刚度和合计三个层次，便于审计和论文分析。
末端力使用 `pinv(J.T)`，不使用 `inv(J.T)`。Jacobian determinant 过小、
condition number 超限、输入或输出非有限、或者力幅值超过配置上限时，
`force_mapping_valid=False`，力字段写为 NaN，并记录 `invalid_reason`。

### 虚拟受试者

内置四组软件验证参数：

| subject_id | 变化 |
|---|---|
| `baseline` | 普通虚拟腿重和刚度 |
| `hip_stiff` | 仅髋刚度由 15 提高到 30 N·m/rad |
| `knee_stiff` | 仅膝刚度由 12 提高到 30 N·m/rad |
| `heavy_leg` | 大腿、小腿质量分别提高 30% |

这些数值只用于验证代码与算法差异，不是医学标准、患者参数或临床阈值。
同一个牵引点位置对不同虚拟受试者可能需要不同的力。

### 生成力地图

先确保第一阶段图谱存在：

```bash
python -m lower_limb_sim.workspace_atlas
```

生成单名受试者：

```bash
python -m lower_limb_sim.build_force_map baseline
python -m lower_limb_sim.build_force_map hip_stiff
python -m lower_limb_sim.build_force_map knee_stiff
python -m lower_limb_sim.build_force_map heavy_leg
```

一次生成全部力地图和共同姿态比较：

```bash
python -m lower_limb_sim.build_force_map --all
```

数据保存在：

```text
lower_limb_sim/data/force_maps/
├── force_map_baseline.csv
├── force_map_baseline.npz
├── force_map_hip_stiff.csv
├── force_map_hip_stiff.npz
├── force_map_knee_stiff.csv
├── force_map_knee_stiff.npz
├── force_map_heavy_leg.csv
├── force_map_heavy_leg.npz
├── virtual_subject_comparison.csv
├── virtual_subject_comparison.png
├── baseline/
│   ├── force_magnitude_map.png
│   ├── fx_map.png
│   ├── fz_map.png
│   ├── hip_torque_map.png
│   ├── knee_torque_map.png
│   └── force_vector_field.png
└── ...其他受试者同样的六张图
```

绘图只使用 `force_mapping_valid=True` 的数据。颜色显示采用百分位裁剪以
避免少量极端值压缩色阶，但 CSV/NPZ 原始数据不会被裁剪或修改。

### 力方向约定

保存的 `fx_robot_on_leg_n`、`fz_robot_on_leg_n` 表示机器人通过束缚带
施加在腿上的力。未来真实机器人传感或控制器估计可能表示腿施加在机器人
上的反作用力；理想情况下：

```text
force_leg_on_robot = -force_robot_on_leg
```

在真实数据接入前必须再次核对传感器、控制器和坐标系的实际符号，不能仅凭
本仿真命名直接假设 ROKAE 的力方向。

### 当前模型的适用边界

第二阶段准静态力地图本身只适用于非常缓慢、可以忽略动态效应的运动，
只包含：

- 重力；
- 独立的线性髋、膝被动刚度。

当前不包含：

- 速度、加速度、阻尼、惯性和科氏项；
- 主动肌肉收缩、非线性/耦合刚度和滞后；
- 束缚带滑移、触觉传感器和真实 ROKAE 接口；
- 动态轨迹、参数辨识、PINN、MPC 或轨迹优化。

位置不能唯一决定真实动态力。同一个位置经过的速度、加速度、肌肉激活和
束缚带状态不同，真实力也会不同。

## 第三阶段：连续轨迹完整动态仿真

第三阶段链路为：

```text
software_test_trajectory(t)
        ↓ analytic minimum jerk
q(t), q_dot(t), q_ddot(t)
        ↓ full inverse dynamics
tau_inertia + tau_coriolis + tau_gravity
            + tau_damping + tau_stiffness
        ↓ F = pinv(J.T) @ tau_total
Fx_robot_on_leg, Fz_robot_on_leg
```

仍然严格使用：

```text
theta_thigh       = q_hip
theta_shank       = q_hip - q_knee
theta_shank_dot   = dq_hip - dq_knee
theta_shank_ddot  = ddq_hip - ddq_knee
```

质量矩阵由该定义下的质心动能和独立转动惯量展开得到，不套用
`q1+q2` 机器人公式。baseline 大腿、小腿示例转动惯量分别为 `0.12`
和 `0.06 kg·m²`；它们是独立虚拟参数，不使用 `L2` 推算。`heavy_leg`
按本项目统一 `mass_scale=1.3` 同时缩放质量和惯量。默认髋、膝线性阻尼
分别为 `2.0` 和 `1.5 N·m·s/rad`。

### 软件测试轨迹

导师正式参考轨迹尚未获得，所以所有输出都明确命名为：

```text
software_test_trajectory
```

它不是临床参考轨迹或康复标准轨迹。当前路径为：

```text
(q_hip, q_knee): (20°, 20°)
    → minimum-jerk flexion → (120°, 120°)
    → minimum-jerk extension → (20°, 20°)
```

三条轨迹的几何路径完全相同，只改变单程时间：

| speed_profile | 单程 | 往返总时长 |
|---|---:|---:|
| `slow` | 12 s | 24 s |
| `nominal` | 6 s | 12 s |
| `fast` | 3 s | 6 s |

默认采样频率为 100 Hz。角度、角速度和角加速度由五次最小 jerk 多项式
解析计算，不使用简单差分作为主要结果。

### 运行动态仿真

生成一条轨迹：

```bash
python -m lower_limb_sim.simulate_dynamic_trajectory baseline slow
python -m lower_limb_sim.simulate_dynamic_trajectory baseline nominal
python -m lower_limb_sim.simulate_dynamic_trajectory baseline fast
```

一次生成4名虚拟受试者、3种速度共12条轨迹：

```bash
python -m lower_limb_sim.simulate_dynamic_trajectory --all
python -m lower_limb_sim.compare_speed_profiles --all
```

每条轨迹目录包含：

```text
data/dynamic_trajectories/<subject>/<speed>/
├── trajectory.csv
├── trajectory.npz
├── metadata.json
├── joint_angles_vs_time.png
├── joint_velocities_vs_time.png
├── joint_accelerations_vs_time.png
├── pull_point_path.png
├── leg_animation.gif
├── joint_torque_components.png
├── endpoint_force_vs_time.png
└── force_vector_along_path.png
```

`metadata.json` 保存全部虚拟受试者参数、L1/L2、时间和角度定义、模型包含/
排除项、力方向、生成时间、软件版本以及可安全读取时的 Git commit。
`torque_reconstruction_error_nm` 审计 `J.T @ F` 对总力矩的重构误差。

速度比较输出在每名受试者目录的
`speed_profile_comparison.csv/.png`。程序自动验证路径范围相同、fast
速度和加速度高于 nominal/slow、惯性与阻尼项随速度增大，并验证同一
姿态下的重力和刚度项不随速度改变。

### 完整动态模型边界

当前模型包含：

- 双连杆惯性耦合；
- 科氏和离心项；
- 重力；
- 独立线性关节阻尼；
- 独立线性被动刚度。

当前模型不包含：

- 真实患者主动肌肉力；
- 非线性关节末端阻力、痉挛、摩擦或迟滞；
- 束缚带弹性和滑移；
- 分布式接触力和触觉压力；
- 机器人本体动力学；
- 真实 wrench 的延迟、噪声、参考点和坐标变换误差；
- 第三阶段轨迹生成器本身不做参数辨识；第四阶段在独立模块中实现传统
  参数辨识，但仍不包含 PINN、MPC 或轨迹优化。

力方向仍为：

```text
force_leg_on_robot = -force_robot_on_leg
```

真实 ROKAE wrench 的符号必须根据 SDK 定义、参考点和坐标变换另行验证。
当前输出不能直接作为真实 ROKAE 控制输入或安全阈值。大力来源分析见
[dynamic_force_audit.md](dynamic_force_audit.md)。

## 第四阶段：虚拟受试者动力学参数辨识

前三阶段是“已知患者参数，预测运动所需的力”；第四阶段是“只观察运动
和力，反过来估计患者参数”：

```text
q, q_dot, q_ddot, Fx_observed, Fz_observed
        ↓ tau_measured = J(q).T @ F_observed
measured hip/knee generalized torque
        ↓ bounded scipy least_squares
mass_scale, K_hip, K_knee, B_hip, B_knee
        ↓ estimated-parameter inverse dynamics
train / validation / test torque prediction
```

辨识器继续严格使用：

```text
theta_shank = q_hip - q_knee
```

并复用现有 `jacobian.py` 和 `full_dynamics.py`，不复制普通
`q_hip + q_knee` 机械臂公式。

### 辨识和已知参数

当前只估计五个参数：

| 参数 | 含义 | 默认边界 |
|---|---|---:|
| `mass_scale` | 同时缩放 baseline 大小腿质量和惯量 | 0.6–1.6 |
| `K_hip` | 髋线性被动刚度 | 0–60 N·m/rad |
| `K_knee` | 膝线性被动刚度 | 0–60 N·m/rad |
| `B_hip` | 髋线性阻尼 | 0–10 N·m·s/rad |
| `B_knee` | 膝线性阻尼 | 0–10 N·m·s/rad |

已知量是 `L1`、`L2`、质心距离、中性角、重力和 baseline 人体质量/惯量
比例模板。当前不辨识完整 `2×2 M/B/K` 矩阵、两个独立质量、两个独立
惯量、质心距离、`q0`、非线性或髋膝耦合刚度、主动肌肉输出、束缚带滑移
和触觉特征。

优化采用 `scipy.optimize.least_squares` 的有界解，默认
`loss="soft_l1"`，并分别缩放髋/膝残差和五个参数量纲。不会产生负质量、
负刚度或负阻尼。可切换 `--loss linear` 作为普通线性损失对照。

### 辨识激励和数据划分

每名虚拟受试者使用 3 个 family × slow/nominal/fast，共 9 条解析最小
jerk 往返轨迹：

| family | 起点 → 最大屈曲 | 目的 |
|---|---|---|
| `coupled` | 20°/20° → 70°/120° | 总体范围覆盖 |
| `hip_dominant` | 20°/30° → 120°/80° | 增强髋参数信息 |
| `knee_dominant` | 20°/20° → 50°/120° | 增强膝参数信息 |

题设示例的 hip 终点 120°/60° 会使当前 `L1/L2` 下的牵引点越过髋后方，
knee 终点 40°/120° 会落到床面以下。因此使用上表中仍保持主导关节特征、
同时满足 `x_pull>=0`、`z_pull>=0` 的端点。`hip_dominant` 明确达到用户
要求的 120° 最大髋屈曲角。全部轨迹标记为
`identification_excitation_trajectory`，不是临床参考轨迹。

轨迹级划分固定为：

- train：coupled slow/fast、hip_dominant nominal、knee_dominant nominal；
- validation：hip_dominant slow、knee_dominant fast；
- test：coupled nominal、hip_dominant fast、knee_dominant slow。

测试轨迹不会进入参数拟合。保存的 `tau_measured_*` 由观测力重新执行
`J.T @ F` 得到；估计器内部还会再次重算，绝不读取第三阶段
`tau_total_*` 作为测量值。泄漏检查见
[parameter_leakage_audit.md](parameter_leakage_audit.md)。

### 运行

从仓库根目录运行单个实验：

```bash
python -m lower_limb_sim.run_identification baseline clean
python -m lower_limb_sim.run_identification knee_stiff force_noise_medium
```

批量运行：

```bash
python -m lower_limb_sim.run_identification --all-clean
python -m lower_limb_sim.run_identification --all-scenarios
```

每个实验保存在
`lower_limb_sim/data/identification/<subject>/<scenario>/`，包括：

```text
training_data.csv
validation_data.csv
test_data.csv
estimated_parameters.json
metrics.json
metadata.json
predicted_vs_measured.csv
parameter_estimates.csv
dataset_metrics.csv
identification_summary.json
identifiability_summary.csv
parameter_correlation_matrix.csv
sensitivity_singular_values.csv
force_amplitude_sensitivity.csv
geometry_filtered_estimated_parameters.json
true_vs_estimated_parameters.png
predicted_vs_measured_hip_torque.png
predicted_vs_measured_knee_torque.png
torque_residuals_vs_time.png
parameter_relative_errors.png
parameter_correlation_heatmap.png
sensitivity_singular_values.png
clean_vs_noise_comparison.png
```

辨识根目录另汇总 `parameter_estimates.csv`、`dataset_metrics.csv` 和
`identification_summary.json`。

### 可辨识性不是 optimizer success

程序用统一量纲缩放的数值灵敏度矩阵比较：

1. 一条 coupled nominal；
2. coupled 的三种速度；
3. 三个 family 的全部九条轨迹。

同时报告奇异值、数值秩、condition number、参数相关矩阵和高相关参数对。
还单独去除 Jacobian condition 最高的 5% 几何区域做敏感性分析，并按力
幅值四分位报告灵敏度；主辨识仍使用全部有效样本，不因约 150–238 N 的
虚拟峰值力而缩放或删除高力样本。

### 噪声和时序场景

`clean` 通过后，可分别运行 0.5 N/2.0 N 高斯力噪声、固定力偏置、
16/32 ms 因果 wrench 延迟、5% 随机丢帧、约 200–250 ms stale freeze
和组合现实场景。固定随机种子保证复现。无历史延迟、dropout 和 stale
样本显式标记无效；不使用未来样本补缺，也不静默插值长冻结。

`advanced_angle_noise` 是独立高级场景：只给 q 加噪，使用离线
Savitzky–Golay 滤波并从带噪 q 数值微分得到 dq/ddq，禁止继续使用仿真
真值 dq/ddq。它不与 clean 验收混合。

只有四名虚拟受试者在 clean 条件下先满足质量/刚度误差小于 2%、阻尼误差
小于 5%，并且噪声、延迟和丢帧场景仍能稳定恢复参数后，才有理由进入未来
的轨迹个性化优化研究。即便通过，本结果仍是软件虚拟患者辨识，不是真实
患者参数估计。

## 阶段 4.5A：wrench 与运动状态时间对齐

参数辨识不能假设 wrench 和状态数组的同一行代表同一物理时刻。阶段 4.5A
为两条数据流分别保存：

```text
state_timestamp_s
wrench_timestamp_s
wrench_age_s
state_wrench_skew_s
```

固定正延迟定义为：

```text
F_observed(t) = F_clean(t - delay)
```

也就是正延迟表示 wrench 内容落后于状态。虚拟固定延迟记录中的
`state_timestamp_s` 和 `wrench_timestamp_s` 是同一个轨迹局部单调时钟
上的原始采集时间；`wrench_age_s` 记录虚拟信号内容年龄，仅用于保存和最终
审计。自动估计器会删除 age、旧 delay、场景名、subject id、真值字段和
DataFrame attrs。候选补偿使用：

```text
wrench_effective_timestamp_s =
    wrench_timestamp_s - candidate_delay_s
```

再把 Fx/Fz 对齐到状态时间轴，并重新通过 `J(q).T @ F` 计算测量关节力矩。
不能按数组行号直接配对，否则延迟会把错误时刻的力对应到当前的 `q`、
`dq` 和 `ddq`。阻尼力矩直接依赖速度，且屈曲/伸展时速度换符号，因此在
当前虚拟实验中，时间错位对 `B_hip`、`B_knee` 的影响明显大于普通零均值
力噪声。

### 自动延迟选择与数据隔离

固定网格范围为 `-50~+50 ms`，步长 `1 ms`，共 101 个候选。每个候选都：

1. 仅用 train 重新拟合五个动力学参数；
2. 仅用 validation combined torque RMSE 评分；
3. 用 validation 最优点选择延迟；
4. 在选择完成后才使用 test 做最终评价。

自动接口不接受 test DataFrame 或 true delay。所有候选使用共同有效样本
支持和固定力矩尺度，避免某个候选仅靠丢弃更多样本获得更低 RMSE。RMSE
完全相同时，稳定地优先选择绝对值较小的延迟，再选择数值较小的有符号
延迟。若选中 `-50 ms` 或 `+50 ms`，输出
`search_boundary_hit=true` 和范围外警告；边界命中不能解释为精确估计。
1 ms 网格也意味着非整数延迟只能落在邻近候选，不能声称亚毫秒分辨率。

### 离线和因果模式

两种模式明确分开：

- `offline_only` 使用目标时刻前后的有效 wrench 做线性插值，可能读取未来
  样本，并记录 `alignment_used_future`。它只适合离线分析和参数辨识。
- `causal_history` 只使用状态时刻已经到达的当前/历史 wrench，以最近历史
  值保持；所用 wrench 时间戳不得晚于状态时刻。它用于模拟在线信息约束，
  但不能像离线双向插值一样提前得到尚未到达的数据，精度可能更低。

两种模式都不外推，不跨轨迹 family/speed 边界。最大插值或保持间隔为
`20 ms`。dropout、连续缺失或 stale freeze 造成更长缺口时，样本保持
`sample_valid=false`；stale 的有限旧值不能作为正常新样本，也不会被
长距离静默插值。

### 运行与输出

从仓库根目录运行 baseline 的六个固定延迟：

```bash
python -m lower_limb_sim.run_delay_compensation_experiment baseline
```

也可只运行一个延迟：

```bash
python -m lower_limb_sim.run_delay_compensation_experiment baseline \
    --delay-ms 16
```

输出位于 `lower_limb_sim/data/delay_compensation/`：

```text
baseline/
├── delay_000ms/
├── delay_008ms/
├── delay_016ms/
├── delay_024ms/
├── delay_032ms/
└── delay_040ms/
```

每个实验目录保存：

```text
delay_search_curve.csv
delay_compensation_summary.csv
compensated_dataset.csv
metadata.json
delay_search_curve.png
before_after_parameter_error.png
```

根目录保存 `all_delay_summary.csv`、`delay_estimation_accuracy.png`、
`test_rmse_before_after.png` 和 `damping_error_before_after.png`。
`delay_compensation_summary.csv` 同时保留 uncorrected、known-delay 和
automatic 三种结果；`metadata.json` 保存 split 轨迹列表、随机种子、
搜索范围、20 ms 门限、Git commit/软件版本和角度定义
`theta_shank = q_hip - q_knee`。详细假设、边界和泄漏检查见
[delay_compensation_audit.md](delay_compensation_audit.md)。

这一阶段仍然只是在虚拟数据上验证固定延迟的离线补偿。离线延迟估计不等价
于真实在线控制；当前模块不接入真实 ROKAE 控制环，不修改采集、安全、
硬件代码或安全阈值，也没有加入模型失配、轨迹优化、PINN、MPC 或触觉。

## 阶段 4.5B：变化延迟的因果缓存匹配

阶段 4.5A 假设整条记录具有一个固定延迟；4.5B 进一步模拟分段变化、缓慢
漂移、抖动、双峰延迟、50–105 ms 长尾、5% dropout、100–250 ms
stale freeze 和组合场景。随机种子固定，因此这些场景可复现，但它们仍是
软件虚拟数据，不代表真实传感器或患者。

### 时间戳和因果状态缓存

4.5B 明确区分：

```text
state_timestamp_s
wrench_arrival_timestamp_s
wrench_sample_timestamp_s
```

正延迟等于 `wrench_arrival_timestamp_s - wrench_sample_timestamp_s`。
可靠 sample timestamp 存在时优先使用；缺失时才用
`arrival - estimated_delay` 得到目标状态时刻。`true_delay_s` 只在生成
和最终评价侧可见，不进入自动跟踪。

`StateHistoryBuffer` 默认保存最近 `2 s` 严格递增的 q、dq、ddq，并裁剪
过期样本。查询可选 `nearest` 或 `linear_interpolation`，但只能访问已经
缓存的历史，不允许未来外推。最大状态插值间隔为 `20 ms`，最终状态匹配
误差门限为 `5 ms`。过期、未来、无 bracket 和间隔超限分别保留明确原因。

调用端还必须保证 buffer 中没有晚于当前 wrench 到达时刻的状态。
`CausalSampleMatcher` 会再次拒绝未来到达、未来目标、未来匹配状态以及
包含未来状态的缓存。模型角度没有改变：

```text
theta_shank = q_hip - q_knee
```

### 四种比较方法

4.5B 分开比较：

1. `row_index_alignment`：同行直接配对，只是忽略延迟的负面对照；
2. `global_fixed_delay`：整段数据使用一个固定延迟，不能跟踪漂移和抖动；
3. `causal_history_latest`：把当前状态与当前已经到达的最新 wrench 配对，
   不读取未来，但也没有消除“旧力配当前运动”的时间错位；
4. `causal_buffered_matching`：优先使用可靠 sample timestamp，否则使用
   最新延迟估计，再从有限状态缓存匹配 q、dq、ddq。

前两项是比较基线；后两项模拟因果信息约束。即使
`causal_buffered_matching` 在回放中表现最好，它也仍是软件因果回放，
不是经过真实实时线程和机器人硬件验证的控制接口。

### 滑动窗口跟踪和低激励保持

窗口跟踪器默认使用 `2 s` 历史、每 `0.5 s` 更新一次，在
`-50~80 ms` 范围按 `1 ms` 步长搜索。结果只能声称 1 ms 候选网格分辨率，
不能声称亚毫秒精度。输出同时记录置信度、有效样本数、激励分数、边界
命中、平滑前后延迟和单次变化限制。

近静止或低激励窗口不强行估计：

```text
delay_update_valid = false
delay_update_reason = insufficient_excitation
delay_value_held = true
```

有效样本不足、评分不可用、低置信度或搜索边界低置信度时也保持上一次延迟
值。保持表示“没有足够证据更新”，不表示旧值已被重新验证。tracker 只接受
`train` 或明确的 `online` 数据，validation/test 不进入窗口更新。

### dropout、冻结和在线边界

dropout、非有限力、stale/frozen wrench、重复 sample timestamp、
超过 `100 ms` 的 wrench age、超过 `20 ms` 的状态缺口以及超过 `5 ms`
的匹配误差都 fail closed。冻结期间的有限旧力不能当作新鲜样本，长缺口
不能为了提高有效率而用未来数据静默填补。

长尾和抖动可能让一个尚未见过的 sample timestamp 以乱序方式迟到；这种
唯一样本仍可在 age 和缓存门限内匹配，不能仅因时间戳小于上一到达样本就
误判为 freeze。当前主实验提供可靠的虚拟 sample timestamp，因此缓存法
的极低 RMSE 主要验证“可靠 sample timestamp + 因果历史状态查询”。仅有
arrival timestamp 时的 `arrival - estimated_delay` fallback 已单独测试，
但不能把主实验结果解释为真实系统中的盲时延辨识精度。

`offline_only` 双向插值仍只允许用于离线参照和参数辨识。4.5B 的因果
缓存与tracker虽然按时间顺序回放且禁止未来数据，但尚未验证设备时钟同步、
实时调度、deadline、SDK查询、锁竞争、坐标系和真实wrench语义，所以不能
直接接入真实在线控制，也不能设定安全阈值。

详细字段、四种方法边界、拒绝原因和泄漏检查见
[variable_delay_alignment_audit.md](variable_delay_alignment_audit.md)。
阶段 4.5B 没有修改任何真实机器人控制、采集、安全或硬件代码，结果不能
解释为真实患者参数估计、临床有效性、舒适性或机器人安全验证。

运行示例：

```bash
python -m lower_limb_sim.run_variable_delay_experiment baseline piecewise_delay
python -m lower_limb_sim.run_variable_delay_experiment baseline combined_realistic
python -m lower_limb_sim.run_variable_delay_experiment --all-baseline
python -m lower_limb_sim.run_variable_delay_experiment --all
```

每个实验目录保存 `raw_variable_delay_dataset.csv`、`matched_dataset.csv`、
`delay_tracking_history.csv`、`delay_search_windows.csv`、
`method_comparison.csv`、`parameter_estimates.csv`、`metrics.json` 和
`metadata.json`，并生成七张延迟、匹配、力矩、参数和有效样本图。

## 阶段 4.5C：模型失配与未见轨迹预测

阶段 4 的 clean 实验证明：当数据生成模型与辨识模型完全相同时，现有
五参数辨识器可以恢复生成参数。4.5A 和 4.5B 又证明：即使动力学公式正确，
wrench 与运动状态错位也会显著损害辨识，特别是阻尼参数。

4.5C 改变研究问题：虚拟“真实腿”可以具有三次刚度、由势能推导的髋膝
耦合、速度平方阻尼和小幅平滑结构残余，但估计器仍然只能拟合：

```text
[mass_scale, K_hip, K_knee, B_hip, B_knee]
```

因此这里的 K 和 B 是训练运动区域内的**等效线性参数**，不等于直接测得
的组织刚度或阻尼，也不能因为它们与复杂生成器中的线性系数不同就自动判定
辨识失败。主要评价依据改为：简化模型能否预测没有参与拟合的新路径上的
关节力矩和束缚带端点力。

### 复杂生成场景与模型隔离

内置九个确定性软件场景：`matched_linear`、三次刚度 mild/strong、髋膝
耦合 mild/strong、`nonlinear_damping_mild`、`structured_residual`、
`combined_mild` 和 `combined_strong`。复杂参数只用于生成和事后审计。
调用估计器前，数据会被硬白名单投影为 q、dq、ddq、Fx、Fz 和有效性字段；
以下内容不会进入拟合：

- `tau_complex_true` 及各复杂力矩分项；
- 生成参数和场景名称；
- subject ID、split 标签和 test 轨迹；
- validation、interpolation、boundary 或 outside-domain 样本。

观测力矩仍从 `tau_measured = J.T @ F` 重新构造。辨识器本身没有增加三次
刚度、耦合或残余项，也没有为了改善结果改变原五参数边界。

### 未见轨迹和适用域

拟合继续使用既有 identification excitation 的固定 train 轨迹；validation
只评价。另增加六条明确标记为 `software_validation_trajectory`、且不是
临床参考的轨迹：

- `phase_shift_small`、`amplitude_mix`、`intermediate_speed`、
  `asymmetric_flexion_extension` 属于 `interpolation_test`；
- `boundary_near` 属于 `boundary_test`；
- `outside_domain` 属于 `outside_domain_test`，只用于外推风险审计。

所有轨迹仍严格使用：

```text
theta_shank = q_hip - q_knee
```

髋角总上限仍为 120°。`outside_domain` 只超出训练状态覆盖，不超出配置的人体
总关节和床面以上工作空间，但它不能用于支持“模型有效”的主结论。模型只应
在已经验证的角度、速度、加速度和轨迹邻域内使用。

### 指标、残差和通用模型对照

每条轨迹和每个 split 都比较：

1. `generic_baseline_model`：固定 baseline 五参数；
2. `identified_equivalent_model`：只用 train 数据估计的五参数。

保存关节力矩 RMSE/MAE/峰值误差、相关系数、VAF、端点力误差和残差均值。
NRMSE 固定定义为：

```text
NRMSE = RMSE / (max(true torque) - min(true torque) + epsilon) * 100%
```

真实力矩范围过小时不输出误导性的巨大百分比，而标记
`nrmse_unreliable_small_range`。残差还会与 q、dq、ddq 做相关性诊断；
`possible_nonlinear_stiffness_mismatch`、
`possible_nonlinear_damping_mismatch` 和
`possible_joint_coupling_mismatch` 只是筛查线索，不能证明真实生理机制。

运行示例：

```bash
python -m lower_limb_sim.run_model_mismatch_experiment baseline matched_linear
python -m lower_limb_sim.run_model_mismatch_experiment baseline nonlinear_stiffness_mild
python -m lower_limb_sim.run_model_mismatch_experiment baseline combined_mild
python -m lower_limb_sim.run_model_mismatch_experiment --all-baseline
python -m lower_limb_sim.run_model_mismatch_experiment --all
```

每个实验目录保存五个固定 split 数据集、`estimated_parameters.json`、
`generator_parameters.json`、`prediction_metrics.csv`、
`generic_vs_identified_comparison.csv`、`residual_feature_correlations.csv`、
`predicted_vs_true_torque.csv`、`metadata.json`，并生成八张规定结果图及四张
残差-特征图。详细泄漏边界、等效参数解释和外推限制见
[model_mismatch_audit.md](model_mismatch_audit.md)。

本阶段默认使用无延迟可靠对齐，不把变化时延和模型失配混为同一主变量；
它没有接入真实 ROKAE、患者或临床轨迹，也没有修改控制、采集、安全、硬件
或 SDK 代码。只有复杂虚拟数据下的未见轨迹预测验证，不能替代真实患者
参数估计或临床验证。

## 阶段 4.5D：几何标定与运动学观测误差

阶段 4.5C 检验的是“人体动力学结构不完全正确时，简化五参数模型还能否
预测未见轨迹”。阶段 4.5D 检验另一条独立误差链：即使动力学模型本身可以
工作，如果髋中心、腿长、束缚带作用点或角度测量错误，参数辨识和未见轨迹
预测是否仍然可靠。

真实机器人 TCP 位置不等于人体真实髋膝角度。当前虚拟观测链明确展开为：

```text
true joint motion
    -> true pull-point position
    -> geometry/sensor error
    -> reconstructed joint angles
    -> estimated velocity and acceleration
    -> reconstructed joint torque
    -> five-parameter identification
    -> unseen-trajectory prediction
```

角度约定没有改变：

```text
theta_shank = q_hip - q_knee
```

髋屈曲总上限仍为 120°，禁止套用 `q_hip + q_knee` 的普通机械臂公式。

### 真实几何与假设几何严格分开

仿真生成侧保存 `true_geometry`：真实 `L1/L2`、髋中心和髋膝中性角
`q0`；观测与辨识侧只能读取测得的牵引点、测得的 `Fx/Fz` 和
`assumed_geometry`。估计器不能用真实关节状态、真实 Jacobian、真实力矩
或真实几何纠正结果。

这里的 `L2` 是膝到束缚带等效牵引点的距离。每次重新穿戴束缚带后，作用
点可能改变，因此真实实验必须重复标定，不能把一次安装的 `L2` 永久沿用。
本阶段只模拟一次安装后保持不变的静态 `L2` 偏差，不加入随时间变化的
束缚带滑移；动态滑移需要在后续独立研究。

### 三种运动学观测模式

本阶段比较：

1. `oracle_true_joint_state`：直接使用仿真真实 `q/dq/ddq`，只表示理论
   上界，不是现实可实现的传感方案；
2. `tcp_inverse_kinematics`：只用测量 TCP、假设髋中心和假设 `L1/L2`
   反解关节角，再从重建角度估计 `dq/ddq`；
3. `independent_joint_measurement`：模拟侧视相机或两个 IMU 独立测量髋膝
   角，再经过滤波和微分。

独立相机或 IMU 可以绕过一部分 TCP 逆运动学误差，并帮助检查关节角是否
可信；但它不能自动修复束缚带作用点、错误 Jacobian、力坐标系或 `q0`。
`q0` 误差会改变 `K * (q - q0)` 的零点并偏置刚度 K。主估计器仍只有
`mass_scale`、`K_hip`、`K_knee`、`B_hip`、`B_knee` 五个参数；K–`q0`
相关性只作附加审计，不把主问题扩成七参数拟合。

### 导数、未来样本与数据隔离

带噪角度不能直接简单差分：一次微分会放大高频噪声，二次微分对加速度的
影响更大，进而污染阻尼和质量缩放辨识。本阶段区分：

- `central_difference_offline`、`savitzky_golay_offline` 会使用当前时刻
  之后的角度，输出必须标记 `uses_future_samples=true`，仅适合离线分析；
- `causal_backward_difference`、`causal_filter_and_difference` 只允许
  当前及历史数据，但可能具有更大噪声或滤波延迟；
- 无效 IK、分段边界和长缺口不得被跨越微分，质量标志必须保留。

滤波窗口和微分参数只能预先固定，或使用 train/validation 选择。test、
interpolation、boundary 和 outside-domain 数据不参与参数拟合、滤波选择、
几何选择或模型选择，只用于最终评价。

### 力矩、适用域和结论边界

观测力矩必须由估计状态和假设几何重建：

```text
tau_measured_est = J_assumed(q_est, L1_assumed, L2_assumed).T @ F_observed
```

`J_true.T @ F_true` 只用于事后计算力矩观测误差，不能进入辨识器。运行时
适用域也只能依据重建的 `q_est/dq_est/ddq_est`；真实状态仅用于事后计算
域内误接收和误拒绝率。模型仍只应在已经由 train/validation 和未见插值
轨迹覆盖、并通过误差检查的运动区域内使用。`outside_domain` 仅用于暴露
外推风险，不能支持“模型有效”的主结论。

运行示例：

```bash
python -m lower_limb_sim.run_geometry_error_experiment baseline matched_geometry
python -m lower_limb_sim.run_geometry_error_experiment baseline L2_error_2cm
python -m lower_limb_sim.run_geometry_error_experiment baseline combined_geometry_mild
python -m lower_limb_sim.run_geometry_error_experiment --all-baseline
python -m lower_limb_sim.run_geometry_error_experiment --all-sensitivity --no-plots
python -m lower_limb_sim.run_geometry_error_experiment --noise-seed-study --noise-seeds 20
python -m lower_limb_sim.run_geometry_error_experiment --all-core-subjects
```

结果保存到 `lower_limb_sim/data/geometry_error/`。每个实验保留真实、观测和
重建轨迹，五个固定数据 split，真实与假设几何，运动学/辨识/预测/适用域
指标及三种观测模式比较；带随机噪声的敏感性汇总使用固定种子并报告多次
运行统计。详细的信息边界、K–`q0`、静态 `L2` 和几何误差吸收风险见
[geometry_error_audit.md](geometry_error_audit.md)。

`--all-sensitivity` 用于 L1、L2 与髋中心正负方向比较；
`--noise-seed-study` 对随机噪声场景执行固定种子的重复统计；
`--all-core-subjects` 支持 baseline、hip_stiff、knee_stiff 和 heavy_leg 的
核心场景软件批处理。它们都不会调用真实机器人。

阶段 4.5D 仍然只是虚拟几何和观测误差的离线软件验证。它不接入真实
ROKAE 控制环，不导入真实 SDK，不修改机器人控制、采集、安全、硬件代码
或安全阈值，也不包含轨迹优化、PINN、MPC 或触觉。

## 阶段 5A：标准康复骨架参考轨迹导入

阶段 5A 把外部三维人体关键点 CSV 转换为当前二维下肢模型可复用的参考
轨迹。它先由左右髋、静止侧腿和运动侧腿建立身体局部矢状面，再从大腿和
小腿投影计算髋膝角。原始 CSV 的任意两个坐标轴不会被直接冒充当前模型的
床面 `x-z` 坐标。角度约定始终是：

```text
theta_shank = q_hip - q_knee
```

脚踝关键点只是骨架观测。机器人束缚带等效牵引点仍由现有
`forward_kinematics` 和配置中的 `L1/L2` 重新计算；因此
`x_ankle_observed_m/z_ankle_observed_m` 与 `x_pull_m/z_pull_m` 被并列保存，
但不会被强制设成同一个点。超出当前髋膝范围的样本只会标记
`joint_limit_valid=false`，不会静默裁剪成边界角度。

输入文件没有可靠单位或采样率元数据时，调用者必须显式声明 `--unit`。
未提供 `--fps` 时只生成坐标系、角度、平面性、周期、正运动学和几何图；
速度、加速度以及四名虚拟受试者的动力学结果保持不可用，不能用默认帧率
伪造。提供可信 fps 后，导数优先复用阶段 4.5D 的离线 Savitzky-Golay
估计，并调用现有完整动力学和力映射模块。

几何分析示例（没有动态结论）：

```bash
python -m lower_limb_sim.run_reference_trajectory \
  --input bone_return_3_leg.csv --unit mm --leg right
```

只有知道采集帧率后才可运行动态接入，例如：

```bash
python -m lower_limb_sim.run_reference_trajectory \
  --input bone_return_3_leg.csv --unit mm --leg right --fps FPS_HZ
```

默认周期选择只考虑完整、连续且较平滑的屈伸周期；文件首尾被截断的周期
不会静默成为标准周期。可用 `--start-frame/--end-frame` 做显式帧段选择，
或用 `--cycle-index` 选择已检测周期。局部坐标轴、平面外误差、骨段长度、
单位来源、帧率来源、周期选择原因以及所有跳过项都会写入 `metadata.json`。

该参考轨迹来自用户提供的骨架 CSV，目前没有临床来源、单位和帧率元数据
可以证明其为临床验证轨迹。所有结果标记为
`source_trajectory_type=provided_rehabilitation_reference` 和
`simulation_status=software_only`，不接入真实 ROKAE 控制环，也不修改控制、
采集、安全、硬件、SDK 或安全阈值。

## 阶段 5B：参考路径重定时与虚拟受试者比较

阶段 5B 不重新解析骨架 CSV，也不猜测原始 fps。它读取阶段 5A 的已处理
代表周期，把屈曲和伸展分别写成几何相位路径 `q_ref(s)`，再给路径指定一个
新的最小 jerk 时间律。输出中固定记录：

```text
source_timing_status = unknown
retimed_trajectory = true
retimed_timing_is_original = false
```

因此图和 CSV 中的秒、速度和加速度只属于软件重新指定的慢速、标称和快速
方案，不能解释为原骨架动作速度。PCHIP 在每段的几何路径相位上插值，最小
jerk 只决定沿这条曲线如何前进；它不会把髋膝路径退化成起点到终点的关节
直线。

解析导数使用标准链式法则：

```text
q_dot  = dq_ref/ds * ds/dt
q_ddot = d2q_ref/ds2 * (ds/dt)^2 + dq_ref/ds * d2s/dt2
```

第二式必须是加号；伸展方向已经包含在 `dq_ref/ds` 的符号中。实现不会对
最终角度做简单 `np.diff`，也不会使用未知 fps 下阶段 5A 留空的导数字段。
三种默认新时间方案是 `slow=12+12 s`、`nominal=6+6 s`、
`fast=3+3 s`。

### ROM 门控和幅值映射

逐点 `clip` 被禁止。当前正式配置直接读取 `ROM_PROTOCOL_V2` 的
`0~120° / 5~145°`。旧 Stage 5B 数据仍按其生成时协议保留为 legacy；重新
运行时，143° 路径在 V2 下合法，不再需要临时扩展 ROM：

```bash
python -m lower_limb_sim.run_reference_retiming
```

旧的显式幅值映射接口只保留用于 legacy 方法回归；它不得成为正式 active
pipeline 的临时 ROM 覆盖。V2 正式重跑不映射、不裁剪 active reference：

```bash
python -m lower_limb_sim.run_reference_retiming
```

也可以选择阶段 5A 已检测的其他完整周期，或只运行一个自定义时长方案：

```bash
python -m lower_limb_sim.run_reference_retiming \
  --cycle-index 2 --profile nominal \
  --flexion-duration 8 --extension-duration 10
```

批准 ROM 后，四名虚拟受试者复用完全相同的几何路径，动力学继续调用现有
`full_dynamics`、`force_mapping`、Jacobian、`dynamic_subject` 和配置中的
`L1/L2`。骨架 `RAnkle` 仍只作解剖观测比较；`x_pull_m/z_pull_m` 始终由
`forward_kinematics` 的束缚带等效作用点重建。

`domain_membership_estimated` 是相对于第四阶段 clean 训练状态的六维
`q/dq/ddq` 轴对齐包围盒检查。它会明确报告域外比例，但不是统计置信域、
临床安全域或机器人可执行性证明。轨迹超出 ROM、辨识适用域或力映射有效域
时不会静默删除，也不会被称为可执行轨迹。

PCHIP 保证角度路径连续且一阶导连续，但其分段二阶导在内部结点可以变化；
因此软件计算的加速度和惯性力矩仍应看作路径平滑敏感性结果。当前代表周期
的起点和终点也不完全重合，CSV 会保存 closure error；它可以作为一次往返
参考动作，但在额外闭环处理前不能直接无缝循环播放。

阶段 5B 仍保留阶段 5A 的平面外误差、局部坐标系和单位不确定性边界。它只
证明用户提供的路径形状可以在软件中被重新定时并用于虚拟受试者比较；不含
模型失配、几何误差扩展、PINN、MPC、轨迹优化或触觉，也不接入真实 ROKAE
控制、采集、安全、硬件或 SDK。

## 阶段 5C：闭合执行参考、局部辨识与候选筛选

阶段 5C 永久并列保存两个版本。`reference_measured_asymmetric` 是阶段 5A/5B
提取的完整屈曲和伸展路径，保留真实的首尾差异并标记
`repeatable_loop=false`。`reference_closed_symmetric` 只取测得的“起点到
最大膝屈曲”分支，再用该分支的严格时间反向作为伸展路径。后者的首尾角度
和等效牵引点闭合，但对称回程是软件构造，绝不标成原始骨架测量。

### 显式膝 ROM 审批和 fail-closed 门控

阶段 5C 的正式动力学、五参数辨识和候选筛选直接读取
`formal_experiment_manifest.json`。命令行不再接受 ROM 上下限，也不能形成
第二套 active ROM。`None` 和旧幅值映射只存在于低层 legacy 回归测试；原始
角度列永久保留，程序从不逐点 `clip`。

默认仅审计命令：

```bash
python -m lower_limb_sim.run_reference_candidate_evaluation
```

正式 V2 候选审计命令为：

```bash
python -m lower_limb_sim.run_reference_candidate_evaluation
```

5°～130° 只允许作为明确标记的 legacy 回归或冲突输入，用于验证 loader
fail closed；它不是正式 active 候选实验范围。

### 参考邻域辨识、适用域和候选边界

局部辨识数据固定为四条 train、一条 validation 和一条 test。test 不参与
五参数拟合、适用域建立、阈值选择或候选扰动幅度选择。估计器仍只读取
`q/dq/ddq` 和 `Fx/Fz`，由 `J.T @ F` 重建测量力矩；输入表不含真实参数或
`tau_total`。适用域只由 train 的六维估计状态建立，并分别报告 q、dq、
ddq 哪一类状态缺少覆盖；不会通过放宽邻域阈值掩盖覆盖不足。

主结果只使用 slow（12+12 s）和 nominal（6+6 s）。fast 保留为
`software_stress_test`，不进入 C0～C8 候选排名。幅值扰动使用端点为零的
平滑基函数，相位扰动保持膝角峰值，所有候选继续满足：

```text
theta_shank = q_hip - q_knee
```

候选硬约束包括 ROM、闭合、工作空间、力映射、Jacobian condition 和局部
辨识域覆盖。任一不满足都会输出明确拒绝原因。排名只对可行候选做无权重
Pareto 比较：峰值髋力矩、峰值膝力矩、RMS 力矩、joint jerk cost 和总
时长；不会把这些量任意加权成“舒适度”。末端力只用于虚拟软件相对比较，
不是患者康复力结论，也不是机器人安全阈值。

阶段 5C 仍然只是未知原始 fps、仍有平面外运动和几何不确定性的虚拟数据
验证。它不接入真实 ROKAE 控制环，不导入 SDK，也不修改 hardware、
collection、control、安全配置或安全阈值。

### 本次正式ROM审批与C2闭合参考

正式 reference experiment 与全局 active pipeline 统一读取
`ROM_PROTOCOL_V2`：髋 `0°～120°`、膝 `5°～145°`。现 active measured
asymmetric reference 的膝范围约 `18.319°～124.787°`，完整保留原幅度，
`rom_mapping_applied=false`，不做逐点裁剪。旧 symmetric Stage 5C 结果仍位于
`data/reference_candidates/`，仅作 legacy provenance。

原 `reference_closed_symmetric` 使用PCHIP，保留作来源审计，不被覆盖。新增
`reference_closed_c2` 只对测得的flexion分支拟合五次B-spline，extension仍是
flexion的严格反向。为避免反向路径在最大屈曲点和周期首尾翻转奇阶导数，
flexion起点和峰值处的一阶、三阶相位导数都约束为0；因此内部样条和完整
global-phase周期在两个接缝处都达到C4（高于本阶段要求的C2）。相位CSV只保存
一次共享峰值，`global_phase` 严格递增。离线形状门要求髋/膝最大偏差不超过
0.5°、等效牵引点最大偏差不超过2.5 mm、无新增ROM违规且本次固定采样审计的
加速度离群告警为0；任一不满足都会fail closed。正式输出包括：

```text
reference_closed_c2_phase.csv
reference_closed_c2_slow.csv
reference_closed_c2_nominal.csv
reference_c2_comparison.csv
```

minimum-jerk仍只控制沿C2路径的相位。slow为12+12 s，nominal为6+6 s，起止
速度和加速度为0，`theta_shank=q_hip-q_knee`，脚踝观测仍不等于等效牵引点。
这些平滑和偏差门是软件数据质量门，不是真实机器人安全阈值。

### 实测非对称周期闭合参考（当前 active）

当前正式机器人 reference 不再使用上述 `extension=reverse(flexion)`。离线命令：

```bash
python -m lower_limb_sim.run_reference_measured_asymmetric
```

会从 Stage 5A 的完整 550 帧序列，用批准 ROM 归一化后的髋/膝联合 PCA 相位、
双关节方向和同相位边界联合搜索检测自然周期；旧 knee-only cycle 表只作交叉
引用，不参与边界或评分。四个候选的 `(start, peak, end)` 为：

```text
C0 5593,5686,5729  pull=2.946 mm  score=0.734  projection/ROM fail
C1 5737,5798,5830  pull=2.695 mm  score=1.511  projection fail
C2 5844,5895,5934  pull=4.507 mm  score=1.032  eligible, selected
C3 5945,5995,6029  pull=180.043 mm score=40.500 eligible
```

因此不能只取数值总分最小的 C0；先严格要求 complete、projection-valid 和
hip `0~120°` / knee `5~145°`，再在 eligible 集合中取联合 closure score 最小的
C2。完整原始分项保存在 `reference_cycle_closure_audit.csv`。Stage 5A metadata
没有可靠 source fps，所以物理 `dq/dt` closure 留空并记录
`source_fps_not_provided`，不会把每帧差分伪称为 rad/s。

`reference_measured_raw.csv` 是 5844→5895→5934 的逐值副本，包含实测 flexion
和实测 return，保持 `source_values_modified=false` 和
`measured_extension_is_reversed_flexion=false`。天然接缝误差为
`Δhip=-0.452446°`、`Δknee=-0.216276°`、`Δx=+1.836 mm`、
`Δz=-4.116 mm`、pull `4.507 mm`，高于数值闭合容差，因此另建
`reference_measured_asymmetric_closed`，不覆盖 raw。

闭合版本先做 5 点三阶局部 Savitzky-Golay 数据质量平滑，再对 4 倍细分的
361 个 phase anchor 拟合 periodic cubic B-spline；所有改变都相对未平滑 raw
PCHIP 密集量化。最大/RMS 改变量分别为 hip `0.246189°/0.015139°`、knee
`0.189134°/0.017795°`、pull `2.255804/0.134420 mm`，均在既有离线路径门
`0.5°/0.5°/2.5 mm` 内。原始屈伸不对称 RMSE 为 hip `19.781039°`、knee
`20.118822°`、pull `140.013450 mm`；闭合后分别保留 `99.999226%`、
`99.999133%`、`99.998682%`。

周期接缝、最大屈曲点和全部内部 knots 的 position/velocity/acceleration
continuity warning count 都是 0。minimum jerk 只推进两条实测分支的 phase：
slow 为 `13.6+10.4=24 s`，nominal 为 `6.8+5.2=12 s`；首末和峰值速度/加速度
为 0，FK pull 与 start-anchored relative 首末位移严格闭合。输出包括：

```text
reference_cycle_closure_audit.csv
reference_measured_raw.csv
reference_measured_asymmetric_closed.csv
reference_measured_asymmetric_closed_slow.csv
reference_measured_asymmetric_closed_nominal.csv
reference_measured_asymmetric_domain_coverage.csv
reference_version_manifest.csv
reference_measured_asymmetric_metadata.json
all_detected_cycles_closure.png
selected_natural_cycle.png
measured_flexion_vs_extension.png
raw_vs_periodic_closed.png
asymmetry_preservation.png
new_reference_pull_path.png
```

冻结的既有 Stage 5C train-only 六维轴对齐域没有用新 reference 重新拟合，90%
门也没有放宽。新 slow 覆盖 `401/401=100%`，可作为唯一 active slow；新
nominal 只覆盖 `266/401=66.334%`，因此 `active_reference=false`、
`formal_execution_allowed=false`。active slow 文件 SHA-256 为
`f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`。
旧 symmetric/C2 文件继续保留作 legacy software comparison，均不是 active。

## 阶段 6A：人体参考坐标到机器人 Base 的离线转换

阶段 6A 默认读取已通过审计的
`reference_measured_asymmetric_closed_slow.csv`，把等效牵引
点从人体局部矢状面 `H` 转换到机器人 Base 坐标系 `B`，再用固定工具偏置
换算TCP坐标系 `T` 原点的位置。旧 `reference_closed_symmetric` PCHIP 和
`reference_closed_c2` 文件仍可被显式指定用于审计，但不再是默认机器人离线输入。输出是供未来人工审核的
CSV文件，不是机器人运动命令发送器。模块没有SDK、连接、上电、伺服或运动
代码，metadata始终记录：

```text
robot_execution_approved = false
trajectory_generated_offline_only = true
```

实验标定必须从外部 JSON 显式提供；仓库没有默认实验室坐标，也不会猜测
人体相对机器人放置。仓库中的
`lower_limb_sim/config/robot_rehab_calibration_template.json` 只是待填写和待复核
的表单，坐标与姿态字段保持 `null`、`reviewed=false`，不能用于导出。经过人工
标定复核的文件必须把所有字段填写为有限值，并将 `reviewed` 明确设为布尔值
`true`。必需字段是：

```text
hip_center_in_base_m: [h_x, h_y, h_z]
human_x_axis_in_base: [x_Bx, x_By, x_Bz]
human_z_axis_in_base: [z_Bx, z_By, z_Bz]
tool_offset_m: [o_Tx, o_Ty, o_Tz]
tcp_orientation:
  representation: euler_xyz_rad
  values_rad: [r_x, r_y, r_z]
approved_hip_rom_deg: [0, 120]
approved_knee_rom_deg: [5, 145]
reviewed: true
notes: "human reviewer and calibration provenance"
```

`human_x_axis_in_base` 和 `human_z_axis_in_base` 必须已经是单位正交向量；程序
会拒绝而不是静默归一化或正交化。右手系的 `human_y_axis_in_base` 由
`z_H × x_H` 得到，`R_base_from_human` 的三列依次是 `[x_H, y_H, z_H]`。
Stage 6A 命令 CSV 的 `tcp_rx_rad/tcp_ry_rad/tcp_rz_rad` 只接受
`euler_xyz_rad`，与项目现有 active XYZ-Euler/RPY 数学约定一致；它不会把
旋转向量三个分量混写进同名列。但这些数值是否与未来 ROKAE 具体 pose 接口
语义一致仍未真机验证，不能由离线文件推断。

按上述变量定义，`R_base_from_human` 的列向量就是人体正轴在 Base 中的方向。
所以人体点沿 `+x_H` 或 `+z_H` 增加时，Base 中的位移必须分别与
`human_x_axis_in_base` 或 `human_z_axis_in_base` 同向。坐标变换为：

```text
p_pull_B = hip_center_B + R_base_from_human @ [x_pull_H, 0, z_pull_H]
```

`tool_offset_m` 定义为“在 `T` 中表达的、从 TCP 原点指向束缚带连接点的
向量”（不是反方向），因此牵引点位置等于 TCP 原点加上旋转到 Base 的偏置，
反解 TCP 原点时使用：

```text
p_tcp_B = p_pull_B - R_base_from_tcp @ tool_offset_T
```

生成命令（把路径替换为本次实验经过审核的真实标定文件）：

```bash
python -m lower_limb_sim.run_robot_trajectory_export \
  --calibration-json PATH_TO_REVIEWED_CALIBRATION.json
```

如果不提供标定，或者标定仍为 `reviewed=false`、含 `null`/非有限值、轴未
单位化/不正交或旋转矩阵非法，程序会 fail closed；不会制造
`reference_robot_trajectory.csv` 或预览图。若输入保留阶段 5C 的
`formal_execution_allowed=false`（例如未审批或上游审计无效的版本），这个
门控会逐样本传播到 Stage 6A；坐标数学正确也不能把上游无效段改成有效命令。
此时显式标定仍可生成一份供审计的 CSV，但运行结果保持
`preexecution_audit_passed=false`、CLI 返回非零，不能把“文件已生成”解释成
“轨迹已批准”。输入若缺少逐样本有效性来源，也默认无效而不是默认通过。
此外，轨迹中持久化的approved hip/knee ROM必须与reviewed标定文件中的ROM
完全一致，且每个角度样本必须位于该范围内；不一致或越界都会逐样本标记并
阻止执行前审计，不能靠 `reviewed=true` 绕过审批。

输出审计只检查时间严格递增、有限值、位置/速度/加速度连续性、首尾闭合、
单帧位置跳变、旋转正交性以及工具偏置恢复误差，并报告实际 TCP XYZ 范围、
最大位移、最大笛卡尔速度和最大笛卡尔加速度。它没有自行设置真实机器人的
速度、加速度或工作空间安全阈值。默认C2路径已消除本次PCHIP的12处加速度
离群告警；若显式审计旧PCHIP，二阶导告警仍会保留，不能解释成真实机器人
安全判定。

已生成 CSV 的 dry-run 只读取、打印或验证样本，不按时间回放，也不连接
机器人。它不会只信任 CSV 自报的 `trajectory_valid`，而会重新检查上游
sample/formal gate、`theta_shank`、正运动学牵引点一致性、闭合和单帧跳变：

```bash
python -m lower_limb_sim.run_robot_trajectory_export \
  --dry-run lower_limb_sim/data/robot_trajectories/reference_robot_trajectory.csv

# 额外打印前 10 行；使用 -1 可打印全部
python -m lower_limb_sim.run_robot_trajectory_export \
  --dry-run lower_limb_sim/data/robot_trajectories/reference_robot_trajectory.csv \
  --print-samples 10
```

阶段 6A 仍然只是虚拟参考轨迹与显式离线标定的文件级验证。离线正交矩阵、
闭合轨迹和 dry-run 成功均不能替代真实 ROKAE 的坐标标定、安全审批或低速
分阶段验证。

## 后续扩展边界

```text
workspace atlas
      ↓
quasi-static virtual-subject force map
      ↓
software-test full dynamic trajectories
      ↓
software virtual-subject parameter identification
      ↓
timing alignment and model-mismatch generalization validation
      ↓
geometry calibration and kinematic-observation validation
      ↓
external skeleton reference-trajectory import
      ↓
reference-path retiming and virtual-subject dynamics audit
      ↓
closed reference, local identification and candidate Pareto screening
      ↓
C2-continuous closed rehabilitation reference
      ↓
offline human-H to robot-Base trajectory transform and pre-execution audit
      ↓
start-anchored real-robot acquisition and validation
      ↓
offline real-episode identification and candidate comparison
```

工作空间图谱、虚拟准静态力地图、完整动态测试轨迹、第四阶段虚拟参数辨识
以及 4.5C/4.5D 软件验证、阶段 5A 外部骨架参考导入，都不能直接解释为
真实患者参数、受力安全性、舒适性、临床有效性或机器人可执行轨迹。
