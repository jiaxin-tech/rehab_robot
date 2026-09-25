# MyoLeg 附加弹簧—阻尼开发试验

本试验把有单位的附加机械元件接到 MyoLeg 的 prescribed-state replay 力矩上，回答“固定 ROM 和 V3 候选下，明确的机械配置差异能否改变值得选择的辅助方案”。它不是患者模型、前向控制仿真、真实绑带力测量或个体化算法胜负试验。真机继续 NO-GO。

## 为什么修正原接口

上一版 `physical_interaction.py` 尚未接入 MyoLeg，存在三处物理语义问题：

1. 轨迹生成器已经按照持续时间缩放 `dq_rad_s`，阻力模块再次除以 duration 会重复计算速度变化。
2. `abs(q-q0)` 再乘 `-sign(dq)` 是始终阻碍运动的摩擦样力矩，不是恢复到静止位置的弹簧。真实弹簧在回程可以释放储能。
3. 辅助时机和 hip share 不应直接控制主体阻力的起始和分配，否则调整机器人辅助等于改变主体本身。

现采用 `EXTERNAL_LINEAR_SPRING_DAMPER_V2`：

```text
tau_elastic     = -scale * K * (q - q0)
tau_viscous     = -scale * B * dq
tau_interaction = tau_elastic + tau_viscous
tau_net         = tau_native - tau_interaction - tau_assistance
```

`K` 单位为 Nm/rad，`B` 为 Nm·s/rad，q 为 rad，dq 为实际 rad/s。弹性力矩依赖角度，本版 K 是常数，没有加入非线性刚度、反射或疲劳。Native 力矩已经包含 MyoLeg 的被动组织和肌肉贡献，不能再扣一次 native passive torque。附加元件是一个单独声明的外部机械条件。

辅助仍使用既有 `CONTROLLED_ACTUATION_V3` 的公开运动方向与固定 2 Nm L1 峰值。辅助时机、髋膝比例只控制辅助；duration 通过轨迹生成器对 q/dq/ddq/time 的转换生效。

## 预先固定的范围

使用排序后的前 3 个冻结 development MyoLeg 主体，每个主体交叉同样 4 种机械配置。没有读取确认主体响应，也没有扩大 V1 的种子数。

| 配置 | K：髋/膝，Nm/rad | B：髋/膝，Nm·s/rad | 用途 |
| --- | --- | --- | --- |
| ZERO_INTERACTION | 0 / 0 | 0 / 0 | 应复现旧 V3 受控辅助结果 |
| HIP_SPRING | 6 / 2 | 0.3 / 0.3 | 髋侧恢复力较强 |
| KNEE_SPRING | 2 / 6 | 0.3 / 0.3 | 膝侧恢复力较强 |
| DAMPING | 2 / 2 | 1 / 1 | 增加速度相关阻力 |

所有配置 q0 固定为公开参考轨迹起点；scale=1；20 Nm 是数值模型上限，不是临床安全阈值。以上系数是结果产生前声明的探索值，不是从患者或 native torque 拟合的系数。结果不足时不调大系数来制造成功。

候选仍为 duration=0.9/1.0/1.1、timing=0.25/0.5/0.75、hip share=0.25/0.5/0.75，共 27 个。3 个主体 × 4 个机械配置 × 27 个候选得到 324 条响应，但只有 3 个 native 基础主体，不能写成 12 位独立患者，也不能把 324 条候选当作独立样本。

每个配置的归一化分母为“同一 native 主体 + 同一附加机械元件、不带辅助、duration=1”的参考轨迹。E2≤1.01、peak ratio≤1.10 是既有机械比较约束。公共方案与 oracle 都允许回退到该无辅助参考；因此单纯辅助带来的共同收益不算个体化收益。

公共方案从完整 development 数据中选出，仅用于诊断。这不是独立开发集训练后用于确认集的 fixed common policy。oracle 只由 evaluator 在全域响应采集结束后计算，本试验没有运行 learner。

沿用 V3 工程筛选规则：必须有共享可行候选和不同 oracle，且至少 20% 的配置相对于公共方案存在 ≥0.5% 的额外收益；同时保留原 V3 的排序/参考收益诊断条件。全部等效力映射也必须有效，才能考虑进一步开发。这个筛选不是统计显著性检验，也不自动开放确认实验。

## 输出与复现

运行入口：[physical_interaction_screen.py](../../../lower_limb_sim/myoleg_benchmark/physical_interaction_screen.py)。从仓库根目录执行，目标目录必须不存在：

```powershell
$env:PYTHONUTF8 = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.physical_interaction_screen --output-dir outputs/myoleg_physical_interaction_reproduction
```

protocol 在任何 native 响应请求之前落盘，记录配置系数、候选、约束、公式版本和代码哈希。每个配置的 `traces_*.npz` 保存 q/dq/time、四类力矩、弹性/阻尼分解和势能；`landscape.csv` 保存 E2/E3/peak、功、可行性与力映射诊断；`oracle_summary.csv` 和 `within_profile_summary.json` 分别报告全配置对比与固定元件下的 native 主体对比。`completion.json` 记录完整性和所有输出文件哈希。

力通过同一工程坐标系下的 `J.T @ F = tau` 映射。交互等效力与净需求等效力分开保存。0.42/0.30 m 是声明的平面两连杆几何近似，500 N 是数值筛选上限；这些量不等于 MyoLeg 真实 cuff 接触力，更不是传感器实测。奇异、非有限或超限映射保留无效标记和原因。

单元与集成检查包括：弹簧回程释能、闭环弹性功、阻尼耗散、持续时间只缩放一次、改变辅助不改变附加阻力、零附加元件复现旧适配器、力矩平衡、力映射虚功、协议先于响应、失败不伪装为完成和 NPZ 重建指标。

实测重复性、噪声鲁棒性、参数辨识和独立确认均不由本次无噪声力学屏幕替代。若公共 regret 不足，保留该负结果，下一步转向测量有效性与可辨识性；若达到门槛，才设计冻结的 K=4/K=8 算法对比，不预定 SAST 获胜。

## 2026-09-25 实际开发结果

[运行报告](../../../outputs/myoleg_physical_interaction_pilot_v1/REPORT.md)、[协议](../../../outputs/myoleg_physical_interaction_pilot_v1/protocol.json)和[汇总](../../../outputs/myoleg_physical_interaction_pilot_v1/summary.json)已生成，324 条响应完整有效。

| 结果 | 实际值 | 含义 |
| --- | ---: | --- |
| 基础主体 / 机械配置 / 候选 | 3 / 12 / 27 | 交叉敏感性设计，非 12 位独立患者 |
| 不同可行 oracle | 6 | 机械配置确实可以改变最优候选 |
| 所有配置共享的可行候选 | 7 | 仍存在可共用方案 |
| 相对公共方案额外收益中位数 / 最大值 | 0.245606% / 0.466442% | 均未超过 0.5% 工程门槛；达标配置为 0/12 |
| E2 超限 / peak 超限 | 184 / 0 | 全域离线筛查中的不可行候选，非在线执行违规 |
| 无效力映射 | 0 条 | 数值等效力映射通过，不等于真实力传感器验证 |
| 力矩分解最大误差 | 0 Nm | 四类力矩平衡一致 |
| 弹簧功加势能变化最大绝对残差 | 0.00017657 J | 使用既有离散 q/dq 轨迹积分的数值残差，非实测误差 |
| 零附加元件的最大公共 regret | 0.003589% | 保留旧 V3 null 对照水平 |

结论为 `HOLD_NO_PRACTICAL_PERSONALIZATION_SIGNAL`。相比原先几乎没有决策差异的辅助试验，这次物理配置差异带来了更明显的选择变化，但本次设定下的收益仍不足以越过预设门槛。不能因为最大值接近 0.5% 就降低门槛或继续调系数。

固定元件后只比较三个 native 基础主体，最大公共 regret 分别为：ZERO 0.003589%、DAMPING 0.013843%、HIP_SPRING 0.266316%、KNEE_SPRING 0%。因此跨元件的可控机械条件贡献了主要排序变化；这个结果不能升级为患者异质性结论。

本轮没有运行 K=4/K=8 新算法比赛，也没有声称 SAST 成功或失败。下一步保持本版本冻结：先把实际测量的坐标、符号、时间对齐、重复性和原生阻塞问题处理清楚，再用可用测量辨识机械参数及误差范围。只有测量支持的主体差异超过噪声且带来实际 common regret，才冻结新的低预算比较。无有效测量时，以 native null + 本次机械敏感性负结果 + 解析软件压力测试组成有限范围论文证据。
