# CONTROLLED_ACTUATION_V3：受控辅助机制 pilot

本实验检验辅助时机、髋膝辅助力矩分配与运动持续时间是否暴露有决策意义的个体差异。版本与冻结 V1、V2 轨迹域分开；V1 的 30-seed 扩展继续暂停。

## 候选与模型

| 因素 | 固定设置 |
|---|---|
| 运动持续时间 | 参考周期的 `0.9 / 1.0 / 1.1` 倍 |
| 辅助时机 | 每个屈曲、伸展分支内相位 `0.25 / 0.50 / 0.75` |
| 髋分配比例 | `0.25 / 0.50 / 0.75`；膝比例为 `1 − 髋比例` |
| 总辅助强度 | 每个分支采样后的 `max(|髋辅助扭矩| + |膝辅助扭矩|) = 2 Nm` |
| 波形 | C2 紧支撑多项式 `(1−u²)³`，支撑外为零，相位半宽 `0.20` |
| 方向 | 公开参考轨迹每分支的关节位移方向；所有主体一致 |

共 27 个有辅助候选。`domain.reference` 是 `(1.0, 0.50, 0.50)` 的有辅助候选；评估器另外请求同主体、时长比例 1 的**无辅助** native trace 作为统一归一化参考。ROM、q 曲线保持原样，dq 除以时长比例，ddq 除以时长比例平方。

实现为 `tau_net = tau_native − tau_assistance`。这里 `tau_net` 是预定运动下扣除指定辅助力矩后的残余所需驱动力矩，不是肌肉用力、测得的束带力或前向动力学控制结果。公开运动方向与所需驱动力矩方向可能相反，因此辅助也可能增大负荷；实现不读取主体 torque 决定辅助方向，不裁剪负净力矩。

固定峰值不等于固定冲量、机械功或平方力矩积分。输出逐关节的峰值、绝对冲量、正负功、平方力矩积分以及原始分支 RMS；同时保留每个时长的无辅助 E3，区分时长效应与辅助效应。

## 先运行 development pilot

默认选冻结 development ID 排序后的前三个，不根据响应挑主体。每主体评估 27 个受控响应；相同运动轨迹的 native trace 可以共享内容寻址缓存。封存主体在读取模型前被拒绝。

运行前写 `protocol.json`，记录主体列表、全部因子、约束、判据、源码哈希、作用域和无噪声条件；之后保存其哈希至 `completion.json`。异常只能产生 `complete=false`。已有输出目录不能覆盖。

所有 E3/E2/peak 均相对同主体无辅助参考；机械约束沿用 `E2 ≤ 1.01`、`peak_ratio ≤ 1.10`，这些仅为模拟指标限制。辅助幅值校验另行记录。跨主体比较同时报告：

1. 每主体可行 oracle、数值并列数与跨主体排序相关。
2. 在所有 pilot 主体均可行的候选中，平均 E3 最小的单一公共候选。
3. 公共候选与每主体 oracle 的相对 regret：`(E3_common − E3_oracle) / E3_common`。
4. 允许公共策略和个体 oracle 都选择无辅助参考后的 regret。若所有辅助候选都差于无辅助，实际收益空间为零，不能凭辅助方案彼此的差异推进。

公共候选使用完整 pilot truth 选择，仅为诊断个性化空间的比较对象；不能把它冒充独立训练得到的测试基线。没有共同可行候选时明确记录缺失；没有可行 oracle 时不会用不可行点顶替。

## 运行前的决策判据

保留上一版探索规则：至少两个主体且每主体有可行 oracle；oracle 种类大于 1；并满足中位 Spearman < 0.99，或至少 20% 主体相对无辅助参考的 regret ≥ 0.5%。

在首次 V3 pilot 前增加必要条件：存在共同可行有辅助候选；至少 20% 主体相对**最佳共同可行策略（允许无辅助回退）**的 regret ≥ 0.5%，个体 oracle 同样允许选择无辅助。仍报告强制辅助条件下的 regret 以便区分两者。这是工程筛查阈值，不是显著性检验；3 主体时，20% 条件实际要求至少 1 个主体。

两个规则均通过，状态才为 `DEVELOPMENT_FOLLOWUP_ELIGIBLE`，允许再制定扩大 development、比较算法和 null/positive 对照的协议。未通过则为 `HOLD_NO_PRACTICAL_PERSONALIZATION_SIGNAL`，保留结果并停止该版本扩展。任何 pilot 结果均不直接进入确认实验；独立主体、对照、算法与门控冻结仍是后续条件。

## 复现

```powershell
$env:PYTHONUTF8 = '1'
.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.actuation_screen `
  --output-dir outputs\myoleg_controlled_actuation_pilot_v3_reproduction
```

入口：[候选与适配器](../../lower_limb_sim/myoleg_benchmark/controlled_actuation.py)、[pilot 评估器](../../lower_limb_sim/myoleg_benchmark/actuation_screen.py)。结果：[pilot 报告](../../outputs/myoleg_controlled_actuation_pilot_v3/REPORT.md)。这一实验没有运行 SAST/EG-CPI-BO，因此不报告个性化算法收益。

## 本次 pilot 结果与决定

已完成 `MYOLEG_VP_001/002/003` × 27 候选，共 81 条有限、有效的响应。有效不等于可行：三主体分别只有 7、9、8 个候选通过机械指标限制；57/81 条违反 E2，0 条违反 peak 限制。所有主体共同可行的有辅助候选共 7 个。

| 指标 | 结果 |
|---|---:|
| 不同的可行 oracle 数 | 2 |
| 全候选 E3 排序 Spearman 中位数 / 最低 | 0.981685 / 0.942613 |
| 公共方案 regret 中位数 | 0 |
| 公共方案最大相对 regret | 0.003589% |
| regret ≥ 0.5% 的主体数 | 0/3 |

最佳公共辅助方案是 `CONTROLLED_ACTUATION_V3:8`：持续时间 `0.9`、分支内辅助时机 `0.75`、髋膝分配 `0.50/0.50`。它也是 001 和 003 的个体 oracle；002 的 oracle 是 `:12`（`1.0/0.25/0.75`），额外收益仅 `0.003589%`。允许无辅助回退后的结论相同。

因此记录 `HOLD_NO_PRACTICAL_PERSONALIZATION_SIGNAL`：本版本停止扩展，不进入 24 主体或确认实验。该结果说明本次小 pilot 的最优决策差异很小，不能据此否定其他主体、剂量或生理模型下的个性化。下一版本若要修改辅助机制，应先明确辅助作用与主体差异的物理依据，再单独定义协议；本版阈值、强度和结果保留原样。

13 项新增测试覆盖波形预算、观测隔离、统一无辅助参考、common 可行性、无辅助回退与失败状态；连同相关 benchmark、门控及缓存回归共 70 项通过。输出的 protocol 哈希和执行源码哈希已核对。没有访问封存主体，没有运行任何确认实验或 V1 扩展。
