# MyoLeg 个性化实验执行计划 V2

**版本：** 2026-09-23
**实验截止：** 2026-10-31
**论文写作起点：** 2026-11-01
**主线定位：** simulation algorithm / evidence-gated personalization

## 当前执行结果（Stage 2 已完成）

24 个 development 主体的两个候选族已经完成 evaluator-only 全域重放：`BETA_TIMING` 为 324 点/主体，`KEY_POSTURE_TIMING` 为 387 点/主体，共 17,064 条 landscape 记录。两个候选族各自只有一个可行 oracle：BETA `:288` 和 KEY `:383`，均为 24/24 主体共享；对应公共候选的 common regret 对所有主体均为 0。K=4 时 `PHYSICS_GREEDY`、`RESIDUAL_GP_GREEDY` 和 `MODEL_INFORMED_BO_EI` 在两个候选族均捕获 full oracle。结果位于 `outputs/myoleg_personalization_audit_v1`，`completion.json` 为 `complete=true`，`scope_audit.json` 记录 sealed 访问为 0。

这意味着当前 V1 的“没有决策相关个体最优差异”已经不是执行池漏采样造成的；它仍可能有负荷幅度和可行域差异，但不能用来证明当前 coordination personalization 有收益。下一阶段转向灰箱跨轨迹验证和受控正向差异实验。

## 1. 先回答“个性化到底是谁的问题”

当前结果不能归结为 BO 选择器失败。证据指向两个更早的问题：

1. **V1 任务异质性不足。** 24 个 development 主体在完整候选域中的最优轨迹高度一致，subject×trajectory interaction 很小，因此当前任务没有给个性化选择提供足够强的目标。
2. **五参数灰箱模型失配。** MyoLeg 真值包含质量/惯量和六维肌肉因素，而适配器只有五个参数；`mass_scale` 反复撞到上界，参考轨迹 torque RMSE 仍约 3–5 Nm。这不能通过放宽上界来伪造个体差异。

因此研究问题改成：

> 在没有足够个体化证据时，算法能否安全回退公共轨迹；在观测到可重复、决策相关的个体差异时，算法能否在相同预算下降低个体 regret 和约束违反？

V1 是 **null-control**，用于测 gate 的假阳性；新建的 controlled stress cohort 是 **positive-control**，用于测 gate 的检出率和个体化收益。两者都不能被表述为患者生理真值。

## 2. 冻结规则（先写入 manifest，再运行）

- 不扩展当前 V1 的 `mass_scale` 上界，不调整 V1 候选域来制造不同 oracle。
- 不读取 sealed subject；development 只用于开发和方法选择，确认集必须另行冻结。
- learner 每轮只能看到自己的已执行 payload、torque trace/endpoint 和观测噪声；不能读取完整候选表、缓存目录或评估器生成的 oracle。
- 所有主比较使用相同 subject、相同候选预算、相同噪声 seed 的成对结果。
- full-domain oracle 只属于离线 evaluator，用于计算上界、regret 和识别性，不得传给 selector。
- 首次两次试验固定为 `REFERENCE` 和预先声明的 diagnostic probe；probe 参数、顺序、seed 在 truth reveal 前冻结。
- 主效应量为每个主体的
  `Delta_i = loss(COMMON_POLICY)_i - loss(SAST_BO)_i`。
  预注册实用阈值为 `0.5%`，主体等权 bootstrap 95% CI，不能只报告平均值。

## 3. 分阶段执行

### Stage 0 — baseline freeze（9 月 23–24 日）

冻结当前结果目录 `outputs/myoleg_benchmark_v1/development_20260923_r2`，记录 git commit、依赖锁、候选域大小、subject 清单和 cache identity。暂停 30-seed 扩展；已有 V1 结果作为不可变基线。

新增真正的公共策略基线 `COMMON_POLICY`：只用 development 训练主体的预注册规则（reference 或 population mean 的 physics-greedy），冻结后在每个评估主体执行。`REFERENCE` 作为诊断基线保留，但不能代替 common-policy 主比较。

交付物：`baseline_manifest.json`、common-policy candidate、同预算比较表。

### Stage 1 — gray-box cross-validation（9 月 24–27 日）

沿已有执行前缀 `K=1,2,4,8`，只用当时已观测数据拟合五参数适配器，并在未执行候选上评估：

- torque RMSE/MAE、R²/VAF；
- held-out E3 MAE/RMSE；
- E3 排名 Spearman/Kendall；
- 预测 top-1 的真实 regret、top-5 overlap；
- E2/peak 约束 precision/recall；
- 参数边界命中、Jacobian condition、finite coverage。

预注册“可用于决策”的门槛：held-out E3 MAE ≤ 0.005、主体中位 Spearman ≥ 0.90 且 95% CI 下界 ≥ 0.80、预测选择真实 regret ≤ 0.005、至少 80% 主体有限且可行。若边界命中且 held-out MAE > 0.01，写入 `MODEL_MISMATCH`，转 residual-only，不扩大先验范围。

### Stage 2 — full-domain identifiability audit（已完成，9 月 23 日）

对 24 development + native，在不改 V1 代码和候选域的条件下，独立 evaluator 运行 no-noise full landscape。当前三维域实际点数、运动学 rejection 数必须写入结果，不能直接套用旧 V3 的 625 点数字。

输出：

- 每主体 oracle、tie set、boundary、near-oracle coverage；
- common、LOSO-common、personalized oracle 及 relative common regret；
- pairwise oracle distance、transfer regret、rank correlation/top-set overlap；
- 已执行候选池 vs 完整域的 pool capture、exploration regret、recommendation regret；
- `scope_audit.json`：sealed access count 必须为 0，记录 source hash 和 evaluator 版本。

支持“V1 存在决策相关个体化信号”的 conjunction 仍记录为后续版本的候选判断：interaction ≥ 0.25%，unique oracle ≥ 4，median common regret ≥ 0.5%，p75 common regret ≥ 1%，median transfer regret ≥ 0.1%，并且 ranking/top-set 依赖性达到预设阈值。本次 V1 不满足这些条件，因此只作为 null-control，不得声称算法错过了个体最优。

### Stage 3 — SAST-BO（10 月 1–10 日）

新算法正式名：**Subject-Adaptive Safe Trust-Region BO (SAST-BO)**，中文为“主体自适应安全信赖域贝叶斯优化”。它保留证据门控，个性化对象是观测历史和主体响应，不是 subject ID。

每个主体的流程：

1. 执行 reference；
2. 执行固定、正交、风险受限的 calibration probes（髋/膝方向或关键相位偏移）；
3. 由 torque trace、branch features、E2、peak 和重复噪声拟合主体 residual latent 及局部二次 surrogate；
4. 计算 `P(subject×trajectory interaction > δ | D_i)` 和跨主体 rank disagreement；
5. 若证据不足，输出 `PERSONALIZATION_INACTIVE`，使用 COMMON_POLICY/reference；
6. 若证据足够，启用 residual GP + constrained EI，只在安全信赖域内选点；
7. 约束使用预测 E2、peak 的上置信界，要求 `P(E2 <= 1+δ_e2, peak <= 1.1) >= 0.95`；若没有安全候选，触发 `SAFETY_LOCK` 并回退 reference。

建议初始实现：K=4 做可行性 pilot，K=8 做主结果；trust radius 在 `MODEL_MISMATCH` 或高不确定性时收缩。算法日志必须包括 gate 状态、probe、posterior uncertainty、safety lock 和最终推荐，不能只保存最终 candidate。

等预算对照：`COMMON_POLICY`、current `PHYSICS_GREEDY`、raw GP-EI、calibration-only local quadratic、Random/SpaceFill。所有对照使用同一预算和成对 seed。

### Stage 4 — controlled stress test（10 月 11–20 日）

先做 4–8 个 development profile 的小试，再扩到 16 个；参数在 manifest 冻结后才运行。优先使用可解释且不依赖肌肉结构的受控差异：个体负荷/峰值约束、ROM/速度边界、预声明的姿态相关阻力形状。只有通过结构完整性和文献依据后，才考虑肌肉结构因素。

每个 scenario 同时包含：

- `COMMON_OPTIMUM` null：不应触发 gate；
- `DIVERGENT_OPTIMA` positive control：应检出 interaction 并改善 regret；
- 明确的 profile、seed、候选域、噪声、split 和 truth-isolation 记录。

正向结果的主要判据：SAST-BO 相对 COMMON_POLICY 的主体等权 `Delta_i` 中位数 ≥ 0.5%，bootstrap 95% CI 不跨 0；约束违反率不增加；gate 在 null cohort 的假阳性率和 positive cohort 的检出率均报告。

### Stage 5 — development / held-out confirmation（10 月 21–31 日）

锁定算法、超参数、候选域和 evaluator 后，才生成新的 confirmatory development/held-out split。development 只用于冻结参数；held-out 只做一次盲评。完成主表、消融、失败案例、scope audit、复现实验命令和图表清单。

## 4. 必须报告的主结果

1. `COMMON_POLICY` vs `SAST-BO` 的 paired subject-level `Delta_i`、median、mean、95% CI；
2. 最终真实 E3、E2、peak、可行率和约束违反率；
3. common regret、personalized regret、oracle regret（oracle 只在 evaluator）；
4. gate trigger rate、null false-positive rate、positive detection power；
5. 灰箱 held-out 预测误差和 `MODEL_MISMATCH` 比例；
6. pool-vs-full decomposition，区分探索空间不足、模型失配和任务本身没有个体差异；
7. 失败/回退/安全锁定次数，以及每个主体的最终 policy 状态。

禁止把以下内容写成结论：MyoLeg V1 证明患者需要不同轨迹；虚拟 subject ID 代表患者差异；模拟 torque 等同 cuff force、舒适度、安全性或疗效；SAST-BO 在没有 positive-control 的情况下已经证明个性化有效。

## 5. 立即执行命令

先完成审计，不启动 30-seed：

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.personalization_audit `
  --subjects development `
  --families BETA_TIMING KEY_POSTURE_TIMING `
  --executed-results outputs\myoleg_benchmark_v1\development_20260923_r2 `
  --cache-dir .cache\myoleg-benchmark-v1 `
  --output-dir outputs\myoleg_personalization_audit_v1
```

完成 Stage 1–2 后再实现 SAST-BO；不要在 full-domain truth 生成前修改 gate 阈值。所有结果写入版本化目录，V1 输出目录不覆盖。

## 6. 论文进度

10 月 10 日前应得到灰箱诊断和 V1 null-control；10 月 20 日前完成 null/positive stress test；10 月 31 日前锁定 confirmatory 结果、图表和 limitations。11 月 1 日开始写方法、实验和结果，论文定位为“evidence-gated, mechanics-constrained simulation personalization”；真实测量个性化作为后续验证依赖，不在本轮冒充已完成证据。

Latest rerun: `outputs/myoleg_personalization_audit_v1/REPORT.md`; the prior first run is preserved under `.cache/myoleg_personalization_audit_v1_initial`.
