# MyoLeg 个性化实验执行计划 V2

更新：2026-09-23。实验锁定截止：2026-10-31。论文写作起点：2026-11-01。

本计划的主线是“证据门控、测量驱动的约束个性化优化”。当前已经完成 V1 全域审计和 K=1/2/4/8 灰箱预测验证；V1 的 30-seed 扩展暂停。新方法及 controlled cohort 使用独立模块和输出目录，不覆盖 V1。

## 1. 已完成的证据与判断

### V1 全域审计：任务本身缺少决策相关个体差异

[全域报告](../../outputs/myoleg_personalization_audit_v1/REPORT.md)覆盖 24 个 development 主体、两个候选族。`BETA_TIMING` 每主体 324 点，`KEY_POSTURE_TIMING` 每主体 387 点，共 17,064 条响应。BETA 的可行 oracle 为 `:288`，KEY 为 `:383`；各族均由 24/24 主体共享，对应 common regret 全部为 0。该次审计没有包含 native 主体，也没有访问 sealed 主体。

因此 V1 没有个体最优差异，不能归因于 BO 漏采样。V1 仍具有负荷幅度与部分可行性差异，但只能作为当前轨迹任务的 null-control，不能证明个性化优于公共策略。

### 候选空间诊断：数量足够，但有效主体差异不足

[候选空间诊断](../../outputs/myoleg_candidate_space_diagnostic_v1/REPORT.md)直接读取冻结的 V1 全域 landscape，并从原生 `Domain` 重建运动轨迹。BETA 有 324 条候选、81 条运动学拒绝；KEY 有 387 条候选、18 条拒绝。两个候选族的三参数网格均为满秩，但轨迹变化的 95% 方差分别只需要 8 个和 3 个主成分。更关键的是，跨主体 E3 排序 Spearman 中位数分别为 0.999800 和 0.999873，两个候选族的唯一 oracle 都是 1/24，top-10 集合也完全一致。固定候选的跨主体 E3 标准差只约为主体内候选标准差的 2.8–2.9%。

因此当前证据不支持“单纯候选数量太少”这一解释。V1 的候选参数有几何覆盖，但有效响应空间主要表现为共享排序；下一版应预先声明能改变主体—轨迹耦合的因素（例如辅助时机、速度/持续时间或髋膝协同），再比较是否产生不同 oracle。不得为了制造异质性事后扩大 V1 边界。

### V2 机制筛查：速度与协调变化有信号，但仍不足以作为主实验

已建立独立的 `DURATION_COORDINATION_V2` 机制域，包含 27 条候选：总周期缩放 `0.8/1.0/1.2`、平滑膝协调扰动时机 `-0.12/0/0.12`、协调幅度 `-0.04/0/0.04`。它不改 V1 域，所有候选均通过预先声明的速度、加速度和 ROM 限制。

在 24 个 development 主体上的 [native screen](../../outputs/myoleg_mechanism_screen_development_v1/REPORT.md) 中，27 条候选全部有效；约束 oracle 只有 2/24 个，跨主体 E3 排序 Spearman 中位数为 1.000000、最低为 0.985294，参考方案相对 regret 中位数为 0.000390。相对于 V1 的 1/24 oracle，这说明速度/持续时间与协调候选在原生 prescribed-state torque 响应中产生了少量额外排序变化，但主体差异仍很弱；该 oracle 只用于完整 landscape 之后的诊断，不能当作个性化推荐或收益证据，也不能直接进入最终个性化主实验。

原生 prescribed-state MyoLeg 接口没有辅助力输入，也没有把负荷分配作为可控制变量。因此“辅助时机”和“负荷分配”必须在新的明确 actuation/interaction 模型中定义，或仅在 controlled synthetic positive cohort 中做机制压力测试；不能把它们写成当前 native torque 仿真已经验证的因素。

### 灰箱 K=1/2/4/8 验证：排序可用不等于负荷预测可信

[验证报告](../../outputs/myoleg_graybox_cv_development_v1/REPORT.md)、[逐前缀汇总](../../outputs/myoleg_graybox_cv_development_v1/summary.csv)与 [protocol](../../outputs/myoleg_graybox_cv_development_v1/protocol.json)来自已冻结 V1 的执行记录：24 个 development 主体 × 两个候选族 × 三种物理方法（`PHYSICS_GREEDY`、`RESIDUAL_GP_GREEDY`、`MODEL_INFORMED_BO_EI`）× 四个预算。只使用 seed 0、无测量噪声；不包含 native、其他方法、额外种子或确认集。

每组只用当时 `trial <= K` 的已执行轨迹拟合五参数模型，随后在其余候选上评分。共完成 576 个拟合、202,608 条未见候选预测，0 个跳过前缀。这里 held-out 指“当前主体尚未用于拟合的轨迹”，并非未见主体或 sealed cohort。

每个 K 的 144 条“主体 × 候选族 × 方法”记录的中位数如下；torque RMSE 和 endpoint MAE 先在各记录的未见候选内汇总。这些记录存在同主体重复，不能当作 144 名独立主体计算置信区间。

| K | 未见轨迹 torque RMSE (Nm) | E3 MAE | E2 MAE | peak-ratio MAE | E3 Spearman | top regret 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3.698 | 0.1601 | 0.0477 | 0.0304 | 0.8502 | 0 |
| 2 | 3.704 | 0.1675 | 0.0504 | 0.0304 | 0.9631 | 0 |
| 4 | 3.711 | 0.1704 | 0.0513 | 0.0304 | 0.9678 | 0 |
| 8 | 3.713 | 0.1709 | 0.0513 | 0.0304 | 0.9637 | 0 |

576/576 次优化报告成功，但全部命中 `mass_scale` 上界；其中 102 次还命中髋阻尼边界。汇总中的标记全部为 `BOUNDARY_LIMITED`。所有预测 top-1 在本次无噪声 truth 下可行；top regret 的全表最大值约为 0.000603，其余预算最大值为 0。

增加观测到 K=8 没有消除约 3–5 Nm 的 torque 失配，也没有改善绝对 endpoint 校准。原计划 E3 MAE ≤ 0.005 的决策门槛明显未满足。因此保持五参数边界，灰箱仅保留为可被观测修正的均值先验；不能直接以其 E2/peak 均值作安全证明，更不能通过放宽上界制造个体差异。R²/VAF、约束 precision/recall、top-5 overlap 尚未由这批产物报告，不能写成已验证。

## 2. 三层实验作用域

| 实验层 | 数据和目的 | 可以支持的结论 | 不能支持的结论 |
|---|---|---|---|
| 原生 MyoLeg V1 | native/development 的 prescribed-state 逆动力学；V1 全域审计和灰箱验证 | 当前任务的共同最优、模型失配、无个体化信号时的回退表现 | 患者个体差异、舒适度、疗效、真实硬件安全 |
| Controlled synthetic | 在 native 响应外加预声明的候选空间响应场；`COMMON_OPTIMUM` 与 `DIVERGENT_OPTIMA` | 门控、观测隔离、回退、regret 改善的软件机制验证 | 该响应场等价于肌肉/患者生理差异 |
| 后续 MyoLeg 物理验证与真实测量 | 独立声明的物理扰动/任务条件；真实测量另有采集与放行流程 | 只有通过验证的对应层结论 | 用 synthetic 成功替代原生动力学或实测证据 |

Controlled cohort 的 [manifest](../../external_simulation_audits/myoleg_controlled_positive_cohort_v1/MYOLEG_CONTROLLED_POSITIVE_COHORT_V1_MANIFEST.json)已单独定义 profile、seed、split 和响应规则，不改 MuJoCo XML 或 V1 虚拟主体 delta。24 个 execution ID 实际复用三个确定性响应场（common、A、B）；`CONFIRMATORY` 是冻结协议下的保留重复执行集，不是独立主体或新 landscape 泛化集。`DIVERGENT_OPTIMA` 的 policy-level 结果已保存于 [controlled policy output](../../outputs/myoleg_controlled_positive_policy_v2/REPORT.md)：null 门控 0%，A/B 门控 100%，平均 regret 从 0.256912 降至 0.069061（73.1%）。这些数字只证明合成分析压力测试中的机制行为，不是 native MyoLeg 生理证据；native null 与后续物理验证仍须单独执行。

## 3. 已生效的执行约束

- 冻结 `outputs/myoleg_benchmark_v1/development_20260923_r2`；暂停原计划 V1 30-seed 扩展。已有 V1 结果继续作为基线，不因新算法而重写。
- 不扩大 V1 `mass_scale` 边界、不修改 V1 候选域来追求不同 oracle。
- learner 仅接收候选几何与当前主体已执行观测，不读取 profile 真值、缓存目录、完整 landscape 或 oracle。
- 公共均值模型和 COMMON_POLICY 必须在评估前冻结，并且不能用待评估主体的完整候选真值训练；其来源和选择规则写入 protocol。
- 所有方法使用相同最大候选预算和成对噪声 seed；回退提前停止时报告实际使用预算，不把未执行试验算作观测。
- `CONFIRMATORY` 保持封存。先冻结算法、门槛、候选域、分析和判据，再执行一次确认；现有 V1 sealed 主体不随开发运行自动解封。
- 已知 development 结果可以指导开发，但不能把开发后选出的门槛追溯称为预注册。确认实验的冻结时间和源代码哈希须先于确认集访问。

## 4. 算法实现与验证顺序

新方法使用 `SAST-BO`，`EG-CPI-BO` 作为描述性别名；入口为 [personalized_v2.py](../../lower_limb_sim/myoleg_benchmark/personalized_v2.py)。当前实现仍是研究原型，完整实验与置信度校准决定能否作为论文最终方法。

每主体执行顺序为 reference → 预先固定的校准 probe → 主体 residual model → 门控决策。比较对象是冻结公共响应后的主体残差，避免把所有主体共有的轨迹效应当作个性化证据。证据不足时记录 `PERSONALIZATION_INACTIVE` 并使用 COMMON_POLICY；证据足够时使用 residual GP 和 constrained acquisition。无符合预测约束的候选时记录 `SAFETY_LOCK`，回退已观测且可行的公共/参考候选。

优先完成以下验证，再扩大样本：

1. Causal prefix、未来观测/真值隔离、参考优先、probe 固定顺序，以及无噪声与带噪声回退行为。
2. null cohort 的门控假阳性率与 positive cohort 的检出率；名为 posterior/probability 的量必须说明统计模型及校准情况，不能直接把内部评分解释为经验证的 95% 保证。
3. 在同预算下比较 COMMON_POLICY、无门控 residual BO、SAST-BO、Random/SpaceFill；物理方法仅在可比较的原生 MyoLeg 接口下追加。
4. 约束模型使用 E2/peak 观测与不确定性，报告预测覆盖率、实际违反率和回退次数；灰箱边界命中不能被“优化成功”掩盖。
5. 在原生 MyoLeg V1 重新运行新策略的 null 验证，确认 synthetic 适配过程没有将共同轨迹效应误识别为个体交互。该新策略运行写入新目录，不扩展或改写旧 V1。

## 5. 到 10 月 31 日的交付里程碑

| 截止日期 | 工作与交付物 | 进入下一阶段的条件 |
|---|---|---|
| 9 月 23–25 日 | 保存灰箱结果、完成原型单元回归和 controlled development smoke；记录首版源代码与 manifest 哈希 | 无未来观测/真值泄漏；失败、预算与回退均可追溯 |
| 9 月 26–30 日 | Controlled null/positive 全域审计；共同策略基线冻结；K=4/8 的成对 pilot，保存逐轮 gate、约束和推荐日志 | positive 确有可识别 oracle 差异；无差异则如实记录，另立新版本方案而不回填旧结果 |
| 10 月 1–7 日 | 门控阈值与不确定性校准；原生 MyoLeg V1 新策略 null 验证；确定主预算和可复现实验命令 | null 假阳性和约束违反有完整报告；不能以低 regret 隐藏不可信负荷预测 |
| 10 月 8–14 日 | 锁定主算法与 development 结果；消融门控、公共先验、约束模型及校准探针 | 主方法与等预算基线的 subject-level 配对差异、CI 和失败案例齐全 |
| 10 月 15–20 日 | 冻结确认实验 protocol、代码、门槛和图表脚本；条件允许时完成独立 MyoLeg 物理扰动验证 | 任何物理验证均有单独完整性和作用域记录；未完成则论文限定 synthetic/原生 null 的证据范围 |
| 10 月 21–27 日 | 执行冻结后的新确认集一次评估；输出完整性、truth-isolation 和配对统计报告 | 不在查看确认结果后改算法；必要修订另开实验版本 |
| 10 月 28–31 日 | 仅做复现核对、主表/图/局限整理；冻结论文实验包和环境锁 | 新环境可运行核心命令；所有主张可对应到版本化产物 |
| 11 月 1 日起 | 开始完整论文初稿：方法、实验、结果、局限与相关工作 | 优先选择与实际证据匹配的知名机器人/康复工程期刊或会议；投稿档期另行核实 |

10 月 14 日设置研究 go/no-go：以 `Delta_i = loss(COMMON_POLICY)_i - loss(SAST_BO)_i` 为正向收益，报告主体等权 mean/median 和按主体重采样的 95% CI；若目标为中位收益 ≥ 0.5%、CI 不跨 0、约束违反率不增加，须在确认前冻结这一判据。不能把种子或同一主体的重复记录当成独立主体。

若 positive-control 没有收益，先检查共同策略、gate 功效、可识别性和预算消耗；不宣称个性化成功，不把 30-seed V1 作为补救。若 synthetic 成功而原生物理验证未完成，论文主张限于受控算法验证，研究计划保留后续真实测量验证。

## 6. 复现实验入口

以下重跑使用新的输出目录，避免覆盖已经完成的结果：

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.graybox_cv `
  --subjects development `
  --families BETA_TIMING KEY_POSTURE_TIMING `
  --methods PHYSICS_GREEDY RESIDUAL_GP_GREEDY MODEL_INFORMED_BO_EI `
  --seeds 0 --noise-levels 0 --budgets 1 2 4 8 `
  --executed-results outputs\myoleg_benchmark_v1\development_20260923_r2 `
  --output-dir outputs\myoleg_graybox_cv_development_reproduction
```

全域审计入口为 [personalization_audit.py](../../lower_limb_sim/myoleg_benchmark/personalization_audit.py)，controlled cohort 的语义与封存规则见 [cohort README](../../external_simulation_audits/myoleg_controlled_positive_cohort_v1/README.md)。先检查已有产物与完整性，再决定是否重跑；本计划不要求重复已有 17,064 条无噪声 landscape。
