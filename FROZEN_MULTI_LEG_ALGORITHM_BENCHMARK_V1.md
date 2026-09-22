# Frozen Multi-Leg Algorithm Benchmark V1

FROZEN_MULTI_LEG_ALGORITHM_BENCHMARK_V1 = COMPLETE。主结果为 E2；E0 与 LCB 均为预先声明的 sensitivity。此实验评估固定低预算下优化器行为，不评估临床效果、舒适性或真实患者个性化必要性。没有按结果改模型、改算法或选择 endpoint。

## Question and frozen configuration

在相同冻结机械环境、625 候选域与完整轨迹预算下，固定算法的 best measured 输出如何变化？它不要求每条腿具有不同最优轨迹。

| 报告方法 | 实际 API method | 辨识/模型 |
|---|---|---|
| REFERENCE | Reference | 单次参考，不拟合模型 |
| RANDOM | Random | 无模型；预定 seeds 0–19 |
| ADAPTIVE_GREEDY | MODEL_ONLY_GREEDY | 共享时序五参数 adapter，argmin gray-box mean |
| PURE_BO_EI | PURE_BO_EI | scalar-only Standard GP + EI |
| MODEL_INFORMED_BO_EI | MODEL_INFORMED_BO_EI_TIMESERIES_ID | 同一时序 adapter + residual GP + EI |
| MODEL_INFORMED_BO_LCB | MODEL_INFORMED_BO_LCB | 同一时序 adapter + residual GP + LCB；仅敏感性 |

Estimator、五参数含义/初值/边界、GP 核与数值规则全部复用原代码。固定 Matérn-5/2，normalized-beta length_scale=0.7、signal_std=0.6；EI xi=0，LCB kappa=1.5。没有 hyperparameter optimization。没有 trust/failover、active diagnostic、K5、PINN。预定配置在运行前写入 `results/protocol.json`。

## Frozen environments and ROM

模型源：`lower_limb_sim/five_leg_mujoco_v1/FROZEN_FIVE_LEG_MECHANICAL_PARAMETERS_V1.json`。全部使用原 `make_mujoco_model()` 与 `build_leg_domain()`；没有新增 dummy leg。下面按 thigh/shank、hip/knee 顺序列值。完整 COM、neutral angle、质量、惯量、刚度、阻尼与 coupling 记录在 `leg_parameters.csv`。

| LEG_ID | mass kg | inertia kg·m² | linear stiffness N·m/rad | damping N·m·s/rad | cubic stiffness | coupling stiffness / ratio | hip ROM deg | knee ROM deg |
|---|---|---|---|---|---|---|---|---|
| LEG_0_NOMINAL | 7/4 | 0.12/0.06 | 15/12 | 2/1.5 | 0/0 | 0/0 | 29–112 | 18.5–119.5 |
| LEG_1_HEAVY_HIP_STIFF | 10.5/3.4 | 0.24/0.055 | 32/7 | 4.5/1 | 28/4 | 3/0.5 | 35–105 | 20–110 |
| LEG_2_KNEE_DOMINANT | 6.2/5.4 | 0.095/0.14 | 6/34 | 1/4.8 | 2/32 | 6/1.4 | 22–115 | 32–132 |
| LEG_3_NONLINEAR_COUPLED | 7.4/3.8 | 0.13/0.08 | 9/10 | 1.8/1.8 | 45/55 | 18/0.75 | 30–118 | 20–128 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | 9.2/2.6 | 0.2/0.035 | 4/22 | 0.6/5.5 | 90/8 | 28/-0.65 | 15–96 | 10–102 |

原几何：L1=0.42 m，anatomical shank length=0.4 m，cuff-equivalent L2=0.3 m。保持 theta_shank=q_hip−q_knee；L2 不是脚踝长度。所有候选通过原冻结域验证；本次执行 invalid_trials=0。

## Endpoint and observation semantics

主 endpoint=`E2_BRANCH_BALANCED_REFERENCE_NORMALIZED_RMS`，直接调用共享 `branch_rms_components()`、`BranchRMSReference`、`branch_balanced_reference_normalized_rms()`。每个独立环境首先完整执行自身 reference `(0,0)`，四个归一化分母取同腿该次参考；不跨腿/跨 cohort 归一化。E0 仅为 shared `full_cycle_dual_joint_rms()` sensitivity。

每次 `evaluate()` 都只回放被请求的一条完整 V3 轨迹（401 原始样本，24 s），使用原 MuJoCo required-drive torque。没有通过 landscape 查值来生成 observation，也没有为了匹配预期改扭矩。时序辨识 payload 的 F 是通过原项目 Jacobian 的 `J.T @ F = tau` 求得的**代数等效平面力**，不是机器人传感器、柔性袖带接触或真实患者力数据。零新增噪声、endpoint uncertainty=0，为确定性软件 stress-test。

三个物理方法从同一 baseline template 与已执行有效时序数据重估五个**有效**参数。每轮使用全部有效过去 episode；当前 theta 下重新预测过去执行点、重新计算 residual target 后拟合 residual GP。Pure BO 不接收 identification payload、theta 或 physics prediction；Random/Reference 对选择层仅提供标量通道，但 environment 保留完整采集结果。所有方法无未来/held-out/oracle 数据作为选择输入。

诊断保存于 `identification_and_selection_diagnostics.json`：每轮 theta、valid episode/sample count、optimizer success/message、residual statistics、unscaled torque residual L2 norm、conditioning、covariance（原 estimator 提供时）、GP 状态与下一步 acquisition metadata。L2 norm 明确定义为 torque_RMSE_combined × sqrt(2×valid_samples)，不是物理参数可辨识性证明。拟合成功不等于 structural identifiability；代理模型与部分机械腿存在结构和量级失配。

## Budget and evaluation boundary

K_total=[1,2,3,4]，含 Trial 0 reference；K_additional=[0,1,2,3]。ROM determination 不占此预算。Greedy/Pure EI/Model-Informed EI/LCB 对每个 K_total 单独运行，检查较长运行与较短运行执行前缀完全一致。Random 每 seed 执行 K_total=4，取其因果前缀评估四个预算；每个预算恰有 K_total 个实际完整 observation，无重复候选。

Reference baseline 仅执行一次，在四个预算位置延伸显示相同值；表中 `actual_full_trajectory_evaluations=1`，没有把它伪装成同成本 adaptive run。除该明确基线例外，各方法的有效/无效试验均按完整轨迹占用相同预算。独立预算运行用于验证，不把不同运行的数据合并给优化器。

主要最终输出始终是 best VALID measured executed candidate，平局按最早实际试验。Oracle 只由 runner 外的 evaluator 在 sequential run 返回后读取冻结 625 行真值，并按原 candidate_index 打破 oracle 平局；选择器/GP/辨识器均不接收 evaluator。model-recommended candidate 只列为次要诊断，其 truth regret 由 evaluator 计算；Random/Reference 没有 model recommendation，留空。

## Optimization headroom

绝对 regret=truth(selected measured best)−truth(oracle)。normalized regret=absolute regret/(reference−oracle)，**仅在分母大于零时定义**；分母为零时留空并标 NO_OPTIMIZATION_HEADROOM。没有 epsilon 伪造分母。预先把 relative headroom <1% 标 LOW_OPTIMIZATION_HEADROOM，这是描述性标签，不改 endpoint 或选择规则。

| Leg | E2 reference | E2 oracle | Oracle beta | Absolute headroom | Relative headroom % | Status |
|---|---:|---:|---|---:|---:|---|
| LEG_0_NOMINAL | 1.000000000000 | 1.000000000000 | [0.0,0.0] | 0.000000000000 | 0.000000000 | REFERENCE_ALREADY_ORACLE |
| LEG_1_HEAVY_HIP_STIFF | 1.000000000000 | 1.000000000000 | [0.0,0.0] | 0.000000000000 | 0.000000000 | REFERENCE_ALREADY_ORACLE |
| LEG_2_KNEE_DOMINANT | 1.000000000000 | 1.000000000000 | [0.0,0.0] | 0.000000000000 | 0.000000000 | REFERENCE_ALREADY_ORACLE |
| LEG_3_NONLINEAR_COUPLED | 1.000000000000 | 1.000000000000 | [0.0,0.0] | 0.000000000000 | 0.000000000 | REFERENCE_ALREADY_ORACLE |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | 1.000000000000 | 0.999293658900 | [-0.03,0.0225] | 0.000706341100 | 0.070684037 | LOW_OPTIMIZATION_HEADROOM |

Leg 0–3 参考已经是 oracle，没有优化余量。Leg 4 最大可获得绝对改善仅 0.000706341100；相对 oracle 的 headroom 约 0.070684%。共同参考的五腿平均 relative regret 仍为 0.014137%。

## Primary E2 results

完整 `main_results.csv` 有 1000 行：每个 endpoint × leg × method/seed × budget，含用户要求的全部列及真实评估次数。以下为 K_additional=3 的确定性方法；Random 每 seed 另行保存，见后文统计。

| Leg | Method | Selected measured beta | Best measured E2 | Regret | Best found additional trial |
|---|---|---|---:|---:|---:|
| LEG_0_NOMINAL | REFERENCE | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_0_NOMINAL | ADAPTIVE_GREEDY | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_0_NOMINAL | PURE_BO_EI | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_0_NOMINAL | MODEL_INFORMED_BO_EI | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_0_NOMINAL | MODEL_INFORMED_BO_LCB | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_1_HEAVY_HIP_STIFF | REFERENCE | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_1_HEAVY_HIP_STIFF | ADAPTIVE_GREEDY | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_1_HEAVY_HIP_STIFF | PURE_BO_EI | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_1_HEAVY_HIP_STIFF | MODEL_INFORMED_BO_EI | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_1_HEAVY_HIP_STIFF | MODEL_INFORMED_BO_LCB | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_2_KNEE_DOMINANT | REFERENCE | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_2_KNEE_DOMINANT | ADAPTIVE_GREEDY | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_2_KNEE_DOMINANT | PURE_BO_EI | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_2_KNEE_DOMINANT | MODEL_INFORMED_BO_EI | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_2_KNEE_DOMINANT | MODEL_INFORMED_BO_LCB | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_3_NONLINEAR_COUPLED | REFERENCE | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_3_NONLINEAR_COUPLED | ADAPTIVE_GREEDY | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_3_NONLINEAR_COUPLED | PURE_BO_EI | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_3_NONLINEAR_COUPLED | MODEL_INFORMED_BO_EI | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_3_NONLINEAR_COUPLED | MODEL_INFORMED_BO_LCB | [0.0,0.0] | 1.000000000000 | 0.000000000000 | 0 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | REFERENCE | [0.0,0.0] | 1.000000000000 | 0.000706341100 | 0 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | ADAPTIVE_GREEDY | [0.0,0.0] | 1.000000000000 | 0.000706341100 | 0 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | PURE_BO_EI | [-0.03,0.03] | 0.999293658900 | 0.000000000000 | 3 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | MODEL_INFORMED_BO_EI | [-0.03,0.03] | 0.999293658900 | 0.000000000000 | 1 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | MODEL_INFORMED_BO_LCB | [-0.03,0.03] | 0.999293658900 | 0.000000000000 | 1 |

Leg 4 存在 oracle 平台：冻结表的确定性代表是 (-0.03,0.0225)，实际观察到的 (-0.03,0.03) 具有完全相同的 E2=0.9992936589002313。达到等值候选的 regret 为零，不表示篡改 oracle beta。

跨腿先在每腿内平均 Random seeds，再等权平均五腿；不让 20 个 Random seed 增加其腿权重。E2 中位数始终为 0，主要因为五腿中四腿无优化余量，不是算法等价证明。

| Method | Mean regret K_add=0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| REFERENCE | 0.000141268220 | 0.000141268220 | 0.000141268220 | 0.000141268220 |
| RANDOM | 0.000141268220 | 0.000123372818 | 0.000116906432 | 0.000099570017 |
| ADAPTIVE_GREEDY | 0.000141268220 | 0.000141268220 | 0.000141268220 | 0.000141268220 |
| PURE_BO_EI | 0.000141268220 | 0.000141268220 | 0.000141268220 | 0.000000000000 |
| MODEL_INFORMED_BO_EI | 0.000141268220 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| MODEL_INFORMED_BO_LCB | 0.000141268220 | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Random 预定 seeds 0–19：Leg 4 在 K_add=3 的平均 regret=0.000497850083，中位数=0.000597506073，标准差=0.000231510971，范围 [0.000122493740, 0.000706341100]。12/20 seeds 比参考改善，0/20 达到 oracle。其余四腿所有 seeds 的 best-measured regret 均为 0。`random_summary.csv` 另含每预算 p05/p95；这里只描述有限固定 seeds，不作总体统计显著性宣称。

## Reference-only physics prior quality

在所有 sequential runs 完成后，仅重建 Trial 0 reference 的初始拟合，预测 625 候选并由 evaluator 与真值比较。这个分析复现第一轮自适应前可用的 prior，但真值质量指标从未传回算法。NRMSE 分母是各腿自身 frozen landscape 的 max−min；因此它不同于 headroom normalized regret。

| Leg | E2 RMSE | NRMSE / landscape range | Spearman | Predicted best beta | Regret of predicted best |
|---|---:|---:|---:|---|---:|
| LEG_0_NOMINAL | 3.10819304e-15 | 1.94537996e-13 | 1.000000000 | [0.0,0.0] | 0.000000000000 |
| LEG_1_HEAVY_HIP_STIFF | 0.00094037551 | 0.0644586342 | 0.991389768 | [0.0,0.0] | 0.000000000000 |
| LEG_2_KNEE_DOMINANT | 0.00243714432 | 0.171535682 | 0.909049889 | [0.0,0.0] | 0.000000000000 |
| LEG_3_NONLINEAR_COUPLED | 0.00129137598 | 0.0952606143 | 0.984786210 | [0.0,0.0] | 0.000000000000 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | 0.00131650003 | 0.117469156 | 0.967650213 | [0.0,0.0] | 0.000706341100 |

五条腿的初始 E2 prior 都预测 reference 最好，包括 Leg 4；因此即使 Leg 4 Spearman 较高，也不能推出 predicted minimizer 正确。所有 theta 仍为 effective gray-box 参数，不等价于真实材料/生理参数。

## Comparison questions A–F

**A — EI 是否比 Adaptive Greedy 更快？** E2 的 Leg 4 是：Model-Informed EI 在第 1 次追加试验达到 oracle 等值候选，Greedy 至第 3 次仍选出的 best 是 reference。Leg 0–3 无优化余量，不能据这些平局支持探索优越性。E0 sensitivity 中 Greedy 已与 Model-Informed EI 同样在第 1 次达到全部五腿 oracle。

**B — 是否比 Pure BO 更快？** E2 Leg 4：第 1 次 vs 第 3 次；最终 K_add=3 两者 regret 均为零。E0 sensitivity：Model-Informed EI 全部五腿第 1 次达到 oracle，Pure EI 为第 2 或第 3 次。因此这里体现预算内速度优势，而非最终更低的可达 optimum。

**C — Pure BO 是否优于 Random？** 在 K_add=3 的 best-measured 最终 regret 上，Pure EI 在 E2 Leg 4 优于 20-seed Random 均值，E0 五腿最终也为零。不能概括为全预算优于：K_add=1、2 的 E2 Leg 4 Pure EI 尚无改善，而 Random 已有部分改善；E0 的早期跨腿均值也存在同样情况。

**D — physics prior 在哪帮助？** 与 Pure EI 比，E2 仅能在有非零余量的 Leg 4 观察到更早改善；E0 sensitivity 的五腿均观察到更早到达 oracle。

**E — physics prior 在哪伤害？** 本次 0–3 追加预算的 measured-best 指标中未观察到 Model-Informed EI 比 Pure EI 更差。不能据此宣称 prior 始终可靠：Leg 4 的初始 predicted minimizer 错误，纯 Greedy 留在低余量局部邻域；E0 部分腿具有很大的绝对 prediction RMSE，但排序仍相对一致。

**F — 平局原因？** E2 Leg 0–3 是 reference-already-oracle，不能区分算法质量；E2 Leg 4 在 K_add=3 的 EI/Pure/LCB 平局是已观察到同一 oracle 平台，达到时间不同。E0 的最终平局同样不等于运行轨迹或达到速度相同。

## E0 and LCB sensitivity

E0 SENSITIVITY ONLY：shared E0 已兼容，因此按相同预定方法、预算与 seeds 运行，未据胜负选择 endpoint。E0 的 Greedy、Model-Informed EI 和 LCB 全部五腿第 1 次追加试验达到 oracle；Pure EI 的到达次数为 Leg 0–4：[2,3,3,3,2]。各腿的 E0 reference、oracle、headroom、先验误差与所有 seed 明细在相同 CSV 的 endpoint=E0 行。主结论始终基于 E2。

LCB SENSITIVITY ONLY：冻结 kappa=1.5、相同 adapter/GP/endpoint/domain/budget/初始观察。独立闭环 rollout 在选择分歧后允许产生不同历史；没有强行把不同选择的数据当成相同。为隔离 acquisition 本身，另在已记录 EI 的 30 个历史状态上用同一 fitted posterior 同时计算 EI 与 LCB，**不执行新候选**。`matched_history_ei_lcb.csv` 保存这项只读诊断；28/30 状态选择相同，另外两项如下。

| Leg | Endpoint | History count | EI beta | LCB beta |
|---|---|---:|---|---|
| LEG_0_NOMINAL | E2 | 3 | [-0.03,-0.0275] | [-0.03,-0.03] |
| LEG_0_NOMINAL | E0 | 3 | [0.0275,0.03] | [0.03,0.03] |

本次 EI/LCB 的所有预算 best-measured regret 相同；这只是当前固定问题下的 sensitivity 结果，不说明 acquisition 公式等价。

## Figures and video-ready data

`results/figures/` 含 11 张 PNG：E2 主实验与 E0 sensitivity 各自的 Figure 1（跨腿 mean/median）、Figure 1b（每腿曲线）、Figure 2（最终 regret）、Figure 3（初始化 prior rank quality vs EI benefit）、Figure 4（每腿执行 beta 路径）；另有 Figure 5（EI/LCB sensitivity）。Figure 4 的 Random 只显示预先约定 seed 0，统计仍使用全部 20 seeds；未挑选表现更好的路径。

`trial_history.csv` 共 970 行，可不重跑优化器直接驱动动画，包含 leg/method/endpoint/seed/trial_index、beta、candidate/ROM identity、measured endpoint、明确 evaluator-only 的 truth、best measured value/beta、theta、所选点的预测 mean/std、acquisition type/value、validity。Trial 0 是 reference；其 pre-selection prediction/acquisition 不存在，留空而不造数。`theta_hat` 是该 trial 观察后的拟合；predicted mean/std 与 acquisition 来自该 trial 选择**之前**的上一轮 metadata，两者时间语义不同。纯标量方法 theta 留空。

`main_results.csv` 保存每预算的独立结果和 model-recommended 次要诊断；`headroom.csv`、`prior_quality.csv`、`random_summary.csv`、`cross_leg_summary.csv`、`e2_positive_headroom_normalized_summary.csv` 与完整辨识诊断供论文/视频引用。没有生成视频或继续扩展优化器。

## Validation and limitations

49 项直接相关测试通过，包括 4 项新 benchmark 测试及既有时序 EI/共享 E2 测试。覆盖统一域/endpoint/预算、oracle 文件访问阻断、无重复候选、不同未来预算不改变已有选择、每腿 reference normalization、scalar-only 隔离、measured-best 主语义和逐预算导出一致性。所有实际观测 endpoint 与 evaluator 读到的冻结真值在浮点容差内一致；运行失败=0，invalid trials=0。

限制：五个独立冻结机械 stress-test 模型而非真实患者总体；只有一条 E2 腿有非零且很小的余量。确定性环境与 20 个 Random seeds 不等同于多患者、多噪声、真机研究。辨识来自等效力变换，未模拟真实袖带测力。先验质量诊断不授权修改模型或算法。E0 sensitivity 的排序成功不能提升它为主 endpoint。

历史 `E2_PERSONALIZATION_NECESSITY = NOT_SUPPORTED` 和 `READY_FOR_E2_FROZEN_ALGORITHM_COMPARISON=false` 均未改写。当前用户明确请求的是一个新的冻结算法 stress-test，不是对历史研究 readiness 或个性化必要性的翻案。

## Reproduction

在仓库根目录使用已配置临时 Python 环境；需要 MuJoCo、numpy/scipy、pandas、matplotlib。无实时控制、无 OpenGL 渲染需求。

```sh
/private/tmp/rehab_architecture_review_venv/bin/python -m lower_limb_sim.frozen_multi_leg_algorithm_benchmark_v1.run
/private/tmp/rehab_architecture_review_venv/bin/python -m lower_limb_sim.frozen_multi_leg_algorithm_benchmark_v1.acquisition_sensitivity
/private/tmp/rehab_architecture_review_venv/bin/python -m lower_limb_sim.frozen_multi_leg_algorithm_benchmark_v1.report
```

原 frozen landscape、E2 necessity、历史 BO 输出均未覆盖。所有新增实验文件在 `lower_limb_sim/frozen_multi_leg_algorithm_benchmark_v1/`。运行器只调用既有算法 API；绘图脚本只读取已完成结果。无自动 commit。

```text
FROZEN_MULTI_LEG_ALGORITHM_BENCHMARK_V1 = COMPLETE
PRIMARY_ENDPOINT = E2
PRIMARY_METHOD = MODEL_INFORMED_BO_EI_TIMESERIES_ID
LEGS = [LEG_0_NOMINAL, LEG_1_HEAVY_HIP_STIFF, LEG_2_KNEE_DOMINANT, LEG_3_NONLINEAR_COUPLED, LEG_4_STRONG_STRUCTURAL_MISMATCH]
K_ADDITIONAL = [0,1,2,3]
MODEL_INFORMED_BO_EI = COMPLETE
PURE_BO_EI = COMPLETE
ADAPTIVE_GREEDY = COMPLETE
RANDOM = COMPLETE
LCB_SENSITIVITY = COMPLETE
E0_SENSITIVITY = COMPLETE
PERSONALIZATION_NECESSITY_CONCLUSION_CHANGED = false
ALGORITHM_PARAMETERS_RETUNED = false
LEG_PARAMETERS_CHANGED = false
V3_CHANGED = false
E2_CHANGED = false
ROBOT_CODE_MODIFIED = false
```
