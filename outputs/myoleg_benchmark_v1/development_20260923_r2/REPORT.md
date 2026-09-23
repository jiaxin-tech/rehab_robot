# MyoLeg 开发阶段算法实验报告

本报告仅分析已完成的 `results.csv` 与 `protocol.json`，不运行模型、不选推荐、不读取oracle或held-out真值。已有3320条预算记录，对应830个方法/种子顺序运行；覆盖24/24个development主体及1个native模型。Native不计入development队列均值。未知主体分类：无。

## 主结果与经验选优

- BETA_TIMING，主预算 K=4：经验排序首位为 MODEL_INFORMED_BO_EI（真实合格率100.000%，selection_loss=0.97627135）；与 PHYSICS_GREEDY、RESIDUAL_GP_GREEDY、SPACE_FILLING 的配对区间未支持唯一优势，不能宣称唯一赢家。
- BETA_TIMING，次级预算 K=8：经验排序首位为 MODEL_INFORMED_BO_EI（真实合格率100.000%，selection_loss=0.97627135）；与 PHYSICS_GREEDY、RESIDUAL_GP_GREEDY、SPACE_FILLING 的配对区间未支持唯一优势，不能宣称唯一赢家。
- KEY_POSTURE_TIMING，主预算 K=4：经验排序首位为 MODEL_INFORMED_BO_EI（真实合格率100.000%，selection_loss=0.97027589）；与 PHYSICS_GREEDY、RESIDUAL_GP_GREEDY 的配对区间未支持唯一优势，不能宣称唯一赢家。
- KEY_POSTURE_TIMING，次级预算 K=8：经验排序首位为 MODEL_INFORMED_BO_EI（真实合格率100.000%，selection_loss=0.97027589）；与 PHYSICS_GREEDY、RESIDUAL_GP_GREEDY、SPACE_FILLING 的配对区间未支持唯一优势，不能宣称唯一赢家。

主结论固定为无噪声、总预算 K=4；K=8 是次级预算比较，不替换主结果。排序先看最终推荐真实合格率，再看selection_loss；该词典序来自运行前已固定并记录源码SHA的报告实现。

上述区间是主体配对的描述性bootstrap；与数值零区间 ±1e-12 相交时不宣称唯一优势。该容差仅处理浮点舍入，不是实际效果阈值，也不代表达到它便具有实际价值。本轮属于开发比较，不能替代冻结后的8个sealed主体确认。

| family | method | budget | subject_count | true_feasible_rate | selection_loss | improvement_pct | constraint_violations | invalid_trials | failure_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BETA_TIMING | MODEL_INFORMED_BO_EI | 4 | 24 | 1 | 0.976271 | 2.37286 | 0 | 0 | 0 |
| BETA_TIMING | MODEL_INFORMED_BO_EI | 8 | 24 | 1 | 0.976271 | 2.37286 | 0.125 | 0 | 0 |
| BETA_TIMING | PHYSICS_GREEDY | 4 | 24 | 1 | 0.976271 | 2.37286 | 0 | 0 | 0 |
| BETA_TIMING | PHYSICS_GREEDY | 8 | 24 | 1 | 0.976271 | 2.37286 | 0 | 0 | 0 |
| BETA_TIMING | PURE_BO_EI | 4 | 24 | 1 | 0.986832 | 1.31675 | 2 | 0 | 0 |
| BETA_TIMING | PURE_BO_EI | 8 | 24 | 1 | 0.978218 | 2.17818 | 2 | 0 | 0 |
| BETA_TIMING | RANDOM | 4 | 24 | 1 | 0.992499 | 0.750118 | 1.44167 | 0 | 0 |
| BETA_TIMING | RANDOM | 8 | 24 | 1 | 0.990142 | 0.985779 | 4.04167 | 0 | 0 |
| BETA_TIMING | REFERENCE | 4 | 24 | 1 | 1 | 0 | 0 | 0 | 0 |
| BETA_TIMING | REFERENCE | 8 | 24 | 1 | 1 | 0 | 0 | 0 | 0 |
| BETA_TIMING | RESIDUAL_GP_GREEDY | 4 | 24 | 1 | 0.976271 | 2.37286 | 0 | 0 | 0 |
| BETA_TIMING | RESIDUAL_GP_GREEDY | 8 | 24 | 1 | 0.976271 | 2.37286 | 0 | 0 | 0 |
| BETA_TIMING | SPACE_FILLING | 4 | 24 | 1 | 0.976271 | 2.37286 | 2 | 0 | 0 |
| BETA_TIMING | SPACE_FILLING | 8 | 24 | 1 | 0.976271 | 2.37286 | 4 | 0 | 0 |
| KEY_POSTURE_TIMING | MODEL_INFORMED_BO_EI | 4 | 24 | 1 | 0.970276 | 2.97241 | 0 | 0 | 0 |
| KEY_POSTURE_TIMING | MODEL_INFORMED_BO_EI | 8 | 24 | 1 | 0.970276 | 2.97241 | 0 | 0 | 0 |
| KEY_POSTURE_TIMING | PHYSICS_GREEDY | 4 | 24 | 1 | 0.970276 | 2.97241 | 0 | 0 | 0 |
| KEY_POSTURE_TIMING | PHYSICS_GREEDY | 8 | 24 | 1 | 0.970276 | 2.97241 | 0 | 0 | 0 |
| KEY_POSTURE_TIMING | PURE_BO_EI | 4 | 24 | 1 | 0.990554 | 0.944626 | 1 | 0 | 0 |
| KEY_POSTURE_TIMING | PURE_BO_EI | 8 | 24 | 1 | 0.973182 | 2.68184 | 1.04167 | 0 | 0 |
| KEY_POSTURE_TIMING | RANDOM | 4 | 24 | 1 | 0.986668 | 1.33323 | 1.2 | 0 | 0 |
| KEY_POSTURE_TIMING | RANDOM | 8 | 24 | 1 | 0.981933 | 1.8067 | 3.8 | 0 | 0 |
| KEY_POSTURE_TIMING | REFERENCE | 4 | 24 | 1 | 1 | 0 | 0 | 0 | 0 |
| KEY_POSTURE_TIMING | REFERENCE | 8 | 24 | 1 | 1 | 0 | 0 | 0 | 0 |
| KEY_POSTURE_TIMING | RESIDUAL_GP_GREEDY | 4 | 24 | 1 | 0.970276 | 2.97241 | 0 | 0 | 0 |
| KEY_POSTURE_TIMING | RESIDUAL_GP_GREEDY | 8 | 24 | 1 | 0.970276 | 2.97241 | 0 | 0 | 0 |
| KEY_POSTURE_TIMING | SPACE_FILLING | 4 | 24 | 1 | 1 | 0 | 3 | 0 | 0 |
| KEY_POSTURE_TIMING | SPACE_FILLING | 8 | 24 | 1 | 0.970276 | 2.97241 | 6 | 0 | 0 |

## 评分与统计口径

最终推荐由算法根据可见观测选择，报告只对这个推荐评分。完成且合格时selection_loss=true_E3；完成但不合格时为max(1,true_E3)；未完成、失败或无可评分推荐时为1。失败单列，不能用评估器在已执行集合中用真值重新挑选候选。较低损失更好，参考E3=1；improvement_pct=100×(1−selection_loss)，是应用不合格/失败惩罚后的改善，须与真实合格率一起解读。

先在每主体内对种子求均值，再在主体间等权汇总；Random的多种子不增加其主体权重。constraint_violations与invalid_trials为每次顺序运行累计次数的主体等权均值；constraint_violations是观测判定的超限次数，有噪时不等同于全部试验的真实超限次数，最终推荐的真实合格率则由独立评估器计算。K是含参考的预算上限，实际次数见subject_summary/summary中的executed_trials；Reference可能只执行一次。原始E3/E2/峰值的均值仅在有可评分推荐的记录上计算，不能替代包含失败的selection_loss。

elapsed_s为完整最长预算运行的wall time，包含模拟/缓存命中先后及模型拟合，同一运行的各预算行沿用该值；它仅供工程诊断，不能用于算法速度排名或较短预算耗时推断。

development方法差值以同主体均值配对，固定seed=20260923、2000次主体bootstrap，给出95%百分位区间。差值方向为A−B，selection_loss负值有利于A，真实合格率正值有利于A。只有一个主体时不计算主体区间。所有逐对结果在pairwise_comparisons.csv；以下列出MI-EI相关的无噪损失比较。未执行oracle全景分析，因此本报告不报告oracle regret。

| family | budget | method_a | method_b | paired_subjects | mean_a_minus_b | ci95_low | ci95_high | ci_includes_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BETA_TIMING | 4 | MODEL_INFORMED_BO_EI | PHYSICS_GREEDY | 24 | 0 | 0 | 0 | True |
| BETA_TIMING | 4 | MODEL_INFORMED_BO_EI | PURE_BO_EI | 24 | -0.0105611 | -0.0107461 | -0.0103745 | False |
| BETA_TIMING | 4 | MODEL_INFORMED_BO_EI | RANDOM | 24 | -0.0162275 | -0.0164767 | -0.0160012 | False |
| BETA_TIMING | 4 | MODEL_INFORMED_BO_EI | REFERENCE | 24 | -0.0237286 | -0.0239637 | -0.0234897 | False |
| BETA_TIMING | 4 | MODEL_INFORMED_BO_EI | RESIDUAL_GP_GREEDY | 24 | 0 | 0 | 0 | True |
| BETA_TIMING | 4 | MODEL_INFORMED_BO_EI | SPACE_FILLING | 24 | 0 | 0 | 0 | True |
| BETA_TIMING | 8 | MODEL_INFORMED_BO_EI | PHYSICS_GREEDY | 24 | 0 | 0 | 0 | True |
| BETA_TIMING | 8 | MODEL_INFORMED_BO_EI | PURE_BO_EI | 24 | -0.00194682 | -0.001976 | -0.00191759 | False |
| BETA_TIMING | 8 | MODEL_INFORMED_BO_EI | RANDOM | 24 | -0.0138709 | -0.0140997 | -0.0136635 | False |
| BETA_TIMING | 8 | MODEL_INFORMED_BO_EI | REFERENCE | 24 | -0.0237286 | -0.0239637 | -0.0234897 | False |
| BETA_TIMING | 8 | MODEL_INFORMED_BO_EI | RESIDUAL_GP_GREEDY | 24 | 0 | 0 | 0 | True |
| BETA_TIMING | 8 | MODEL_INFORMED_BO_EI | SPACE_FILLING | 24 | 0 | 0 | 0 | True |
| KEY_POSTURE_TIMING | 4 | MODEL_INFORMED_BO_EI | PHYSICS_GREEDY | 24 | 0 | 0 | 0 | True |
| KEY_POSTURE_TIMING | 4 | MODEL_INFORMED_BO_EI | PURE_BO_EI | 24 | -0.0202778 | -0.0204237 | -0.0201224 | False |
| KEY_POSTURE_TIMING | 4 | MODEL_INFORMED_BO_EI | RANDOM | 24 | -0.0163918 | -0.0164996 | -0.0162767 | False |
| KEY_POSTURE_TIMING | 4 | MODEL_INFORMED_BO_EI | REFERENCE | 24 | -0.0297241 | -0.0299615 | -0.029478 | False |
| KEY_POSTURE_TIMING | 4 | MODEL_INFORMED_BO_EI | RESIDUAL_GP_GREEDY | 24 | 0 | 0 | 0 | True |
| KEY_POSTURE_TIMING | 4 | MODEL_INFORMED_BO_EI | SPACE_FILLING | 24 | -0.0297241 | -0.0299615 | -0.029478 | False |
| KEY_POSTURE_TIMING | 8 | MODEL_INFORMED_BO_EI | PHYSICS_GREEDY | 24 | 0 | 0 | 0 | True |
| KEY_POSTURE_TIMING | 8 | MODEL_INFORMED_BO_EI | PURE_BO_EI | 24 | -0.00290574 | -0.00391099 | -0.0020637 | False |
| KEY_POSTURE_TIMING | 8 | MODEL_INFORMED_BO_EI | RANDOM | 24 | -0.0116571 | -0.0117197 | -0.0115895 | False |
| KEY_POSTURE_TIMING | 8 | MODEL_INFORMED_BO_EI | REFERENCE | 24 | -0.0297241 | -0.0299615 | -0.029478 | False |
| KEY_POSTURE_TIMING | 8 | MODEL_INFORMED_BO_EI | RESIDUAL_GP_GREEDY | 24 | 0 | 0 | 0 | True |
| KEY_POSTURE_TIMING | 8 | MODEL_INFORMED_BO_EI | SPACE_FILLING | 24 | 0 | 0 | 0 | True |

## Native 噪声与工程验证

| noise_std | seeds_min | seeds_max | family | method | budget | subject_count | true_feasible_rate | selection_loss | improvement_pct | constraint_violations | invalid_trials | failure_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | BETA_TIMING | MODEL_INFORMED_BO_EI | 4 | 1 | 1 | 0.976183 | 2.38167 | 0 | 0 | 0 |
| 0 | 1 | 1 | BETA_TIMING | MODEL_INFORMED_BO_EI | 8 | 1 | 1 | 0.976183 | 2.38167 | 0 | 0 | 0 |
| 0 | 1 | 1 | BETA_TIMING | PHYSICS_GREEDY | 4 | 1 | 1 | 0.976183 | 2.38167 | 0 | 0 | 0 |
| 0 | 1 | 1 | BETA_TIMING | PHYSICS_GREEDY | 8 | 1 | 1 | 0.976183 | 2.38167 | 0 | 0 | 0 |
| 0 | 1 | 1 | BETA_TIMING | PURE_BO_EI | 4 | 1 | 1 | 0.9868 | 1.32001 | 2 | 0 | 0 |
| 0 | 1 | 1 | BETA_TIMING | PURE_BO_EI | 8 | 1 | 1 | 0.978139 | 2.18612 | 2 | 0 | 0 |
| 0 | 5 | 5 | BETA_TIMING | RANDOM | 4 | 1 | 1 | 0.992301 | 0.769857 | 1.4 | 0 | 0 |
| 0 | 5 | 5 | BETA_TIMING | RANDOM | 8 | 1 | 1 | 0.989946 | 1.00545 | 4 | 0 | 0 |
| 0 | 1 | 1 | BETA_TIMING | REFERENCE | 4 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | BETA_TIMING | REFERENCE | 8 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | BETA_TIMING | RESIDUAL_GP_GREEDY | 4 | 1 | 1 | 0.976183 | 2.38167 | 0 | 0 | 0 |
| 0 | 1 | 1 | BETA_TIMING | RESIDUAL_GP_GREEDY | 8 | 1 | 1 | 0.976183 | 2.38167 | 0 | 0 | 0 |
| 0 | 1 | 1 | BETA_TIMING | SPACE_FILLING | 4 | 1 | 1 | 0.976183 | 2.38167 | 2 | 0 | 0 |
| 0 | 1 | 1 | BETA_TIMING | SPACE_FILLING | 8 | 1 | 1 | 0.976183 | 2.38167 | 4 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | MODEL_INFORMED_BO_EI | 4 | 1 | 1 | 0.976183 | 2.38167 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | MODEL_INFORMED_BO_EI | 8 | 1 | 1 | 0.976183 | 2.38167 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | PHYSICS_GREEDY | 4 | 1 | 1 | 0.976238 | 2.37624 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | PHYSICS_GREEDY | 8 | 1 | 1 | 0.976238 | 2.37624 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | PURE_BO_EI | 4 | 1 | 1 | 0.9868 | 1.32001 | 2 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | PURE_BO_EI | 8 | 1 | 1 | 0.978139 | 2.18612 | 2 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | RANDOM | 4 | 1 | 1 | 0.992186 | 0.781389 | 1.6 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | RANDOM | 8 | 1 | 1 | 0.986978 | 1.3022 | 3.7 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | REFERENCE | 4 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | REFERENCE | 8 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | RESIDUAL_GP_GREEDY | 4 | 1 | 1 | 0.976238 | 2.37624 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | RESIDUAL_GP_GREEDY | 8 | 1 | 1 | 0.976238 | 2.37624 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | SPACE_FILLING | 4 | 1 | 1 | 0.976183 | 2.38167 | 2 | 0 | 0 |
| 0.01 | 10 | 10 | BETA_TIMING | SPACE_FILLING | 8 | 1 | 1 | 0.976183 | 2.38167 | 4 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | MODEL_INFORMED_BO_EI | 4 | 1 | 1 | 0.976407 | 2.35927 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | MODEL_INFORMED_BO_EI | 8 | 1 | 1 | 0.976407 | 2.35927 | 0.1 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | PHYSICS_GREEDY | 4 | 1 | 1 | 0.976542 | 2.34582 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | PHYSICS_GREEDY | 8 | 1 | 1 | 0.976654 | 2.33462 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | PURE_BO_EI | 4 | 1 | 1 | 0.98944 | 1.05601 | 2.2 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | PURE_BO_EI | 8 | 1 | 1 | 0.979378 | 2.0622 | 2.2 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | RANDOM | 4 | 1 | 1 | 0.992186 | 0.781389 | 1.7 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | RANDOM | 8 | 1 | 1 | 0.987187 | 1.28128 | 3.8 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | REFERENCE | 4 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | REFERENCE | 8 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | RESIDUAL_GP_GREEDY | 4 | 1 | 1 | 0.976542 | 2.34582 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | RESIDUAL_GP_GREEDY | 8 | 1 | 1 | 0.976654 | 2.33462 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | SPACE_FILLING | 4 | 1 | 1 | 0.976183 | 2.38167 | 2 | 0 | 0 |
| 0.03 | 10 | 10 | BETA_TIMING | SPACE_FILLING | 8 | 1 | 1 | 0.976183 | 2.38167 | 4 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | MODEL_INFORMED_BO_EI | 4 | 1 | 1 | 0.970156 | 2.98444 | 0 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | MODEL_INFORMED_BO_EI | 8 | 1 | 1 | 0.970156 | 2.98444 | 0 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | PHYSICS_GREEDY | 4 | 1 | 1 | 0.970156 | 2.98444 | 0 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | PHYSICS_GREEDY | 8 | 1 | 1 | 0.970156 | 2.98444 | 0 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | PURE_BO_EI | 4 | 1 | 1 | 0.990517 | 0.948269 | 1 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | PURE_BO_EI | 8 | 1 | 1 | 0.974724 | 2.52755 | 1 | 0 | 0 |
| 0 | 5 | 5 | KEY_POSTURE_TIMING | RANDOM | 4 | 1 | 1 | 0.986613 | 1.3387 | 1.2 | 0 | 0 |
| 0 | 5 | 5 | KEY_POSTURE_TIMING | RANDOM | 8 | 1 | 1 | 0.981858 | 1.81416 | 3.8 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | REFERENCE | 4 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | REFERENCE | 8 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | RESIDUAL_GP_GREEDY | 4 | 1 | 1 | 0.970156 | 2.98444 | 0 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | RESIDUAL_GP_GREEDY | 8 | 1 | 1 | 0.970156 | 2.98444 | 0 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | SPACE_FILLING | 4 | 1 | 1 | 1 | 0 | 3 | 0 | 0 |
| 0 | 1 | 1 | KEY_POSTURE_TIMING | SPACE_FILLING | 8 | 1 | 1 | 0.970156 | 2.98444 | 6 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | MODEL_INFORMED_BO_EI | 4 | 1 | 1 | 0.970156 | 2.98444 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | MODEL_INFORMED_BO_EI | 8 | 1 | 1 | 0.970156 | 2.98444 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | PHYSICS_GREEDY | 4 | 1 | 1 | 0.970326 | 2.96742 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | PHYSICS_GREEDY | 8 | 1 | 1 | 0.970326 | 2.96742 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | PURE_BO_EI | 4 | 1 | 1 | 0.990517 | 0.948269 | 1 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | PURE_BO_EI | 8 | 1 | 1 | 0.974324 | 2.56757 | 1 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | RANDOM | 4 | 1 | 1 | 0.987905 | 1.20948 | 1.4 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | RANDOM | 8 | 1 | 1 | 0.982318 | 1.76825 | 3.9 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | REFERENCE | 4 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | REFERENCE | 8 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | RESIDUAL_GP_GREEDY | 4 | 1 | 1 | 0.970326 | 2.96742 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | RESIDUAL_GP_GREEDY | 8 | 1 | 1 | 0.970326 | 2.96742 | 0 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | SPACE_FILLING | 4 | 1 | 1 | 1 | 0 | 3 | 0 | 0 |
| 0.01 | 10 | 10 | KEY_POSTURE_TIMING | SPACE_FILLING | 8 | 1 | 1 | 0.970156 | 2.98444 | 6 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | MODEL_INFORMED_BO_EI | 4 | 1 | 1 | 0.970622 | 2.9378 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | MODEL_INFORMED_BO_EI | 8 | 1 | 1 | 0.970622 | 2.9378 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | PHYSICS_GREEDY | 4 | 1 | 1 | 0.970669 | 2.93308 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | PHYSICS_GREEDY | 8 | 1 | 1 | 0.970669 | 2.93308 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | PURE_BO_EI | 4 | 1 | 1 | 0.990517 | 0.948269 | 1 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | PURE_BO_EI | 8 | 1 | 1 | 0.973676 | 2.63243 | 1.1 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | RANDOM | 4 | 1 | 1 | 0.987905 | 1.20948 | 1.4 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | RANDOM | 8 | 1 | 1 | 0.982318 | 1.76825 | 3.9 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | REFERENCE | 4 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | REFERENCE | 8 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | RESIDUAL_GP_GREEDY | 4 | 1 | 1 | 0.970609 | 2.93906 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | RESIDUAL_GP_GREEDY | 8 | 1 | 1 | 0.970669 | 2.93308 | 0 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | SPACE_FILLING | 4 | 1 | 1 | 1 | 0 | 3 | 0 | 0 |
| 0.03 | 10 | 10 | KEY_POSTURE_TIMING | SPACE_FILLING | 8 | 1 | 1 | 0.970156 | 2.98444 | 5.9 | 0 | 0 |

Native为单一模型。噪声图的95%区间仅描述合成测量/算法重复，不是跨主体推断；无噪确定性方法只有一次时不绘制区间。种子数量以上表及protocol为准，本轮工程重复不能替代10月计划的正式30种子稳健性实验。1%/3%为参考关节RMS尺度的合成采样噪声，不代表实测传感器误差。

## 失败与限制

最终预算记录中的失败运行数：0/830。下表按明确status保留失败；约束超限次数在主表及原始逐预算结果中保留，不因回退参考而抹除。

本阶段暂无对应结果。

两种候选族、相同预算、模型/端点/负荷约束下比较，不以未知真值筛选候选。有效但观测超限的样本保留用于拟合，最终推荐只从观测合格的已执行候选中选择；超限不补预算。E3及这些负荷约束只具有当前仿真机械语义；不支持临床疗效、舒适度、患者代表性、真实机器人安全或在线个体化结论。24个development已经用于开发，不能写成保留集；8个sealed主体仍须按原政策在最终冻结及另行授权后独立评估。没有明确赢家时如实保留并列或“无优势”，不通过改变模型制造算法优势。

## 文件与本次配置

- results.csv：每主体/方法/种子/预算的原始推荐成绩。
- subject_summary.csv：先在主体内汇总种子；summary.csv：再对主体等权汇总。
- pairwise_comparisons.csv：K4/K8的主体配对损失与合格率差值、95%区间。
- convergence.png：无噪最终推荐的预算曲线；noise_robustness.png：native噪声稳健性。
- protocol.json：完整运行设置与代码来源；以下原样摘录可用的关键配置。
- analysis_provenance.json：当前报告源码、运行前报告源码、protocol、completion、results及存在时trial_history的SHA；记录仅修正数值容差、完整性检查与主次预算标签，模拟、推荐及原始结果保持不变。

```json
{
  "experiment_id": "MYOLEG_DEVELOPMENT_BENCHMARK_V1",
  "methods": [
    "REFERENCE",
    "RANDOM",
    "SPACE_FILLING",
    "PHYSICS_GREEDY",
    "RESIDUAL_GP_GREEDY",
    "PURE_BO_EI",
    "MODEL_INFORMED_BO_EI"
  ],
  "subjects": [
    "MYOLEG_NATIVE_P0",
    "MYOLEG_VP_001",
    "MYOLEG_VP_002",
    "MYOLEG_VP_003",
    "MYOLEG_VP_005",
    "MYOLEG_VP_006",
    "MYOLEG_VP_007",
    "MYOLEG_VP_009",
    "MYOLEG_VP_010",
    "MYOLEG_VP_011",
    "MYOLEG_VP_013",
    "MYOLEG_VP_014",
    "MYOLEG_VP_015",
    "MYOLEG_VP_017",
    "MYOLEG_VP_018",
    "MYOLEG_VP_019",
    "MYOLEG_VP_021",
    "MYOLEG_VP_022",
    "MYOLEG_VP_023",
    "MYOLEG_VP_025",
    "MYOLEG_VP_026",
    "MYOLEG_VP_027",
    "MYOLEG_VP_029",
    "MYOLEG_VP_030",
    "MYOLEG_VP_031"
  ],
  "families": [
    "BETA_TIMING",
    "KEY_POSTURE_TIMING"
  ],
  "noise_levels": [
    0.01,
    0.03
  ],
  "noise_subjects": "native",
  "seeds": {
    "random": [
      0,
      1,
      2,
      3,
      4
    ],
    "paired_noise": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9
    ]
  },
  "budgets": [
    1,
    2,
    4,
    8
  ],
  "primary_budget": 4,
  "primary_noise_std": 0.0,
  "constraints": {
    "E2_max": 1.01,
    "joint_peak_ratio_max": 1.1,
    "kinematic_speed_ratio_max": 1.5,
    "kinematic_acceleration_ratio_max": 2
  },
  "git_commit": "f77029b6f17e4cac1e1f87c8f27706aee4bab5f1"
}
```

复现报告：`python -m lower_limb_sim.myoleg_benchmark.report --output-dir outputs/myoleg_benchmark_v1/development_20260923_r2`。此命令只重建汇总和图表。
