# E3 单项负荷约束与低预算实验

两组完成。第一组使用已有完整候选真值；第二组在各候选族同一运动学域内，只读取已选候选的完整时序，不提前筛选真实负荷。6个模型×2个轨迹族×3档约束×24个方法/种子运行=864个独立顺序运行，结果表3456行，逐试验记录3348行。

## 主要结果

- 第一组：新样条在18个模型/阈值组合中，16个可达E3改善高于β＋时间，2个相同。
- 第二组：Model-Informed BO在新样条中的最终已执行回放的最佳改善，相比β＋时间，14个组合更好，0个更差，4个持平。因此“候选域里存在更好的轨迹”不能直接推出“3次试验一定更好”。
- Model-Informed BO EI与Adaptive Greedy最终最佳E3在36/36个组合相同。检查逐预算曲线与下表后再解释探索价值，不能用相同最终值宣称算法本身相同。
- 严格约束下可能多次试验超限而仍保持参考。超限次数是算法表现的一部分，不删除、不补预算。

## 第一组：约束敏感性

继续使用此前统一的速度≤reference 1.5倍、加速度≤2倍、髋膝峰值力矩各≤1.1倍，再要求四项RMS分别≤reference的1、1.01或1.02倍。所有都是离线比较条件，不是临床或机器人安全限值。E3归一化分母始终为原同模型reference。

| model | family | tier | feasible_count | improvement_pct | E2 | parameter_flex | parameter_extend | share_shift |
|---|---|---|---|---|---|---|---|---|
| LEG_0_NOMINAL | BETA_TIMING | 0 | 16 | 1.41368 | 0.999973 | -0.12 | 0.03 | 0.05 |
| LEG_0_NOMINAL | BETA_TIMING | 0.01 | 109 | 2.19828 | 1.00185 | -0.12 | 0.12 | 0 |
| LEG_0_NOMINAL | BETA_TIMING | 0.02 | 168 | 2.19828 | 1.00185 | -0.12 | 0.12 | 0 |
| LEG_1_HEAVY_HIP_STIFF | BETA_TIMING | 0 | 11 | 0.893266 | 0.999899 | -0.03 | 0.06 | 0.05 |
| LEG_1_HEAVY_HIP_STIFF | BETA_TIMING | 0.01 | 104 | 2.09356 | 1.00087 | -0.12 | 0.12 | 0 |
| LEG_1_HEAVY_HIP_STIFF | BETA_TIMING | 0.02 | 162 | 2.09356 | 1.00087 | -0.12 | 0.12 | 0 |
| LEG_2_KNEE_DOMINANT | BETA_TIMING | 0 | 3 | 0.0631205 | 0.999741 | 0 | 0 | 0.1 |
| LEG_2_KNEE_DOMINANT | BETA_TIMING | 0.01 | 36 | 0.469328 | 1.00788 | -0.06 | 0.03 | 0.05 |
| LEG_2_KNEE_DOMINANT | BETA_TIMING | 0.02 | 126 | 0.871489 | 1.01736 | -0.12 | 0.06 | 0.05 |
| LEG_3_NONLINEAR_COUPLED | BETA_TIMING | 0 | 3 | 0.0192777 | 0.999941 | 0 | 0 | 0.1 |
| LEG_3_NONLINEAR_COUPLED | BETA_TIMING | 0.01 | 114 | 1.88789 | 1.0029 | -0.12 | 0.12 | 0 |
| LEG_3_NONLINEAR_COUPLED | BETA_TIMING | 0.02 | 168 | 1.88789 | 1.0029 | -0.12 | 0.12 | 0 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | BETA_TIMING | 0 | 69 | 1.81485 | 0.9967 | -0.12 | 0.12 | 0 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | BETA_TIMING | 0.01 | 126 | 1.81485 | 0.9967 | -0.12 | 0.12 | 0 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | BETA_TIMING | 0.02 | 200 | 1.81485 | 0.9967 | -0.12 | 0.12 | 0 |
| MYOLEG_NATIVE_P0 | BETA_TIMING | 0 | 77 | 2.38167 | 0.996635 | 0.12 | -0.12 | -0.1 |
| MYOLEG_NATIVE_P0 | BETA_TIMING | 0.01 | 144 | 2.38167 | 0.996635 | 0.12 | -0.12 | -0.1 |
| MYOLEG_NATIVE_P0 | BETA_TIMING | 0.02 | 189 | 2.38167 | 0.996635 | 0.12 | -0.12 | -0.1 |
| LEG_0_NOMINAL | KEY_POSTURE_TIMING | 0 | 27 | 2.43681 | 0.999621 | -0.12 | -0.09 | 0.1 |
| LEG_0_NOMINAL | KEY_POSTURE_TIMING | 0.01 | 125 | 2.56572 | 1.00069 | -0.12 | -0.12 | 0.05 |
| LEG_0_NOMINAL | KEY_POSTURE_TIMING | 0.02 | 174 | 2.56572 | 1.00069 | -0.12 | -0.12 | 0.05 |
| LEG_1_HEAVY_HIP_STIFF | KEY_POSTURE_TIMING | 0 | 21 | 1.85031 | 0.999975 | -0.06 | -0.09 | 0.1 |
| LEG_1_HEAVY_HIP_STIFF | KEY_POSTURE_TIMING | 0.01 | 120 | 2.79185 | 1.0005 | -0.12 | -0.12 | 0.05 |
| LEG_1_HEAVY_HIP_STIFF | KEY_POSTURE_TIMING | 0.02 | 145 | 2.79185 | 1.0005 | -0.12 | -0.12 | 0.05 |
| LEG_2_KNEE_DOMINANT | KEY_POSTURE_TIMING | 0 | 3 | 0.0631205 | 0.999741 | 0 | 0 | 0.1 |
| LEG_2_KNEE_DOMINANT | KEY_POSTURE_TIMING | 0.01 | 28 | 0.902043 | 1.00854 | -0.06 | -0.03 | 0.1 |
| LEG_2_KNEE_DOMINANT | KEY_POSTURE_TIMING | 0.02 | 125 | 1.68289 | 1.01762 | -0.12 | -0.09 | 0.1 |
| LEG_3_NONLINEAR_COUPLED | KEY_POSTURE_TIMING | 0 | 3 | 0.0192777 | 0.999941 | 0 | 0 | 0.1 |
| LEG_3_NONLINEAR_COUPLED | KEY_POSTURE_TIMING | 0.01 | 120 | 2.60622 | 1.00363 | -0.12 | -0.12 | 0.05 |
| LEG_3_NONLINEAR_COUPLED | KEY_POSTURE_TIMING | 0.02 | 145 | 2.60622 | 1.00363 | -0.12 | -0.12 | 0.05 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | KEY_POSTURE_TIMING | 0 | 102 | 2.18297 | 0.996499 | -0.12 | -0.12 | 0.05 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | KEY_POSTURE_TIMING | 0.01 | 135 | 2.18297 | 0.996499 | -0.12 | -0.12 | 0.05 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | KEY_POSTURE_TIMING | 0.02 | 198 | 2.18297 | 0.996499 | -0.12 | -0.12 | 0.05 |
| MYOLEG_NATIVE_P0 | KEY_POSTURE_TIMING | 0 | 93 | 2.98444 | 0.995407 | 0.12 | 0.12 | -0.1 |
| MYOLEG_NATIVE_P0 | KEY_POSTURE_TIMING | 0.01 | 135 | 2.98444 | 0.995407 | 0.12 | 0.12 | -0.1 |
| MYOLEG_NATIVE_P0 | KEY_POSTURE_TIMING | 0.02 | 192 | 2.98444 | 0.995407 | 0.12 | 0.12 | -0.1 |

## 第二组：参考＋3次追加试验

Reference仅执行一次并横向保留其值，其余方法总预算4。Random为预先指定的0–19共20个种子；其余确定性方法seed=0。主结果为已执行且满足负荷条件的最好E3；网格可行最优仅在顺序运行结束后由evaluator计算。归一化regret除以同模型同族同约束的reference-to-oracle headroom，不跨族共用分母。

下表先对Random种子求均值，再对6个模型等权平均；单模型与全部种子另有CSV。改善单位为相对reference的百分数；normalized_regret=0表示达到该域可行oracle，1表示仍为reference；constraint_violations是3次追加中平均超限次数。

| family | tier | method | improvement_pct | normalized_regret | constraint_violations |
|---|---|---|---|---|---|
| BETA_TIMING | 0 | REFERENCE | 0 | 1 | 0 |
| BETA_TIMING | 0 | RANDOM | 0.257063 | 0.862405 | 2.74167 |
| BETA_TIMING | 0 | ADAPTIVE_GREEDY | 0.699421 | 0.666667 | 2 |
| BETA_TIMING | 0 | PURE_BO_EI | 0.688355 | 0.672764 | 2.5 |
| BETA_TIMING | 0 | MODEL_INFORMED_BO_EI | 0.699421 | 0.666667 | 2 |
| BETA_TIMING | 0.01 | REFERENCE | 0 | 1 | 0 |
| BETA_TIMING | 0.01 | RANDOM | 0.587307 | 0.703469 | 2.21667 |
| BETA_TIMING | 0.01 | ADAPTIVE_GREEDY | 1.72938 | 0.166667 | 0.5 |
| BETA_TIMING | 0.01 | PURE_BO_EI | 1.67161 | 0.194467 | 1.5 |
| BETA_TIMING | 0.01 | MODEL_INFORMED_BO_EI | 1.72938 | 0.166667 | 0.5 |
| BETA_TIMING | 0.02 | REFERENCE | 0 | 1 | 0 |
| BETA_TIMING | 0.02 | RANDOM | 0.666977 | 0.663521 | 1.80833 |
| BETA_TIMING | 0.02 | ADAPTIVE_GREEDY | 1.72938 | 0.166667 | 0.5 |
| BETA_TIMING | 0.02 | PURE_BO_EI | 1.67161 | 0.194467 | 1.5 |
| BETA_TIMING | 0.02 | MODEL_INFORMED_BO_EI | 1.72938 | 0.166667 | 0.5 |
| KEY_POSTURE_TIMING | 0 | REFERENCE | 0 | 1 | 0 |
| KEY_POSTURE_TIMING | 0 | RANDOM | 0.312878 | 0.875862 | 2.75 |
| KEY_POSTURE_TIMING | 0 | ADAPTIVE_GREEDY | 1.26737 | 0.5 | 1.83333 |
| KEY_POSTURE_TIMING | 0 | PURE_BO_EI | 0.346521 | 0.841262 | 2.5 |
| KEY_POSTURE_TIMING | 0 | MODEL_INFORMED_BO_EI | 1.26737 | 0.5 | 1.83333 |
| KEY_POSTURE_TIMING | 0.01 | REFERENCE | 0 | 1 | 0 |
| KEY_POSTURE_TIMING | 0.01 | RANDOM | 0.744898 | 0.706641 | 2.2 |
| KEY_POSTURE_TIMING | 0.01 | ADAPTIVE_GREEDY | 2.18853 | 0.166667 | 0.5 |
| KEY_POSTURE_TIMING | 0.01 | PURE_BO_EI | 1.60152 | 0.368809 | 1 |
| KEY_POSTURE_TIMING | 0.01 | MODEL_INFORMED_BO_EI | 2.18853 | 0.166667 | 0.5 |
| KEY_POSTURE_TIMING | 0.02 | REFERENCE | 0 | 1 | 0 |
| KEY_POSTURE_TIMING | 0.02 | RANDOM | 0.903962 | 0.638576 | 1.89167 |
| KEY_POSTURE_TIMING | 0.02 | ADAPTIVE_GREEDY | 2.46902 | 0 | 0.333333 |
| KEY_POSTURE_TIMING | 0.02 | PURE_BO_EI | 1.60152 | 0.368809 | 1 |
| KEY_POSTURE_TIMING | 0.02 | MODEL_INFORMED_BO_EI | 2.46902 | 0 | 0.333333 |

## 适配与信息边界

新样条参数是α_flex、α_extend、time_share_shift；旧族是β_flex、β_extend、time_share_shift。没有把第三维丢弃，也没有把样条伪装成冻结625V3域。独立模块保留原Matérn-5/2核实现、length_scale=0.7、signal_std=0.6、noise floor=1e-6和jitter规则；特征尺度事先固定为(0.03,0.03,0.05)，复用原EI函数与选择器，xi=0，不优化GP超参数。这是三维适配实验，不是未经改变的二维API实验；两个参数族的几何尺度不完全等价。

Adaptive Greedy和Model-Informed BO共用原五参数时序估计器与新增E3投影。候选的q/dq/ddq与实际重分配时间用于物理预测；每次更新θ后，重新预测所有候选及过去观测残差。Pure BO只用有效标量E3；Random不用模型。GP的三维输入测试在第三坐标为0时与原二维结果一致。

顺序运行只预先筛选轨迹运动学限制；不使用真实E2、峰值或最终可行名单。完整时序来自已保存的确定性MuJoCo回放，只释放被请求候选，计一次逻辑完整轨迹评估；本次不是新运行MuJoCo或实物执行。Reference先回放，取得当前模型分母和峰值。超限后才标记该次不合格，仍消耗预算；其标量不进入GP、时序也不进入辨识，但原始E3/E2保留在日志。模型没有约束预测器、额外诊断或试验补偿，这会限制严格约束下的表现。

旧Observation/payload的beta_flex/beta_extend字段仅作前两维身份槽，KEY_POSTURE时实际含义是α；candidate_id完整区分第三维。导出明确使用parameter_flex/parameter_extend/share_shift，不将它们解释成生理参数。代数等效平面力仍不等于真实袖带测力。

## 文件与验证

- constraint_sensitivity.csv：第一组可行数量、最优与参数。
- algorithm_results.csv：每预算、方法、种子的最佳E3、regret和超限数。
- trial_history.csv：每次实际请求的候选、真实回放E3/E2、是否超限、下一步选择。
- identification_diagnostics.json：有效过去episode/sample数、θ、拟合诊断。
- per_model_algorithm_summary.csv、cross_model_summary.csv、random_seed_summary.csv：汇总。
- constraint_sensitivity.png、low_budget_curves.png：两组图。

全部864个顺序运行、3348条执行记录及3456条预算汇总已核对参考初始化、无重复、超限判定、最佳值和oracle界限。3项新针对性测试通过：三维核与二维一致/第三维生效；未来预算不改前序选择/真值列篡改不改变运动学候选域；无重复执行/超限占预算/标量隔离/最佳值单调。旧算法、E2/V3、机械模型、控制代码和历史结果均未修改；未自动提交。

复现：`python -m lower_limb_sim.e3_low_budget.run`，然后 `python -m lower_limb_sim.e3_low_budget.report`。
