# 第四阶段参数泄漏审计

本审计只针对 `lower_limb_sim` 的软件虚拟受试者辨识。结论不代表真实患者
参数有效，也不授权任何真实机器人运动、采集或安全阈值更改。

## 数据边界

辨识器的数值输入白名单是：

```text
q_hip, q_knee
dq_hip, dq_knee
ddq_hip, ddq_knee
fx_observed_n, fz_observed_n
sample_valid
L1, L2
BaselineSubjectTemplate
```

`BaselineSubjectTemplate` 只包含 baseline 质量/惯量、质心距离、中性角和
重力；不包含 `subject_id`、真实 `mass_scale`、真实刚度或真实阻尼。

## 逐项检查

1. **估计器没有读取 `VirtualSubject` 真实参数。**  
   `parameter_estimator.py` 不导入 `DYNAMIC_SUBJECTS` 或
   `get_dynamic_subject`。每次残差计算都由
   `candidate_subject_from_parameters()` 新建 frozen 候选参数对象；不会
   修改 baseline 模板或真值对象。

2. **训练 DataFrame 没有 true parameter 字段。**  
   `identification_dataset.py` 输出前验证并禁止 `true_*`、
   `ground_truth*` 和 `tau_total*` 列。训练 CSV 也不保存 mass、inertia、
   K、B 的真值字段。

3. **`subject_id` 不用于查找参数。**  
   `subject_id` 仅作为结果追踪标签。估计器只按明确的数值观测列索引
   DataFrame；改变 `subject_id` 不会改变预测。

4. **真值只在最终评价模块加载。**  
   `run_identification.py` 先完成训练估计，再在
   `_true_parameters_for_evaluation()` 中加载虚拟真值，生成
   `parameter_estimates.csv` 的误差报告。该表从不回传给优化器。

5. **测试轨迹没有参与拟合。**  
   split 在轨迹级固定：train 4 条、validation 2 条、test 3 条。
   `estimate_subject_parameters()` 只接收 `splits["train"]`；validation 和
   test 仅在估计完成后计算预测指标。

6. **`tau_total` 没有被当成测量输入。**  
   数据生成器局部使用第三阶段 `tau_total` 合成虚拟端点力，并仅保存一个
   数值映射一致性误差；不把 `tau_total` 写入辨识表。估计器内部始终由
   `joint_torque_from_endpoint_force()` 重新执行
   `tau_measured = J(q).T @ F_observed`。即使篡改保存的
   `tau_measured_*` 辅助列，估计结果也不变。

7. **预测由估计参数重新运行动力学得到。**  
   train/validation/test 的 `tau_predicted_*` 都由 baseline 模板和五个
   估计参数建立新候选对象，再调用现有 `full_dynamics.inverse_dynamics`
   计算；预测不读取虚拟真值力矩或真值参数。

## 自动化证据

`test_identification_dataset.py` 和 `test_parameter_estimator.py` 覆盖：

- split 不重叠与 test 不进入训练；
- 数据字段无真值/`tau_total`；
- `J.T @ F` 逐样本一致；
- 篡改保存力矩和 `subject_id` 不改变辨识/预测；
- 无效、dropout、stale 样本不进入拟合；
- 固定种子噪声复现；
- advanced angle noise 的 `dq/ddq` 只由带噪 q 滤波微分获得。

## 仍需保持的边界

虚拟数据中的约 150–238 N 峰值只用于软件辨识流程验证。程序不按力幅值
缩小或删除有效样本，也绝不把这些值解释为真实康复力或真实机器人安全
阈值。真实 wrench 的坐标系、符号、延迟、参考点和有效性必须另做真机
验证，本阶段没有接入或修改任何真实 ROKAE 代码。
