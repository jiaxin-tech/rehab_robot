# 冻结算法依赖审计（2026-10-16）

结论：仓库内 7 个依赖和参考候选已通过 SHA-256 核对；5 个冻结算法文件尚未恢复，另有 1 个仓库内脚本与锁定哈希不一致。未用近似文件、简化算法或修改锁定哈希绕过检查。

## 缺失文件

- `C:/Users/CSR001279/Documents/ChatGPT/康复/完整算法验证_2026-09-29/config.py`
  - 期望：`4c0088edc7191fa799e8a69cd185aa5dfbe1fea36f2050339d5a92321f1865c5`
- `C:/Users/CSR001279/Documents/ChatGPT/康复/完整算法验证_2026-09-29/surrogate.py`
  - 期望：`c8243a57dd5c714b661641b06ba0888a4b369d706ad99784c859543ccae5a5b0`
- `C:/Users/CSR001279/Documents/ChatGPT/康复/完整算法验证_2026-09-29/policy.py`
  - 期望：`2ab3ed5ba501974b856342b1b96f7da2de5fec3a2bc1fe4fdd9c1218b9fdbee1`
- `C:/Users/CSR001279/Documents/ChatGPT/康复/升级模型试跑_2026-09-28/coupled_model.py`
  - 期望：`6fb016127bd05b209ab7bd42ce1424e68f784ea9f224e51ec087d65aa3df7186`
- `C:/Users/CSR001279/Documents/ChatGPT/康复/升级模型试跑_2026-09-28/root_cause/mass_model/coordinate_consistent_model.py`
  - 期望：`0900ff8fb6443a5bf558708c6e2833015579eda8d9a84bc25aa7b868a7c29b4e`

已搜索本机 Desktop、Documents、Downloads、可访问的 D 盘、当前仓库全部分支/历史/stash/不可达对象和本机 Claude 会话备份，没有找到这些文件或对应哈希。必须从原实验电脑或可信备份恢复。

## 哈希不一致

- 文件：`external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py`
- 锁定 SHA-256：`8a5c536d30d0031e0e49adaddd98a496c574e95bb1f75f2b3eddc07791e51472`
- 当前 SHA-256：`036b4e8e9a1f55cd5f17188ced9e95d506e0bbe898003e1ba279cd042b7013a1`

当前仓库历史中也未找到锁定哈希。应从生成本轮配置的冻结发布包恢复该精确版本，或由算法负责人审查差异后重新签发完整依赖清单；不能只把配置里的哈希改成当前值。

## 已通过

- `lower_limb_sim/config.py`
- `lower_limb_sim/dynamic_subject.py`
- `lower_limb_sim/parameter_estimator.py`
- `lower_limb_sim/mechanical_endpoints.py`
- `lower_limb_sim/e3_low_budget/run.py`
- `lower_limb_sim/myoleg_benchmark/portable_model.py`
- `external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml`
- `reference_release/reference_measured_asymmetric_closed_slow.csv`
