# 假腿实验

状态：FAILED
训练：DISABLED
原因：final pre-start runtime safety gate: wrench_sample_stale_or_clock_invalid;wrench_stale_or_unavailable;state_wrench_skew_unavailable_or_exceeded;force_magnitude_n_invalid;torque_magnitude_nm_invalid;wrench_stale;state_wrench_skew_limit_exceeded;force_limit_or_value_unavailable;torque_limit_or_value_unavailable

机器人快照（含启动）：有效 22/155；曾就绪 True；就绪后失效 0。
- 力查询 P95/最大：1.193 / 1.294 毫秒
- 状态时龄 P95/最大：8.575 / 16.509 毫秒
- 力时龄 P95/最大：17.086 / 18.059 毫秒
- 状态/力时差 P95/最大：9.596 / 11.063 毫秒
机器人失效原因：{"state_wrench_skew_unavailable_or_exceeded": 133, "wrench_not_ready": 133, "wrench_process_not_ready": 133, "wrench_stale_or_unavailable": 133}
