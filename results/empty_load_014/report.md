# 假腿实验

状态：FAILED
训练：DISABLED
原因：post-log runtime safety gate: state_freshness_failure;wrench_stale_or_unavailable;state_wrench_skew_unavailable_or_exceeded;force_magnitude_n_invalid;torque_magnitude_nm_invalid;wrench_stale;state_wrench_skew_limit_exceeded;force_limit_or_value_unavailable;torque_limit_or_value_unavailable

机器人快照（含启动）：有效 23/162；曾就绪 True；就绪后失效 4。
- 力查询 P95/最大：1.159 / 1.583 毫秒
- 状态时龄 P95/最大：8.725 / 50.485 毫秒
- 力时龄 P95/最大：73.822 / 100.403 毫秒
- 状态/力时差 P95/最大：66.590 / 91.981 毫秒
机器人失效原因：{"state_freshness_failure": 3, "state_stale_or_unavailable": 1, "state_wrench_skew_failure": 2, "state_wrench_skew_unavailable_or_exceeded": 138, "wrench_not_ready": 135, "wrench_process_not_ready": 135, "wrench_stale_or_unavailable": 138}
发送记录：dispatch.csv。
