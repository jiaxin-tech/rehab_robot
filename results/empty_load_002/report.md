# 假腿实验

状态：FAILED
训练：DISABLED
原因：runtime safety gate: state_wrench_skew_failure;wrench_stale_or_unavailable;state_wrench_skew_unavailable_or_exceeded;force_magnitude_n_invalid;torque_magnitude_nm_invalid;wrench_stale;state_wrench_skew_limit_exceeded;force_limit_or_value_unavailable;torque_limit_or_value_unavailable

机器人快照（含启动）：有效 10/139；曾就绪 True；就绪后失效 8。
- 力查询 P95/最大：1.144 / 1.144 毫秒
- 状态时龄 P95/最大：8.487 / 10.537 毫秒
- 力时龄 P95/最大：217.292 / 250.062 毫秒
- 状态/力时差 P95/最大：208.461 / 242.096 毫秒
机器人失效原因：{"state_wrench_skew_failure": 7, "state_wrench_skew_unavailable_or_exceeded": 129, "wrench_not_ready": 121, "wrench_process_not_ready": 121, "wrench_stale_or_unavailable": 129}
