# 假腿实验

状态：FAILED
训练：DISABLED
原因：realtime schedule cancelled: runtime_watchdog:runtime safety gate: state_stale_or_unavailable;state_stale

机器人快照（含启动）：有效 2029/2167；曾就绪 True；就绪后失效 15。
- 力查询 P95/最大：1.187 / 15.368 毫秒
- 状态时龄 P95/最大：8.566 / 54.645 毫秒
- 力时龄 P95/最大：21.153 / 232.945 毫秒
- 状态/力时差 P95/最大：13.517 / 225.114 毫秒
机器人失效原因：{"state_freshness_failure": 3, "state_stale_or_unavailable": 3, "state_wrench_skew_unavailable_or_exceeded": 134, "wrench_not_ready": 123, "wrench_process_not_ready": 123, "wrench_stale_or_unavailable": 134}
发送记录：dispatch.csv。
