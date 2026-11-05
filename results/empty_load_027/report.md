# 假腿实验

状态：FAILED
训练：DISABLED
原因：realtime schedule cancelled: runtime_watchdog:runtime safety gate: state_stale_or_unavailable;state_stale

机器人快照（含启动）：有效 2327/2468；曾就绪 True；就绪后失效 6。
- 力查询 P95/最大：1.214 / 11.188 毫秒
- 状态时龄 P95/最大：8.568 / 31.599 毫秒
- 力时龄 P95/最大：21.041 / 53.896 毫秒
- 状态/力时差 P95/最大：13.449 / 52.114 毫秒
机器人失效原因：{"state_freshness_failure": 2, "state_stale_or_unavailable": 2, "state_wrench_skew_unavailable_or_exceeded": 138, "wrench_not_ready": 135, "wrench_process_not_ready": 135, "wrench_stale_or_unavailable": 136}
发送记录：dispatch.csv。
