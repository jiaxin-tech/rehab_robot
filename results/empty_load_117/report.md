# 假腿实验

状态：FAILED
训练：DISABLED
原因：realtime schedule cancelled: runtime_watchdog:runtime safety gate: state_stale_or_unavailable

机器人快照（含启动）：有效 264/385；曾就绪 True；就绪后失效 1。
- 力查询 P95/最大：1.252 / 11.378 毫秒
- 状态时龄 P95/最大：8.497 / 23.731 毫秒
- 力时龄 P95/最大：21.736 / 42.219 毫秒
- 状态/力时差 P95/最大：14.197 / 34.712 毫秒
机器人失效原因：{"state_freshness_failure": 1, "state_wrench_skew_unavailable_or_exceeded": 120, "wrench_not_ready": 120, "wrench_process_not_ready": 120, "wrench_stale_or_unavailable": 120}
发送记录：dispatch.csv。
