# 假腿实验

状态：FAILED
训练：DISABLED
原因：runtime safety gate: state_stale_or_unavailable;state_stale

机器人快照（含启动）：有效 519/643；曾就绪 True；就绪后失效 2。
- 力查询 P95/最大：1.180 / 12.374 毫秒
- 状态时龄 P95/最大：9.051 / 23.446 毫秒
- 力时龄 P95/最大：19.367 / 66.678 毫秒
- 状态/力时差 P95/最大：11.334 / 58.439 毫秒
机器人失效原因：{"state_freshness_failure": 2, "state_wrench_skew_unavailable_or_exceeded": 123, "wrench_not_ready": 122, "wrench_process_not_ready": 122, "wrench_stale_or_unavailable": 123}
发送记录：dispatch.csv。
