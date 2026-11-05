# 假腿实验

状态：FAILED
训练：DISABLED
原因：runtime safety gate: state_stale_or_unavailable;state_stale

机器人快照（含启动）：有效 16/160；曾就绪 True；就绪后失效 2。
- 力查询 P95/最大：1.076 / 1.099 毫秒
- 状态时龄 P95/最大：8.620 / 80.218 毫秒
- 力时龄 P95/最大：23.112 / 31.374 毫秒
- 状态/力时差 P95/最大：32.492 / 58.729 毫秒
机器人失效原因：{"state_freshness_failure": 2, "state_stale_or_unavailable": 2, "state_wrench_skew_unavailable_or_exceeded": 143, "wrench_not_ready": 142, "wrench_process_not_ready": 142, "wrench_stale_or_unavailable": 142}
发送记录：dispatch.csv。
