# 假腿实验

状态：FAILED
训练：DISABLED
原因：command deadline missed; refusing late catch-up dispatch: lateness=0.032048s;max_runtime_gate_ms=6.404;max_health_snapshot_ms=6.381;fine_entry_remaining_ms=8.196;fine_yields=0;wake_lateness_ms=32.046

机器人快照（含启动）：有效 232/366；曾就绪 True；就绪后失效 0。
- 力查询 P95/最大：1.184 / 1.667 毫秒
- 状态时龄 P95/最大：8.624 / 25.364 毫秒
- 力时龄 P95/最大：21.055 / 39.826 毫秒
- 状态/力时差 P95/最大：13.430 / 31.899 毫秒
机器人失效原因：{"state_stale_or_unavailable": 1, "state_wrench_skew_unavailable_or_exceeded": 134, "wrench_not_ready": 134, "wrench_process_not_ready": 134, "wrench_stale_or_unavailable": 134}
发送记录：dispatch.csv。
