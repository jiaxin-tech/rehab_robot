# 假腿实验

状态：FAILED
训练：DISABLED
原因：command deadline missed; refusing late catch-up dispatch: lateness=0.036102s;max_runtime_gate_ms=8.475;max_health_snapshot_ms=8.460;fine_entry_remaining_ms=0.000;fine_yields=0;wake_lateness_ms=36.101

机器人快照（含启动）：有效 411/535；曾就绪 True；就绪后失效 1。
- 力查询 P95/最大：1.230 / 2.058 毫秒
- 状态时龄 P95/最大：8.574 / 58.356 毫秒
- 力时龄 P95/最大：20.929 / 45.808 毫秒
- 状态/力时差 P95/最大：13.515 / 33.378 毫秒
机器人失效原因：{"state_stale_or_unavailable": 2, "state_wrench_skew_unavailable_or_exceeded": 123, "wrench_not_ready": 123, "wrench_process_not_ready": 123, "wrench_stale_or_unavailable": 123}
发送记录：dispatch.csv。
