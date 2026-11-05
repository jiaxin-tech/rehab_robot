# 假腿实验

状态：FAILED
训练：DISABLED
原因：command deadline missed; refusing late catch-up dispatch: lateness=0.033694s;max_runtime_gate_ms=16.410;max_health_snapshot_ms=16.392;fine_entry_remaining_ms=21.219;fine_yields=7;wake_lateness_ms=33.693

机器人快照（含启动）：有效 965/1099；曾就绪 True；就绪后失效 1。
- 力查询 P95/最大：1.182 / 2.900 毫秒
- 状态时龄 P95/最大：8.604 / 23.460 毫秒
- 力时龄 P95/最大：20.860 / 49.988 毫秒
- 状态/力时差 P95/最大：13.217 / 42.040 毫秒
机器人失效原因：{"state_wrench_skew_unavailable_or_exceeded": 134, "wrench_not_ready": 133, "wrench_process_not_ready": 133, "wrench_stale_or_unavailable": 133}
发送记录：dispatch.csv。
