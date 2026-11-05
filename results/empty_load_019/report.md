# 假腿实验

状态：FAILED
训练：DISABLED
原因：command deadline missed; refusing late catch-up dispatch: lateness=0.036076s;max_runtime_gate_ms=8.842;max_health_snapshot_ms=8.829;fine_entry_remaining_ms=27.724;fine_yields=17;wake_lateness_ms=36.074

机器人快照（含启动）：有效 1491/1612；曾就绪 True；就绪后失效 1。
- 力查询 P95/最大：1.225 / 23.930 毫秒
- 状态时龄 P95/最大：8.578 / 13.972 毫秒
- 力时龄 P95/最大：20.712 / 49.495 毫秒
- 状态/力时差 P95/最大：13.101 / 42.506 毫秒
机器人失效原因：{"state_wrench_skew_unavailable_or_exceeded": 121, "wrench_not_ready": 120, "wrench_process_not_ready": 120, "wrench_stale_or_unavailable": 120}
发送记录：dispatch.csv。
