# 假腿实验

状态：FAILED
训练：DISABLED
原因：command deadline missed; refusing late catch-up dispatch: lateness=0.034528s;max_runtime_gate_ms=7.896;max_health_snapshot_ms=7.879;fine_entry_remaining_ms=5.549;fine_yields=13;wake_lateness_ms=34.527

机器人快照（含启动）：有效 222/354；曾就绪 True；就绪后失效 0。
- 力查询 P95/最大：1.192 / 3.998 毫秒
- 状态时龄 P95/最大：8.523 / 13.716 毫秒
- 力时龄 P95/最大：20.839 / 32.688 毫秒
- 状态/力时差 P95/最大：13.337 / 24.715 毫秒
机器人失效原因：{"state_wrench_skew_unavailable_or_exceeded": 132, "wrench_not_ready": 132, "wrench_process_not_ready": 132, "wrench_stale_or_unavailable": 132}
发送记录：dispatch.csv。
