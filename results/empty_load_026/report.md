# 假腿实验

状态：FAILED
训练：DISABLED
原因：realtime callback deadline missed; refusing late dispatch: index=75;lateness=0.030015s

机器人快照（含启动）：有效 1333/1460；曾就绪 True；就绪后失效 2。
- 力查询 P95/最大：1.200 / 46.028 毫秒
- 状态时龄 P95/最大：8.644 / 22.229 毫秒
- 力时龄 P95/最大：21.227 / 60.227 毫秒
- 状态/力时差 P95/最大：13.540 / 52.923 毫秒
机器人失效原因：{"state_wrench_skew_unavailable_or_exceeded": 127, "wrench_not_ready": 125, "wrench_process_not_ready": 125, "wrench_stale_or_unavailable": 127}
发送记录：dispatch.csv。
