# 假腿实验

状态：FAILED
训练：NOT_APPLICABLE
原因：command deadline missed after intent logging; refusing dispatch

机器人快照（含启动）：有效 220/381；曾就绪 True；就绪后失效 3。
- 力查询 P95/最大：1.057 / 9.512 毫秒
- 状态时龄 P95/最大：7.470 / 73.052 毫秒
- 力时龄 P95/最大：19.885 / 446.974 毫秒
- 状态/力时差 P95/最大：17.565 / 405.113 毫秒
机器人失效原因：{"child:TimeoutError:ack_timeout": 1, "cleanup:RuntimeError:unacknowledged_audit_mailbox_cleanup_unconfirmed": 1, "state_wrench_skew_unavailable_or_exceeded": 160, "wrench_not_ready": 158, "wrench_process_not_ready": 158, "wrench_sample_stale_or_clock_invalid": 1, "wrench_stale_or_unavailable": 160}
发送记录：dispatch.csv。
