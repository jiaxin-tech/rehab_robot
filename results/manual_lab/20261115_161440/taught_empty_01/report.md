# 假腿实验

状态：EMPTY_LOAD_COMPLETE
训练：NOT_APPLICABLE
原因：无

机器人快照（含启动）：有效 3443/3695；曾就绪 True；就绪后失效 72。
- 力查询 P95/最大：1.123 / 12.385 毫秒
- 状态时龄 P95/最大：7.510 / 11.354 毫秒
- 力时龄 P95/最大：20.833 / 456.632 毫秒
- 状态/力时差 P95/最大：17.814 / 456.090 毫秒
机器人失效原因：{"child:TimeoutError:ack_timeout": 2, "cleanup:RuntimeError:unacknowledged_audit_mailbox_cleanup_unconfirmed": 2, "state_wrench_skew_failure": 1, "state_wrench_skew_unavailable_or_exceeded": 252, "wrench_not_ready": 180, "wrench_process_not_ready": 180, "wrench_sample_stale_or_clock_invalid": 2, "wrench_stale_or_unavailable": 251}
发送记录：dispatch.csv。
