# 假腿实验

状态：EMPTY_LOAD_COMPLETE
训练：NOT_APPLICABLE
原因：无

机器人快照（含启动）：有效 8026/8383；曾就绪 True；就绪后失效 193。
- 力查询 P95/最大：1.143 / 28.045 毫秒
- 状态时龄 P95/最大：6.691 / 48.400 毫秒
- 力时龄 P95/最大：19.872 / 448.996 毫秒
- 状态/力时差 P95/最大：17.717 / 406.957 毫秒
机器人失效原因：{"child:TimeoutError:ack_timeout": 1, "cleanup:RuntimeError:unacknowledged_audit_mailbox_cleanup_unconfirmed": 1, "state_wrench_skew_unavailable_or_exceeded": 357, "wrench_not_ready": 164, "wrench_process_not_ready": 164, "wrench_sample_stale_or_clock_invalid": 1, "wrench_stale_or_unavailable": 352}
发送记录：dispatch.csv。
