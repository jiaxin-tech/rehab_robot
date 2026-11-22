# 假腿实验

状态：FAILED
训练：NOT_APPLICABLE
原因：runtime safety gate: child:TimeoutError:ack_timeout;cleanup:RuntimeError:unacknowledged_audit_mailbox_cleanup_unconfirmed;wrench_stale_or_unavailable;state_wrench_skew_unavailable_or_exceeded;force_magnitude_n_invalid;torque_magnitude_nm_invalid;wrench_stale;state_wrench_skew_limit_exceeded;force_limit_or_value_unavailable;torque_limit_or_value_unavailable

机器人快照（含启动）：有效 45/275；曾就绪 True；就绪后失效 54。
- 力查询 P95/最大：0.959 / 1.885 毫秒
- 状态时龄 P95/最大：8.303 / 66.082 毫秒
- 力时龄 P95/最大：1069.614 / 1167.318 毫秒
- 状态/力时差 P95/最大：1064.460 / 1165.901 毫秒
机器人失效原因：{"child:TimeoutError:ack_timeout": 53, "cleanup:RuntimeError:unacknowledged_audit_mailbox_cleanup_unconfirmed": 53, "state_wrench_skew_unavailable_or_exceeded": 229, "wrench_not_ready": 176, "wrench_process_not_ready": 176, "wrench_stale_or_unavailable": 230}
