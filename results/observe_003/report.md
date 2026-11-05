# 假腿实验

状态：OBSERVATION_COMPLETE
训练：DISABLED
原因：无

机器人快照（含启动）：有效 2/612；曾就绪 True；就绪后失效 472。
- 力查询 P95/最大：0.966 / 1.296 毫秒
- 状态时龄 P95/最大：8.411 / 19.787 毫秒
- 力时龄 P95/最大：9509.600 / 10013.018 毫秒
- 状态/力时差 P95/最大：9501.597 / 10005.487 毫秒
机器人失效原因：{"state_wrench_skew_failure": 471, "state_wrench_skew_unavailable_or_exceeded": 609, "wrench_not_ready": 138, "wrench_process_not_ready": 138, "wrench_stale_or_unavailable": 609}
采集：{"rows": 6871, "duration_s": 14.173703399999999, "sample_rate_hz": 484.7004206395345, "usable_for_dynamics": false}
数据问题：receive_time_gaps; hip_missing_flexion_deg; hip_has_no_joint_angle_calibration; knee_missing_flexion_deg; knee_has_no_joint_angle_calibration

原始数据：leg/leg.csv 和 robot/。
