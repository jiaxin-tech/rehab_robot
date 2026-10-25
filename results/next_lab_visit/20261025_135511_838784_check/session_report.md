# Lab session

Action: check
Status: CHECK_COMPLETE_WITH_BLOCKERS



No automatic retry, reset, approval or next-candidate dispatch.

Empty-load: BLOCKED: ValueError:required path is missing

Training blockers:
- loaded setup must explicitly review realtime_preparation=same_session
- RT wrench states: same-session RT wrench config must review rtcontrolling
- collision events: events requires an explicit finite positive collision_event_max_age_s
- round setup must be explicitly reviewed
- analysis missing: sensor_receive_delay_s
- analysis missing: command_application_delay_s
- analysis missing: max_gap_s
- analysis missing: max_read_duration_s
- analysis missing: max_dispatch_time_error_s
- analysis missing: derivative_rate_hz
- analysis missing: derivative_window_s
- analysis missing: max_start_angle_error_deg
- analysis missing: max_tracking_error_deg
- max_leg_start_age_s: finite positive max_leg_start_age_s required
- robot safety: real robot execution blocked by experiment safety: experiment_safety_not_reviewed;max_start_anchor_position_error_m_not_configured;max_start_anchor_orientation_error_rad_not_configured;max_command_lateness_s_not_configured;reviewed_joint_soft_limits_rad_not_configured;reviewed_rt_filter_hz_not_configured;reviewed_rt_network_tolerance_percent_not_configured;joint_soft_limits_not_reviewed;realtime_configuration_not_reviewed
- joint angles/timing: reviewed cross-device timing and analysis configuration required
- planned leg angles: reviewed cross-device timing and analysis configuration required
- site evidence controller_safety: missing, unreviewed or changed evidence
- site evidence acquisition_stability: missing, unreviewed or changed evidence
- site evidence wrench_semantics: missing, unreviewed or changed evidence
- site evidence setup_and_stop: missing, unreviewed or changed evidence
- site evidence dummy_leg_installation: missing, unreviewed or changed evidence
- site evidence cross_device_timing: missing, unreviewed or changed evidence
- algorithm sources: algorithm dependency changed: external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py
