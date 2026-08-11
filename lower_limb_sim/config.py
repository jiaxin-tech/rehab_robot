"""二维下肢运动学模块的集中配置。"""

from pathlib import Path

# 人体下肢几何尺寸，单位为 m。
L1 = 0.42
L2 = 0.30

# 关节角范围仅用于仰卧屈髋、屈膝姿态。内部计算统一使用 rad。
hip_range_deg = (0.0, 120.0)
knee_range_deg = (5.0, 130.0)

# 工作空间图谱采样分辨率。
angle_step_deg = 1.0

# abs(sin(q_knee)) 小于该值时标记为接近膝伸直奇异位形。
singularity_threshold = 0.05

# Jacobian 与准静态力映射有效性阈值。
jacobian_det_threshold = 1e-4
jacobian_condition_limit = 100.0
force_magnitude_limit_n = 1000.0

# 仅用于虚拟受试者质心参数的合法性检查。完整小腿长度不等于 L2；
# L2 仍只表示膝关节到束缚带等效牵引点的运动学长度。
virtual_shank_length_m = 0.40

# 查询点与最近图谱点的最大允许距离，单位为 m。
query_max_distance_m = 0.02

MODULE_DIR = Path(__file__).resolve().parent
workspace_data_dir = MODULE_DIR / "data" / "workspace"
workspace_csv_path = workspace_data_dir / "workspace_atlas.csv"
workspace_npy_path = workspace_data_dir / "workspace_atlas.npy"
force_map_data_dir = MODULE_DIR / "data" / "force_maps"
dynamic_trajectory_data_dir = MODULE_DIR / "data" / "dynamic_trajectories"
identification_data_dir = MODULE_DIR / "data" / "identification"

# 第三阶段 software_test_trajectory 配置。
dynamic_sampling_frequency_hz = 100.0
test_trajectory_start_deg = (20.0, 20.0)
test_trajectory_end_deg = (120.0, 120.0)
speed_profile_one_way_duration_s = {
    "slow": 12.0,
    "nominal": 6.0,
    "fast": 3.0,
}
dynamic_model_version = "lower_limb_sim_dynamic_v1"

# 第四阶段：虚拟受试者动力学参数辨识。辨识轨迹不是临床参考轨迹。
identification_trajectory_id = "identification_excitation_trajectory"
identification_trajectory_endpoints_deg = {
    "coupled": ((20.0, 20.0), (70.0, 120.0)),
    # 题设给出的 (120°, 60°) 会让当前 L1/L2 下的牵引点越过髋后方。
    # 将膝终点提高到 80°，保留髋主导和 120° 最大髋屈曲，同时满足 x_pull>=0。
    "hip_dominant": ((20.0, 30.0), (120.0, 80.0)),
    # 题设给出的 (40°, 120°) 会让牵引点略低于床面；50° 是留有余量的
    # 膝主导路径，髋变化仍显著小于膝变化。
    "knee_dominant": ((20.0, 20.0), (50.0, 120.0)),
}
identification_dataset_split = {
    ("coupled", "slow"): "train",
    ("coupled", "fast"): "train",
    ("hip_dominant", "nominal"): "train",
    ("knee_dominant", "nominal"): "train",
    ("hip_dominant", "slow"): "validation",
    ("knee_dominant", "fast"): "validation",
    ("coupled", "nominal"): "test",
    ("hip_dominant", "fast"): "test",
    ("knee_dominant", "slow"): "test",
}

identification_parameter_names = (
    "mass_scale",
    "k_hip_nm_per_rad",
    "k_knee_nm_per_rad",
    "b_hip_nm_s_per_rad",
    "b_knee_nm_s_per_rad",
)
identification_initial_guess = {
    "mass_scale": 1.0,
    "k_hip_nm_per_rad": 10.0,
    "k_knee_nm_per_rad": 10.0,
    "b_hip_nm_s_per_rad": 1.0,
    "b_knee_nm_s_per_rad": 1.0,
}
identification_lower_bounds = {
    "mass_scale": 0.6,
    "k_hip_nm_per_rad": 0.0,
    "k_knee_nm_per_rad": 0.0,
    "b_hip_nm_s_per_rad": 0.0,
    "b_knee_nm_s_per_rad": 0.0,
}
identification_upper_bounds = {
    "mass_scale": 1.6,
    "k_hip_nm_per_rad": 60.0,
    "k_knee_nm_per_rad": 60.0,
    "b_hip_nm_s_per_rad": 10.0,
    "b_knee_nm_s_per_rad": 10.0,
}
# scipy least_squares 的变量尺度；避免刚度数值量级掩盖 mass_scale/阻尼。
identification_parameter_scales = {
    "mass_scale": 1.0,
    "k_hip_nm_per_rad": 20.0,
    "k_knee_nm_per_rad": 20.0,
    "b_hip_nm_s_per_rad": 3.0,
    "b_knee_nm_s_per_rad": 3.0,
}
identification_loss = "soft_l1"
identification_random_seed = 20260726
identification_model_version = "lower_limb_sim_identification_v1"

# 阶段 4.5A：wrench 与运动状态的时间对齐估计和补偿。
delay_compensation_data_dir = MODULE_DIR / "data" / "delay_compensation"
delay_search_min_ms = -50
delay_search_max_ms = 50
delay_search_step_ms = 1
delay_search_values_ms = tuple(
    range(delay_search_min_ms, delay_search_max_ms + 1, delay_search_step_ms)
)
known_delay_experiments_ms = (0, 8, 16, 24, 32, 40)
max_alignment_interpolation_gap_s = 0.020
delay_search_common_margin_s = 0.050
delay_search_minimum_coverage_ratio = 0.80
delay_compensation_model_version = "lower_limb_sim_delay_alignment_v1"

# 阶段 4.5B：变化延迟、抖动和冻结条件下的因果缓存式时间匹配。
variable_delay_data_dir = MODULE_DIR / "data" / "variable_delay"
variable_delay_random_seed = 20260728
variable_delay_model_version = "lower_limb_sim_variable_delay_alignment_v1"
variable_delay_scenarios = (
    "fixed_16ms",
    "piecewise_delay",
    "gradual_drift",
    "jitter_low",
    "jitter_medium",
    "bimodal_delay",
    "long_tail",
    "stale_freeze",
    "dropout_5pct",
    "combined_realistic",
)
variable_delay_core_scenarios = (
    "fixed_16ms",
    "piecewise_delay",
    "jitter_medium",
    "long_tail",
    "stale_freeze",
    "combined_realistic",
)
state_buffer_duration_s = 2.0
max_state_interpolation_interval_s = 0.020
max_state_match_error_s = 0.005
max_wrench_age_s = 0.100
delay_tracker_window_duration_s = 2.0
delay_tracker_update_interval_s = 0.5
delay_tracker_search_min_ms = -50
delay_tracker_search_max_ms = 80
delay_tracker_search_step_ms = 1
delay_tracker_filter_alpha = 0.5
maximum_delay_change_per_update_ms = 8.0
delay_tracker_minimum_effective_samples = 25
delay_tracker_minimum_excitation_score = 0.05

# 阶段 4.5C：复杂虚拟生成模型与简化五参数模型的失配验证。
model_mismatch_data_dir = MODULE_DIR / "data" / "model_mismatch"
model_mismatch_random_seed = 20260802
model_mismatch_model_version = "lower_limb_sim_model_mismatch_v1"
model_mismatch_scenarios = (
    "matched_linear",
    "nonlinear_stiffness_mild",
    "nonlinear_stiffness_strong",
    "hip_knee_coupling_mild",
    "hip_knee_coupling_strong",
    "nonlinear_damping_mild",
    "structured_residual",
    "combined_mild",
    "combined_strong",
)
# NRMSE = RMSE / (max(true torque) - min(true torque) + epsilon) * 100%.
model_mismatch_nrmse_epsilon_nm = 1e-9
model_mismatch_nrmse_minimum_range_nm = 1e-3
model_mismatch_diagnostic_correlation_threshold = 0.30

# 阶段 4.5D：人体几何标定与运动学观测误差。全部仍为离线软件虚拟数据。
geometry_error_data_dir = MODULE_DIR / "data" / "geometry_error"
geometry_error_random_seed = 20260803
geometry_error_model_version = "lower_limb_sim_geometry_observation_v1"
geometry_observation_modes = (
    "oracle_true_joint_state",
    "tcp_inverse_kinematics",
    "independent_joint_measurement",
)
geometry_default_offline_derivative_method = "central_difference_offline"
geometry_noisy_offline_derivative_method = "savitzky_golay_offline"
geometry_savgol_window_duration_s = 0.31
geometry_savgol_polynomial_order = 3
geometry_causal_filter_time_constant_s = 0.08
geometry_max_derivative_gap_s = 0.05
geometry_ik_domain_clip_tolerance = 1e-10
geometry_max_joint_jump_deg = 8.0
geometry_noise_seed_count = 20
geometry_domain_margin_fraction = 0.0

# 阶段 5A：外部三维骨架参考轨迹导入。输入 CSV 没有可靠帧率元数据，
# 因此这里故意不提供默认 fps；只有调用者显式传入 fps 后才允许生成导数和
# 动力学结果。
reference_trajectory_data_dir = (
    MODULE_DIR / "data" / "reference_trajectories" / "processed"
)
reference_trajectory_model_version = "lower_limb_sim_reference_import_v1"
reference_savgol_window_length = 21
reference_savgol_polynomial_order = 3

# 阶段 5B：只复用阶段 5A 已处理的参考周期形状，并为它指定新的时间律。
# 原始骨架 fps 仍然未知；以下 duration 是明确给定的“重定时”时长，绝不
# 代表原始骨架动作速度。
reference_retiming_data_dir = (
    MODULE_DIR / "data" / "reference_trajectories" / "retimed"
)
reference_retiming_model_version = "lower_limb_sim_reference_retiming_v1"
reference_phase_samples_per_segment = 201
reference_retiming_durations_s = {
    "slow": {"flexion": 12.0, "extension": 12.0},
    "nominal": {"flexion": 6.0, "extension": 6.0},
    "fast": {"flexion": 3.0, "extension": 3.0},
}
