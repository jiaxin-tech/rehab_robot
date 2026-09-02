# CODE_AND_MISMATCH_AUDIT

## Existing mismatch definitions

- `matched_linear`: family `matched_linear`, level `MATCHED`, terms [], parameters `{}`.
- `nonlinear_stiffness_mild`: family `nonlinear_stiffness`, level `MILD`, terms ['nonlinear_stiffness'], parameters `{"k3_hip_nm_per_rad3": 0.8, "k3_knee_nm_per_rad3": 0.6}`.
- `nonlinear_stiffness_strong`: family `nonlinear_stiffness`, level `STRONG`, terms ['nonlinear_stiffness'], parameters `{"k3_hip_nm_per_rad3": 4.0, "k3_knee_nm_per_rad3": 3.5}`.
- `hip_knee_coupling_mild`: family `hip_knee_coupling`, level `MILD`, terms ['hip_knee_coupling'], parameters `{"k_coupling_asymmetry": 0.4, "k_coupling_nm_per_rad": 1.5}`.
- `hip_knee_coupling_strong`: family `hip_knee_coupling`, level `STRONG`, terms ['hip_knee_coupling'], parameters `{"k_coupling_asymmetry": 0.6, "k_coupling_nm_per_rad": 7.0}`.
- `nonlinear_damping_mild`: family `nonlinear_damping`, level `MILD`, terms ['nonlinear_damping'], parameters `{"b2_hip_nm_s2_per_rad2": 0.15, "b2_knee_nm_s2_per_rad2": 0.12}`.
- `structured_residual`: family `structured_residual`, level `SINGLE_DEFINED_LEVEL`, terms ['structured_residual'], parameters `{"residual_torque_frequency": 1.0, "residual_torque_scale_nm": 0.6}`.
- `combined_mild`: family `combined`, level `MILD`, terms ['nonlinear_stiffness', 'hip_knee_coupling', 'nonlinear_damping', 'structured_residual'], parameters `{"b2_hip_nm_s2_per_rad2": 0.08, "b2_knee_nm_s2_per_rad2": 0.06, "k3_hip_nm_per_rad3": 0.5, "k3_knee_nm_per_rad3": 0.4, "k_coupling_asymmetry": 0.6, "k_coupling_nm_per_rad": 1.0, "residual_torque_frequency": 1.0, "residual_torque_scale_nm": 0.3}`.
- `combined_strong`: family `combined`, level `STRONG`, terms ['nonlinear_stiffness', 'hip_knee_coupling', 'nonlinear_damping', 'structured_residual'], parameters `{"b2_hip_nm_s2_per_rad2": 0.7, "b2_knee_nm_s2_per_rad2": 0.55, "k3_hip_nm_per_rad3": 3.5, "k3_knee_nm_per_rad3": 3.0, "k_coupling_asymmetry": 0.9, "k_coupling_nm_per_rad": 6.0, "residual_torque_frequency": 1.6, "residual_torque_scale_nm": 1.5}`.

The actual generator definitions support mild/strong ordering only within the
nonlinear-stiffness, hip-knee-coupling, and combined families. Nonlinear
damping has only a mild definition and structured residual has one defined
level. Therefore this stage does not create a synthetic global severity scalar.

All nine scenario definitions are present in the unchanged 15-case V1 case
plan (`15` cases total, including matched subject-specific and the
previously preregistered strong-mismatch cases).

## Truth-access audit

`build_predicted_map` explicitly reports `truth_evaluated_during_prediction =
false`. The existing V1 freezes its shortlist, persists a manifest, and then
requires `FrozenShortlistTruthGate` authorization. This stress stage additionally
persists `STRESS_TEST_PROTOCOL.json` and `FROZEN_BASELINE_MANIFEST.json`, including
all Random-3 seeds and identities, before the shared truth-open token is issued.
The full 21,025-point truth landscape is read only after frozen V1 candidate
execution and is used as B5/post-selection evaluation, never candidate ordering.

## Frozen inputs

- V1 manifest SHA-256: `7576e5a545878292f2eb1846e9cae780325a2e44bb58093dfb04bae982827498`
- Candidate lattice: 21,025 points.
- Model-domain coverage gate: 90%.
- Mechanical equivalence tolerance: 0.005.
- Active reference SHA-256: `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`.
- No robot, hardware, SDK, control, collection, wrench, or safety code is imported.
