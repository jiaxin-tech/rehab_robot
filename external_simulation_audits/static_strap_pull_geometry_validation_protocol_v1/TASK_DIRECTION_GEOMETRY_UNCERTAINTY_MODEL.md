# Task-Direction Geometry Uncertainty Model

Let `v=p_r-p_l`, `L=norm(v)` and `d=v/L`. First-order propagation uses

`J=(I-d d^T)/L`, `Sigma_v=Sigma_r+Sigma_l-Sigma_rl-Sigma_lr`, and `Sigma_d ~= J Sigma_v J^T`.

For small errors, `E[theta^2] ~= trace(Sigma_d)`. Include point-picking noise, eyelet localization/reinstall, `T_B_R`, surrogate pose, cuff remove/reattach, exit location and free-span fit residual; retain cross-covariance when points share a registration.

Estimate within-setup and between-setup components separately. Confirm the linearization with `100000` deterministic samples (seed `20260901`) through exact normalization. Report angular SD, P95 and maximum observed setup deviation plus endpoint/line uncertainties. Minimum separation, angular, displacement and line-fit thresholds remain null pending prospective metrology/endpoint-error-budget review. No scientific endpoint outcome may tune them.
