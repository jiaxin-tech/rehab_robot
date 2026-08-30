# BO_DESIGN_AUDIT

BO uses the identical 21,025-point geometry-valid alpha grid and the same
case-specific J_pred from theta_hat. BO-A searches all 21,024 non-zero points;
BO-B applies the unchanged 90% coverage and J_pred < 1 screen. Alpha is mapped
from [-5.0, -5.0, -0.03] to [2.0, 2.0, 0.03] into [-1,1]^3 without truth.

The prior is J_pred plus a zero-mean residual Matérn-5/2 ARD GP. C1 is the
frozen model Top-1. Expected Improvement sees only queried residuals. Both BO
variants finish before the full Oracle landscape is revealed. No hyperparameter,
kernel, acquisition, budget, mismatch, or threshold search was performed.
