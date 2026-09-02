# Rehab Setup Frame Definition

`REHAB_SETUP_FRAME` (`R`) is fixed to the rigid bed/setup base, not to a human hip. Its origin is a preregistered durable setup fiducial. `+x_R` is the bed-plane longitudinal direction from proximal to distal/foot-side fiducials. `+z_R` is the upward normal to the fitted rigid reference plane. `+y_R=normalize(z_R cross x_R)`, so `x_R cross y_R=z_R`.

At least four labelled landmarks are required: origin, distal longitudinal and at least two other non-collinear bed-plane fiducials. Their geometry, coordinate values and uncertainty must be frozen in the authorized execution manifest. No transform is populated here.

Robot chain: `p_attach_TCP -> T_B_TCP(t) -> p_robot_attach_B(t)`. Limb chain: `p_exit_S -> T_R_S(t) -> T_B_R -> p_limb_attach_B(t)`. Both points enter `d_task` only in common base frame `B`. Controller world may be used only through a separately validated `T_W_B`.
