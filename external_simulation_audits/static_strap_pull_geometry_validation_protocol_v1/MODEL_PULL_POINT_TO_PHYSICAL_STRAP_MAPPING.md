# Model Pull Point to Physical Strap Mapping

Current classification: `MODEL_PULL_POINT_TO_PHYSICAL_STRAP_MAPPING = NOT_YET_CALIBRATED`.

The 2-D point uses `L2=0.30 m` from the knee and `theta_shank=q_hip-q_knee`. It is an equivalent traction point used for planar FK and start-anchored TCP displacement; it is not the ankle or a measured cuff attachment.

A future mapping audit should transform the L2 point into `REHAB_SETUP_FRAME`, compare it against the measured free-span line at every registered configuration, and report: nearest-point distance, direction-angle difference, configuration dependence and—if independently measured—resultant moment equivalence. Thresholds must be frozen before outcomes. The mapping may become `APPROXIMATE` only if its limited range passes; `DIRECTLY_MATCHED` requires direct geometric/mechanical evidence. This protocol changes neither L2 nor model kinematics.
