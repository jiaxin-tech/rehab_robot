# Endpoint Sensitivity Validation Plan

Goal: determine whether the endpoint separates prespecified small trajectory perturbations from repeatability noise. It is independent of robot approval and is not executed here.

After repeatability passes, preregister a small set of V3 `beta_flex/beta_extend` perturbations from the unchanged family. Select them from geometry/domain and approved exposure considerations, never from force outcomes or MyoLeg oracle ranking. Counterbalance order and repeat the reference/perturbations under the same setup. Estimate paired endpoint differences with uncertainty and compare their scale with within-candidate repeatability noise. Retain tracking and full force profiles as diagnostics.

No numeric effect threshold, sample size or robot-safe beta subset is frozen here. Failure to distinguish the prespecified perturbations means `ENDPOINT_NOT_SENSITIVE_ENOUGH_FOR_PERSONALIZATION`; it does not justify tuning preprocessing on the same data.
