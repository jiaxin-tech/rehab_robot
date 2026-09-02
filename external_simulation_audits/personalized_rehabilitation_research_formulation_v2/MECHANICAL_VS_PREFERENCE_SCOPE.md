# Mechanical versus Preference Scope

## Primary: mechanical measurement-driven personalization

This is selected because it matches current observable engineering channels, has the smaller conceptual and ethical expansion, preserves the fixed-task V3 work, and is more defensible for the present undergraduate-project scope. The primary claim to test later is an equal-budget reduction in an independently evaluated, prespecified measured mechanical-interaction endpoint.

The primary candidate endpoint class is `EPISODE_RMS_VALIDATED_TASK_DIRECTION_INTERACTION_FORCE`. It is **not yet the final objective**: task direction, sign, frame, bias, delay, synchronization, repeatability and physical meaning must be calibrated independently. Until that stage finishes, `PRIMARY_OUTCOME_TYPE = MEASURED_MECHANICAL_INTERACTION_ENDPOINT_PENDING_INDEPENDENT_CALIBRATION`.

Secondary diagnostics may include peak interaction force/torque, time-profile features, pressure peak/concentration/centroid, tracking error and model residual. Safety limits and data-validity gates remain constraints, never reward terms. No arbitrary all-signal weighted score is frozen here.

## Optional extension: preference/comfort

| Label | Burden | Repeatability/bias | Low-budget and BO compatibility |
|---|---|---|---|
| Scalar rating | one response per trajectory; relatively low | scale drift, anchoring and inter-session calibration require study | simple regression/ordinal models; absolute scale may be noisy |
| Pairwise preference | requires explicit comparisons; burden can rise | relative judgments may be easier but order/context bias remains | natural fit for preference BO, but comparison graph must be designed |

If a preference branch is later approved, pairwise preference is the more direct methodological candidate for preference BO, with scalar rating as a possible secondary measure. This is not a human-study decision. `HUMAN_FEEDBACK_REQUIRED` applies to every comfort/preference claim.

Pressure is a possible comfort correlate, not comfort truth. Mechanical improvement and reported comfort may disagree and must be reported separately.
