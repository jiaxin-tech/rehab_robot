# Current P2 decision-rule semantics

## Direct rule

The active P2 V1 research guard computes `I_pred = -deltaJ_pred` and authorizes a supported local candidate only when `I_pred - U_global_max - 0.005 > 0` (`research_decision_guarded_sequential_personalization.py`, lines 502-550). The unchanged `0.005` originates as `OBJECTIVE_EQUIVALENCE_TOLERANCE` (`mechanical_objective.py`, lines 18-19). The same value defines mechanically equivalent ranking and actual-trial acceptance; it is a minimum meaningful objective-magnitude convention, not an uncertainty estimate, probability, or robot-safety threshold.

`U` is an empirical absolute error statistic for predicted objective difference. In current P2 V1 it is the maximum error from the current global validation-pair audit. In the default-off bundle candidate it is an independently calibrated scale-specific P95 endpoint residual. It describes model-decision disagreement; it does not change what 0.005 means.

## Bundle candidate rule

The current default-off bundle comparator uses `I_endpoint_pred - U_scale_P95 - 0.005 > 0`, with 2/3/5-step residual evidence of {2: 0.00164217053717, 3: 0.00244544326845, 5: 0.00400496043747}. A passing endpoint authorizes only the next adjacent formal-grid step, after which the five-parameter model is refit and the full prediction map is recomputed.

## Evidence for addition

The repository contains the additive formula as an implementation/design choice, but the searched code, protocol reports, calibration reports, and manifests contain no theorem, loss-derived risk allocation, physical law, or preregistered scientific argument requiring `required_margin = 0.005 + U`. The calibration report explicitly says it selected no threshold or policy and inferred no universal scale law. Therefore the correct audit label is:

`ADDITIVE_MARGIN_IS_DESIGN_ASSUMPTION`

This does not prove the additive rule is wrong. It means its necessity is not established. The shadow comparators therefore separate magnitude from direction without changing either the 0.005 value or the residual evidence.

Independent one-step direction evidence is descriptive: 306/324 pairs agree in sign and 18/324 reverse. S1 calls a stratum direction-supported only when supporting pairs outnumber contradicting pairs; it makes no probability claim. S2 uses the transparent research interval `[deltaJ_pred-U_P95, deltaJ_pred+U_P95]` and requires its upper endpoint to remain below zero.
