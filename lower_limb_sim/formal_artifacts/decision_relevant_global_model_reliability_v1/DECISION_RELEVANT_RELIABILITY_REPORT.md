# DECISION_RELEVANT_RELIABILITY_REPORT

Protocol: `DECISION_RELEVANT_GLOBAL_MODEL_RELIABILITY_CHARACTERIZATION_V1`. Evidence level: offline virtual research only.

## A. Why imperfect torque prediction need not imply a wrong decision

Personalization compares candidate objectives and directions. A model can have a systematic torque error while preserving relative ordering or the improve/neutral/worse sign near the current trajectory. This stage therefore audits error, sign, rank, false improvements, and regret separately; none is treated as approval by itself.

## B. Matched positive controls

Across four matched controls, maximum mean absolute J error was `2.15654e-13`, minimum sign agreement was `100%`, minimum Spearman rank correlation was `1`, and maximum predicted-best regret was `0`. These near-ideal controls support implementation consistency.

## C. combined_mild

Its existing adequacy trend remains `MODEL_STRUCTURE_LIMITATION`. Improvement-sign accuracy was `95.0488%`, false-improvement rate `0%`, global rank correlation `0.998666`, predicted-best regret `2.777e-05`, and one-step supported local regret `0`. Local diagnostic label: `POTENTIALLY_USEFUL_FOR_LOCAL_DECISION`.

## D. Mild mismatch local decision utility

Cases meeting the exact diagnostic condition (truth-best local choice matched and zero local false improvements): `baseline__nonlinear_stiffness_mild, baseline__hip_knee_coupling_mild, baseline__nonlinear_damping_mild, baseline__structured_residual, baseline__combined_mild`.

Other cases remain review-only: baseline__matched_linear=NO_SUPPORTED_LOCAL_CANDIDATE_REQUIRES_REVIEW; hip_stiff__matched_linear=NO_SUPPORTED_LOCAL_CANDIDATE_REQUIRES_REVIEW; knee_stiff__matched_linear=NO_SUPPORTED_LOCAL_CANDIDATE_REQUIRES_REVIEW; heavy_leg__matched_linear=NO_SUPPORTED_LOCAL_CANDIDATE_REQUIRES_REVIEW.

## E. Supported versus unsupported

Supported (`n=99,350`): mean/P95 absolute J error `0.00106573` / `0.00570122`, sign agreement `97.6417%`, false-improvement rate `0.132864%`. Unsupported (`n=89,875`): `0.000381935` / `0.00243488`, `99.1655%`, `0%`.

## F. Distance from support

Across all diagnostic maps, formal-grid distance had Spearman association `-0.34073` with absolute J error and `-0.0235118` with false-improvement occurrence. The distance metric is descriptive grid geometry, not a physical or safety threshold.

## G. Existing 90% domain-coverage gate

At the closest observed coverage levels below and above 90%, the mean across-case error step (above minus below) was `-5.20428e-05` and sign-accuracy step was `0.0022916`. This does not establish a causal or sharp reliability discontinuity. The 90% gate remains unchanged.

## H. Candidate ingredients for a future reliability rule

Support state, continuous coverage, distance to support, independent validation e_J, local rank consistency, local regret, and false-improvement history all remain candidate factors. No rule, weights, or cutoffs were selected.

## I. Precision error versus local decision ranking

The validation torque error and local decision outcomes are reported side-by-side in the scenario summary. Any mismatch case with a structure-limitation diagnosis but favorable local diagnostic label demonstrates that precise dynamics adequacy and local decision utility are distinct questions; it still is not approved.

## J. False improvements

There were `132` points where the diagnostic model predicted improvement while virtual truth was neutral or worse under the existing 0.005 research equivalence band. Exact cases are preserved in `false_improvement_cases.csv`.

## K. Reliability status

No model-reliability threshold was frozen: `NOT_FROZEN_REQUIRES_REVIEW`.

## L. Initial-identification acceptance

The acceptance rule remains `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW` and was not modified.

## M. Human-ready model status

There is no formal or human-ready theta_hat_0. Every model in this stage is `DIAGNOSTIC_ONLY`, not approved for personalization, and not human-ready.

## N. Execution status

No personalization, explore/exploit action, robot connection, or trajectory execution occurred.

## O. Tests

The final pytest counts and runtime are reported by the task handoff after the complete suite is run; they are not fabricated into this offline artifact before that run.
