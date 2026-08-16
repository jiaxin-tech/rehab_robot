# GLOBAL_RELIABILITY_CANDIDATE_ANALYSIS

Status: `GLOBAL_RELIABILITY_CANDIDATE_ANALYSIS` / rule not frozen.

This is a deterministic factor characterization, not a classifier, threshold search, or executable gate.

## Candidate factors

- model support and continuous domain coverage
- formal-grid distance to the nearest supported point
- validation-only objective error from the selected diagnostic trial
- local rank consistency and one-step local regret
- false-improvement history

## Descriptive associations

- coverage vs absolute J error Spearman: `0.392291`
- distance vs absolute J error Spearman: `-0.34073`
- coverage vs false-improvement indicator Spearman: `0.0161846`
- distance vs false-improvement indicator Spearman: `-0.0235118`
- validation e_J vs local regret across cases Spearman: `NA`
- radius-one local rank vs local regret across cases Spearman: `NA`

`NA` case-level associations mean the observed local-regret/rank outcome lacked enough variation for a correlation; it is not evidence of no relationship. These values are descriptive only. No factor cutoff or multivariable rule was selected.

## combined_mild diagnostic

- existing adequacy trend: `MODEL_STRUCTURE_LIMITATION`
- precise dynamics status: `MODEL_INADEQUATE_FOR_PRECISE_DYNAMICS`
- validation combined torque RMSE: `0.787185 N m`
- improvement-sign agreement: `95.0488%`
- false-improvement rate: `0%`
- global rank correlation: `0.998666`
- predicted-best regret: `2.777e-05`
- one-step local regret: `0`
- diagnostic local utility: `POTENTIALLY_USEFUL_FOR_LOCAL_DECISION`

Even a favorable local label would remain diagnostic-only and would not approve personalization.

## Leakage and decision boundary

Held-out final-test data was not read or used. Virtual truth was attached only after prediction, support, and predicted-best IDs were fixed. The existing 90% support gate and 0.005 algorithmic equivalence tolerance were not modified.
