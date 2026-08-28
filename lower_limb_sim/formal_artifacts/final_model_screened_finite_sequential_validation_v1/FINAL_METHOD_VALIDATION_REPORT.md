# FINAL_MODEL_SCREENED_FINITE_SEQUENTIAL_VALIDATION_V1

Manifest SHA-256: `7576e5a545878292f2eb1846e9cae780325a2e44bb58093dfb04bae982827498`

## A–G. Method in plain language

The subject first completes the unchanged initial identification protocol. The
five-parameter model predicts J for all 21,025 geometrically valid grid points,
then freezes at most three supported candidates before candidate truth is read.
Each round validates one complete trajectory, refits the model, recomputes the
full landscape, and reranks only the remaining frozen candidates. No C4 or
later candidate can enter. The final trajectory is the lowest measured/virtual-
truth J among the reference and actually validated candidates.

This is still “step by step,” but one step is one **whole-trajectory trial**, not
a local alpha-grid perturbation. Search is finite because the persisted
shortlist is immutable and the registered budget is at most three. The full
landscape is recomputed to diagnose model change; a new predicted optimum is
never physically/virtually added. Prediction screens, measurement decides.

## H. B0/B1/B2/B3

| Comparator | Candidate trials | Mean final J | Mean regret | False improvements |
|---|---:|---:|---:|---:|
| B0 reference | 0 | 1.000000 | 0.035042 | 0 |
| B1 predicted best only | 15 | 0.967272 | 0.002314 | 0 |
| B2 frozen Top-3 finite | 45 | 0.967272 | 0.002314 | 0 |
| B3 truth oracle | offline lower bound | 0.964958 | 0 | 0 |

## I. Budget sensitivity

- Budget 0: mean J=1.000000, regret=0.035042, false improvements=0, candidate cost=0.
- Budget 1: mean J=0.967272, regret=0.002314, false improvements=0, candidate cost=15.
- Budget 2: mean J=0.967272, regret=0.002314, false improvements=0, candidate cost=30.
- Budget 3: mean J=0.967272, regret=0.002314, false improvements=0, candidate cost=45.

These values do not change the preregistered three-candidate research budget.

## J. Matched versus mismatch

- MATCHED: mean J=0.973471, regret=0.003686, prediction error=0.000000, false improvements=0.
- MISMATCH: mean J=0.961848, regret=0.001113, prediction error=0.009733, false improvements=0.

Any mismatch gap is an `APPLICABILITY_LIMITATION`; the five-parameter model is
not changed here.

## K. Subject specificity

The old BUNDLE_5 final alpha has `1` unique full alpha; the finite
method has `5`. This difference arises from unmodified simulation
measurements, not a diversity reward or truth-driven shortlist.

## L. Trial-count comparison

Old fixed BUNDLE_5 used `116` personalization trials over
15 cases. B2 uses `45` complete candidate validations, a
`61.2%` reduction. Initial identification is excluded from
both sides as the common baseline. This is an offline architecture comparison,
not a prospective or clinical head-to-head conclusion.

## M–N. Visualizations

Nine static figures and two deterministic GIFs are generated. The mandatory
GIFs are `FINAL_METHOD_WORKFLOW_ANIMATION.gif` and
`SUBJECT_SPECIFIC_LANDSCAPE_COMPARISON.gif`.

## O–P. Final interpretation

Final status: `FINAL_SIMPLIFIED_METHOD_SUPPORTED_WITH_LIMITATIONS`

`STOP_FURTHER_P2_EXPANSION = true`

The evidence remains `OFFLINE_ONLY`, `NOT_HUMAN_READY`, and
`NOT_ROBOT_APPROVED`. If supported with limitations, the next work is method
freeze, manuscript Method/Offline Results writing, and separately governed
limited fixed-candidate physical validation—not P2 V3 or another optimizer.
