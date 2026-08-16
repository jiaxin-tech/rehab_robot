# Data Leakage Audit

## Selection inputs

- Executed identification observations through trial `i-1` only.
- Current temporary five-parameter estimate.
- Predeclared candidate specifications and their model-only sensitivity.
- Global model constraints and the unchanged patient operational envelope.

## Prohibited inputs

- Truth subject label or truth five-parameter vector.
- Future virtual trial outcomes.
- Held-out test data.
- Mechanical personalization objective `J`.
- Final prediction-map results.

The selector function has no truth-oracle or held-out-test argument.  The
virtual oracle is called only after a candidate has been selected.  Truth is
used only to generate the post-selection observation and to describe the case
afterward.  The within-identification validation rows are sampled from trials
that have already been executed and never enter candidate ranking as future
outcomes.

## Case audit

- `baseline__matched_linear`: 2 oracle calls, selection-before-execution asserted, held-out test absent.
- `hip_stiff__matched_linear`: 2 oracle calls, selection-before-execution asserted, held-out test absent.
- `knee_stiff__matched_linear`: 2 oracle calls, selection-before-execution asserted, held-out test absent.
- `heavy_leg__matched_linear`: 2 oracle calls, selection-before-execution asserted, held-out test absent.
- `baseline__combined_mild`: 5 oracle calls, selection-before-execution asserted, held-out test absent.
- `LIMITED_ROM_VIRTUAL_SUBJECT__matched_linear`: 2 oracle calls, selection-before-execution asserted, held-out test absent.

## Stop-rule audit

The existing repository supplies numerical rank/SVD/correlation/uncertainty
metrics but no approved complete set of stopping thresholds.  Therefore the
default is `IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW`.  The values used in
this illustrative software comparison are recorded in metadata.  They are not
claimed as a selected scientific threshold and must not be promoted to a
real-subject release rule by interpreting virtual recovery results.
