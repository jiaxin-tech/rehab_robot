# Identifiability Stop Rule Audit

The repository already computes numerical rank, singular values, condition
number, parameter correlation, information diagonals, covariance-shaped
uncertainty, optimizer standard errors, and residual metrics.  It does not
contain an approved complete conjunction of rank, conditioning, correlation,
per-parameter uncertainty/sensitivity, and validation thresholds for a new
subject.

Therefore the authoritative state is
`IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW`.  The numeric rule recorded in
`metadata.json` is an illustrative virtual-research comparator used only to
test early-stop and fail-closed behavior.  It is not a human, clinical, robot,
or safety release rule, and this artifact makes no scientific threshold-
selection claim.  Runtime selection never reads held-out test results.
