# Endpoint Repeatability Validation Plan

This is a future protocol design, not an experiment. Use one independently approved safe candidate and identical frozen setup/attachment/command conditions. The repeat count is `N_REPEATS_NOT_YET_FROZEN`; choose it prospectively from precision/reliability requirements, not from observed trajectory ranking.

Before collection freeze episode definition, wrench/task-direction semantics, bias/filter policy, sampling/synchronization gates, setup factors and run-order strategy. Report valid repeat count, mean, SD, CV where the mean makes CV interpretable, within-session drift, between-block/session bias, and ICC (with the exact ICC model) only when the design supports it. Plot time profiles and residuals without changing preprocessing after viewing results.

`ENDPOINT_REPEATABILITY_GATE` thresholds remain null until justified by measurement requirements. Fail or indeterminate repeatability blocks personalization use; it must not be repaired by removing inconvenient repeats after outcome inspection.
