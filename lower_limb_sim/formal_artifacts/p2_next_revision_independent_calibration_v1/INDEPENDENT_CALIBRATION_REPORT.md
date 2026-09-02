# Independent calibration report

Calibration manifest SHA-256: `08f930692704c24f10f85f094eabf45fc5e0842ec3f479345e62bee892df1729`.

## Direct residual distributions

- 1-step: P90=0.00127184341262, P95=0.00196762280892, P99=0.0036585815112, max=0.00724074485354, n=324
- 2-step: P90=0.00127764040456, P95=0.00164217053717, P99=0.00338315735589, max=0.00378029590608, n=216
- 3-step: P90=0.0018866659079, P95=0.00244544326845, P99=0.00519358473251, max=0.00556871183843, n=216
- 5-step: P90=0.00319594940929, P95=0.00400496043747, P99=0.00792314538021, max=0.00893051968704, n=216

The old development one-step P95 was `0.000430956758924` and is retained only as `DEVELOPMENT_ESTIMATE_ONLY`. The independent P95 is 4.56571 times the old value, so the old estimate is descriptively optimistic for this new, broader matched/mismatch calibration cohort. The comparison label is `INDEPENDENT_P95_HIGHER_THAN_DEVELOPMENT`. No pooled distribution or threshold is created.

The one-step source plan contains canonical positive orientation only. Negative one-step direction summaries are an explicitly labelled symmetric reverse view of the same 324 pairs, not 324 additional independent samples. Bundle positive/negative directions are independently preregistered pairs.

## Matched versus mismatch

- 1-step MATCHED: P95=3.52384788016e-14, max=6.10622663544e-14, n=162
- 1-step MISMATCH: P95=0.0029738006789, max=0.00724074485354, n=162
- 2-step MATCHED: P95=2.84217094304e-14, max=3.37507799486e-14, n=108
- 2-step MISMATCH: P95=0.00231378184495, max=0.00378029590608, n=108
- 3-step MATCHED: P95=4.25104396129e-14, max=4.88498130835e-14, n=108
- 3-step MISMATCH: P95=0.00356075890304, max=0.00556871183843, n=108
- 5-step MATCHED: P95=7.19035941898e-14, max=8.39328606617e-14, n=108
- 5-step MISMATCH: P95=0.00600146151555, max=0.00893051968704, n=108

These are descriptive empirical comparisons only. The five-parameter model is not modified.

## Scale, axis, and direction

Bundle P95 increases strictly from 2 to 3 to 5 steps: `true`. P95 across 1/2/3/5 is strictly increasing: `false`; the 2-step P95 is below the one-step P95 because the frozen one-step and bundle plans have different trust-step and orientation composition. No universal scale law is inferred.

- 2-step: lowest axis/direction P95=phase/POSITIVE 6.05628291924e-05; highest=knee/POSITIVE 0.00290197801541.
- 3-step: lowest axis/direction P95=phase/NEGATIVE 8.23596406916e-05; highest=knee/NEGATIVE 0.00442544393001.
- 5-step: lowest axis/direction P95=phase/POSITIVE 0.000154368687225; highest=knee/NEGATIVE 0.00733312624632.

The dominant descriptive heteroscedasticity is by axis (knee largest, phase smallest), while positive/negative bundle directions within an axis are comparatively similar. No heteroscedasticity pass/fail threshold or axis-specific uncertainty threshold is created.

## Decision-scale conclusion

Research-calibrated bundle scales by complete-design criteria: `[2, 3, 5]`. This allows a later research uncertainty-candidate design, not a policy. Residual scale is reported empirically; no formula, percentile, cumulative rule, or stopping rule is selected.

Evidence is sufficient to enter a separate `NEXT_REVISION_POLICY_DESIGN` task: **YES**, but it is not sufficient to freeze or enable a policy in this task.

The result is calibration-only. P2 V1 and rejected V2A remain unchanged, no P2 V3 exists, no prospective personalization ran, held-out final test was not read, and no robot or human approval is implied.
