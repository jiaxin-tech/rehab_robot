# MyoLeg controlled positive cohort V1

This directory contains the frozen manifest for the controlled stress test
used to validate evidence-gated personalization. It is independent of the
native/development MyoLeg V1 cohort and does not change the MuJoCo XML,
virtual-patient deltas, or any V1 result.

The manifest has two predeclared scenarios:

- `COMMON_OPTIMUM` is a null control. Every profile returns the native
  response landscape, so a personalization gate should remain inactive except
  at its declared false-positive rate.
- `DIVERGENT_OPTIMA` is a positive software control. Arms A and B apply
  opposite, smooth candidate-space response fields to the native response.
  The field is fixed before algorithm results are inspected and is not a
  physiological or patient heterogeneity claim.

Each split contains four common IDs, four arm-A IDs, and four arm-B IDs. Across
the 24 IDs there are only three deterministic response fields (common, A, B):
the recorded seed does not enter the gain field. Therefore `CONFIRMATORY` is a
reserved frozen-protocol replication split, not an independent subject or
landscape-generalization set. It must remain sealed until the algorithm, gate
thresholds, and analysis code are frozen. The learner receives only the
response for its requested candidate through `ControlledCohortBackend`; the
manifest and full candidate truth stay outside the learner.

The policy-level smoke runner is
`lower_limb_sim.myoleg_benchmark.controlled_positive_experiment`; its output is
a synthetic analytical stress test, not native MyoLeg physiology or a safety
validation. A native follow-up must reuse the frozen MyoLeg cache and report its
own scope.

Regenerate or verify the manifest with:

```powershell
.\.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.controlled_positive_cohort
```

The manifest fingerprint is checked whenever profiles are loaded.
