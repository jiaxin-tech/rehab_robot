# Predictive-Evidence Physics-Prior Failover V1

## Status and scope

`PREDICTIVE_EVIDENCE_PHYSICS_PRIOR_FAILOVER_V1_IMPLEMENTED_WITH_LIMITATIONS`

This is pure offline algorithm-development evidence. It is not robot, human,
comfort, safety, clinical, or effectiveness validation. The subject-specific
ROM, V3 candidate domain, five-effective-parameter gray-box semantics, GP
implementations, K=4 budget, and the two existing BO experts are unchanged.

`READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK = false`

## Why arbitration replaces continuous trust mixing

Adaptive Mixture Trust V1 continuously combines the Standard BO and
Physics-Informed BO predictive moments. Under K=4, an intermediate trust value
can continue to give a misleading physics prior substantial influence over the
acquisition. V1 therefore keeps that method as a historical baseline and uses
explicit model arbitration as the new primary mechanism: remain with the fixed
Physics-Informed BO expert or fail over permanently to the fixed Standard BO
expert.

No posterior mean or variance mixture is used by the new method.

## Frozen primary rule

The complete rule is stored in
`personalization/predictive_failover_v1/PRIMARY_FAILOVER_RULE_V1.json` and was
fixed before the P0-P3 benchmark.

- Initial mode: `PHYSICS_BO`.
- Primary budget: K=4.
- First scored observation: Trial 2.
- Predictive variance floor: standard deviation 0.05.
- Failover threshold: `T = -ln(10) = -2.302585092994046`.
- Trigger: after a valid observation, fail over if `E_k < T`.
- Direction: one-way, from Physics BO to Standard BO.
- Invalid observation: consumes budget but adds no score or evidence and cannot
  trigger failover.

The threshold means that cumulative predictive evidence favors Standard BO by
more than 10:1 on the predictive-density scale. It was selected from this score
interpretation and generic semantic cases, not from P0-P3 final regret.

## Strict prequential scoring

For Trial k, both unchanged experts are first evaluated at the already selected
candidate using only `D_(k-1)`. The ledger freezes:

```text
beta_k
standard_bo_mean_before
standard_bo_std_before
physics_bo_mean_before
physics_bo_std_before
```

Only after those values are stored is `y_k` revealed. Neither expert is refit
with `y_k` before scoring that observation.

For model M, the Gaussian log predictive density is

```text
S_M(k) = -0.5 * [log(2*pi*v_M,k) + (y_k-mu_M,k)^2/v_M,k]

v_M,k = max(sigma_M,k^2 + sigma_observation,k^2, 0.05^2)
```

Thus the score includes both predictive mean and uncertainty and remains finite
for a zero reported model variance.

### Causal physics offset

To avoid treating a constant physics magnitude bias as immediate prior failure,
the score uses a running median offset derived only from earlier valid trials:

```text
b_(k-1) = median(y_i - mu_PI,i), i < k
adjusted physics scoring mean = mu_PI,k + b_(k-1)
```

`b_0=0`. The raw Physics-Informed BO prediction is retained in the ledger; the
offset changes only its predictive score, not either expert's acquisition or
posterior.

The causal evidence increment and cumulative statistic are

```text
Delta S_k = S_PI(k) - S_BO(k)
E_k = sum(Delta S_i, i=2..k)
```

Positive evidence favors Physics BO. Negative evidence favors Standard BO.
Trial 1 supplies no model-comparison score, although its frozen forecast error
can causally calibrate Trial 2. The earliest failover is therefore after the
Trial 2 observation and before Trial 3 selection.

## Expert behavior

Before failover, the next candidate is selected by the existing
`PhysicsInformedResidualModel` and existing LCB selector. After failover, the
next candidate and final model recommendation use the existing
`StandardGaussianProcess` and existing LCB selector on the current legal
history. Physics has no acquisition weight and no tie-breaking role after the
switch. Tests reconstruct each active expert from the same history and confirm
exact candidate and acquisition equality.

## Generic semantic cases

The preregistration tests passed for accurate physics, constant offset,
scaling mismatch, ranking inversion, one uncertain noisy outlier, and an invalid
observation. They also verify deterministic cumulative evidence, numerical
stability, no oracle inputs, one-way switching, K=4, frozen-ROM enforcement, and
duplicate protection. These cases are algorithm semantics only, not patient
evidence.

## Formal P0-P3 benchmark

The paired benchmark reused the frozen ROM, V3 domain, P0-P3 definitions, four
truth landscapes, zero/low/moderate noise, seeds 0-4, noise realization, and
K=4 budget. It contains 960 method runs (240 paired conditions x four methods).
All 144 checked aggregate metrics for Standard BO, Fixed Physics BO, and
Adaptive Mixture Trust V1 exactly match the frozen historical benchmark.

### Informative prior: P0/P1/P2

Failover occurred in 0 of 180 informative-prior episodes. Consequently, the
new method exactly matched Fixed Physics BO in every P0/P1/P2 noise cell and
retained 100% of its positive mean-regret benefit over Standard BO. This is
descriptive offline benchmark behavior; it does not prove general calibration.

### Misleading prior: P3

| Noise | Standard mean regret | Fixed Physics | Adaptive V1 | Failover V1 | Failover count | Median trial | W/T/L vs Standard | Failover negative transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero | 0.126198 | 0.177757 | 0.143260 | 0.217080 | 10/20 | 2.5 | 5/0/15 | 0.090882 |
| low | 0.107655 | 0.180462 | 0.163312 | 0.198121 | 12/20 | 2.0 | 6/1/13 | 0.090467 |
| moderate | 0.111985 | 0.200706 | 0.206821 | 0.148524 | 11/20 | 2.0 | 8/3/9 | 0.036539 |

The P3 failover-trial counts were:

- zero noise: Trial 2 = 5, Trial 3 = 5, missed = 10;
- low noise: Trial 2 = 7, Trial 3 = 5, missed = 8;
- moderate noise: Trial 2 = 9, Trial 3 = 2, missed = 9.

Relative to Adaptive Mixture V1, failover reduced mean P3 moderate-noise
negative transfer by 0.058296 (from 0.094836 to 0.036539). It did not generalize
across noise levels: negative transfer increased by 0.073819 at zero noise and
0.034809 at low noise. P3 failover occurred in 33 of 60 episodes, leaving 27
missed failovers.

An exact post-failover Standard BO decision uses the history collected before
the switch, which was partly chosen by Physics BO. It is therefore not the same
counterfactual history as a run that used Standard BO from Trial 1. This explains
why exact local recovery of Standard BO does not guarantee the Standard BO
baseline's final regret.

## Limitations and decision

The method cleanly preserves the informative-prior baseline and improves the
specific P3 moderate-noise limitation, but it performs worse than Adaptive
Mixture V1 and Fixed Physics BO for P3 zero and low noise. It also misses
failover in 45% of P3 episodes. The requested scientific question therefore has
a mixed, not affirmative, answer under K=4.

Because misleading-prior improvement is not stable across all three noise
levels, the algorithm is implemented with limitations and is not ready to
freeze for the five-leg MuJoCo benchmark.

Detailed machine-readable results are under
`personalization/benchmarks/results_predictive_failover_v1/`.

Targeted validation of the new method plus the directly related Physics BO V1,
ROM-gated V2, and Adaptive Trust V1 regressions completed with `73 passed`.
