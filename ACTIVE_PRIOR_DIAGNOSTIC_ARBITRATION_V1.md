# Active Prior Diagnostic Arbitration V1

## Status and evidence level

`ACTIVE_PRIOR_DIAGNOSTIC_ARBITRATION_V1_IMPLEMENTED_WITH_LIMITATIONS`

`READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK = false`

`K4_PRIOR_IDENTIFICATION_AND_OPTIMIZATION_MAY_BE_INFORMATION_LIMITED`

This is pure offline algorithm-development evidence. It is not robot, human,
comfort, safety, clinical, or effectiveness validation. The frozen
subject-specific ROM stage, 25x25 V3 beta grid, 625 candidates, gray-box
semantics, Standard BO, Fixed Physics BO, Adaptive Trust V1, and Passive
Predictive Failover V1 were not changed.

## Frozen K=4 episode

The rule was fixed in
`personalization/active_diagnostic_v1/PRIMARY_ACTIVE_DIAGNOSTIC_RULE_V1.json`
before the formal P0-P3 benchmark:

```text
Trial 1: reference beta=[0,0]
Trial 2: active prior diagnostic candidate selected from D1
         reveal y2 and arbitrate once
Trial 3: selected expert optimization
Trial 4: the same selected expert optimization
Final:   recommendation from the selected expert
```

The diagnostic trial consumes one of the four trials. No retry is made if it is
invalid.

## Diagnostic score

For each unexecuted frozen-ROM candidate, Standard BO and Fixed Physics BO
produce Gaussian predictive distributions after Trial 1. The physics mean is
adjusted only for scoring and diagnosis by the causal Trial 1 offset

```text
b1 = y1 - physics_mean_before_y1
adjusted_physics_mean(beta) = physics_mean_D1(beta) + b1
```

If Trial 1 is invalid, `b1=0`. Neither expert's posterior or acquisition is
modified.

The unique primary diagnostic score is the average symmetric KL divergence:

```text
D(beta) = 0.5 * [KL(Standard || Physics) + KL(Physics || Standard)]
beta_2  = argmax D(beta)
```

For univariate Gaussians,

```text
KL(P||Q) = 0.5 * [log(v_Q/v_P) + (v_P + (mu_P-mu_Q)^2)/v_Q - 1]
v = max(sigma^2, 0.05^2)
```

The score therefore responds to both mean disagreement and predictive
uncertainty. Ties use the lowest candidate index. Candidate selection reads
only the two D1 predictive distributions and the frozen domain; it has no
truth, case-label, optimum, regret, or future-observation input.

## Frozen Trial 2 predictions and arbitration

Before revealing `y2`, the ledger freezes beta, diagnostic score, offset, and
both experts' mean/std values. After `y2` is revealed, Gaussian predictive log
density is computed with model variance, observation uncertainty, and the same
0.05 standard-deviation floor.

```text
Delta S_2 = S_PI(y2) - S_BO(y2)
margin = ln(10)

Delta S_2 >  margin  -> PHYSICS_BO
Delta S_2 < -margin  -> STANDARD_BO
otherwise            -> INCONCLUSIVE -> PHYSICS_BO
```

An invalid Trial 2 is also inconclusive and defaults to Physics BO. This is the
only arbitration event. The chosen expert remains fixed for Trials 3-4 and the
final recommendation. Standard mode has no physics acquisition or tie-breaking
input; Physics mode calls the unchanged Fixed Physics BO expert.

## Generic semantic validation

Before the formal benchmark, tests covered accurate physics, constant offset,
scaling mismatch, ranking inversion, one noisy diagnostic observation,
identical-expert ties, invalid Trial 2, strict D1 selection, predictions frozen
before `y2`, one-time arbitration, exact expert recovery, K=4, frozen ROM, and
duplicate protection. These are algorithm cases, not patient evidence.

## Formal paired benchmark

The benchmark reused the same ROM, 625-candidate domain, four truth landscapes,
P0-P3 prior definitions, zero/low/moderate noise, seeds 0-4, noise realization,
and K=4 budget. It contains 240 paired conditions and 1200 method runs across:

```text
Standard BO
Fixed Physics BO
Adaptive Mixture Trust V1
Passive Predictive Failover V1
Active Diagnostic Arbitration V1
```

All 192 checked aggregate metrics for the four historical methods exactly match
the previous frozen benchmark.

## Which diagnostic candidates were selected?

Only six of 624 eligible Trial 2 candidates were selected:

- For P0/P1/P2, each of the four domain corners `(+-0.03, +-0.03)` was selected
  15 times per prior quality. These were the maximum-disagreement regions.
- For P3, `(-0.0025, 0)` was selected 30/60 times, `(0.0025, 0)` 25/60 times,
  and `(0.03, 0.03)` 5/60 times.

These are post-run evaluator groupings. The algorithm never receives a P-label.

## Informative-prior result and diagnostic opportunity cost

P0/P1/P2 had zero false rejection in 180 episodes: the rule always selected
Physics BO after Trial 2. Across cells where Fixed Physics BO had positive mean
benefit over Standard BO, aggregate benefit retention was 104.34%.

The diagnostic trial was not uniformly cost-free. Relative to Fixed Physics BO,
the nine informative cells had a mean final-regret difference of -0.004215
(range -0.024575 to +0.013162) and a mean best-seen-regret difference of
+0.000762 (range -0.012528 to +0.023133). In particular, all P1 cells paid a
positive final-regret cost of 0.004608 to 0.013162, while several P0/P2 cells
benefited from the diagnostic candidate itself. Values above 100% retention
therefore describe this benchmark, not a general claim that diagnosis is free.

## P3 misleading-prior result

| Noise | Standard | Fixed Physics | Adaptive V1 | Passive failover | Active diagnostic | Rejection | Missed active/passive | W/T/L vs Standard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero | 0.126198 | 0.177757 | 0.143260 | 0.217080 | 0.037207 | 15/20 (75%) | 5/10 | 15/0/5 |
| low | 0.107655 | 0.180462 | 0.163312 | 0.198121 | 0.134188 | 14/20 (70%) | 6/8 | 11/0/9 |
| moderate | 0.111985 | 0.200706 | 0.206821 | 0.148524 | 0.386666 | 12/20 (60%) | 8/9 | 5/0/15 |

Active diagnosis reduced missed rejection from 27/60 to 19/60. It improved
mean final regret over passive failover by 0.179872 at zero noise and 0.063934
at low noise. At moderate noise it was worse by 0.238142 and had negative
transfer of 0.274682 relative to Standard BO.

The moderate-noise failure is concentrated but severe: the eight episodes that
kept Physics BO had mean final regret 0.688874, versus 0.185194 for the twelve
that selected Standard BO. Even the Standard-selected runs retain the actively
chosen diagnostic observation in their history, so they are not the
counterfactual Standard BO trajectory from Trial 1.

## Scientific conclusion

Active disagreement sampling does identify misleading P3 earlier and reduces
the total missed-rejection count. It also preserves the informative-prior
advantage. However, one noisy diagnostic observation is not reliable enough at
moderate noise: the remaining wrong arbitration decisions are very costly, and
the result is less stable than Adaptive V1 or passive failover.

The K=4 budget is therefore serving two competing purposes--prior
identification and optimization--without enough repeated evidence to make both
reliable. No additional trust heuristic is added in this stage. A separate
future study may compare K=4 with K=5, but the primary K remains unchanged here.

Detailed machine-readable results are in
`personalization/benchmarks/results_active_diagnostic_v1/`.

Targeted validation of this stage plus the directly related Physics BO V1,
ROM-gated V2, Adaptive Trust V1, and Passive Predictive Failover V1 regressions
completed with `89 passed`.
