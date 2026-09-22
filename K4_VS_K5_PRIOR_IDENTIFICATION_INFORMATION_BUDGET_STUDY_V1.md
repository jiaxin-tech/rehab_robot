# K4 vs K5 Prior-Identification Information-Budget Study V1

## Status and evidence level

`K4_VS_K5_PRIOR_IDENTIFICATION_INFORMATION_BUDGET_STUDY_V1_COMPLETED`

`scientific_conclusion = EVIDENCE_MIXED`

`FREEZE_PRIMARY_PERSONALIZATION_BUDGET = NONE`

`READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK = false`

This is pure offline algorithm-development evidence. It is not robot, human,
comfort, safety, clinical, or effectiveness validation. The frozen P0-P3
landscapes, subject-specific ROM, 25x25 V3 beta domain, truth functions, physics
priors, noise definitions, and paired seeds were reused without modification.
No robot action, PINN training, or five-leg MuJoCo benchmark was performed.

## Frozen comparison

The existing `ACTIVE_PRIOR_DIAGNOSTIC_ARBITRATION_V1` K=4 algorithm was not
modified:

```text
Trial 1: Reference
Trial 2: Active diagnostic from D1
         -> score Delta S_2 -> arbitrate once
Trials 3-4: Fixed selected expert
```

The only new variant is
`REPEATED_ACTIVE_DIAGNOSTIC_ARBITRATION_K5_V1`:

```text
Trial 1: Reference
Trial 2: Active diagnostic #1 from D1
Trial 3: Active diagnostic #2 from D2 over remaining candidates
         -> E_3 = Delta S_2 + Delta S_3 -> arbitrate once
Trials 4-5: Fixed selected expert
```

Both diagnostic candidates maximize the same average symmetric Gaussian KL
score with the same 0.05 standard-deviation floor and causal predictive
uncertainty semantics. Trial 3 excludes the Trial 1 and Trial 2 candidates and
is selected only after Trial 2 is observed. Both experts remain live until
Trial 3 is observed.

Each `Delta S_t = S_PI,t - S_BO,t` uses distributions frozen before the
corresponding observation. Valid scores are summed without re-scoring. An
invalid diagnostic consumes its budget, contributes no evidence, and is not
retried. The frozen arbitration margin remains `ln(10)`:

```text
E_3 >  ln(10) -> PHYSICS_BO
E_3 < -ln(10) -> STANDARD_BO
otherwise     -> INCONCLUSIVE -> PHYSICS_BO
```

No second switch is allowed after arbitration.

## Formal benchmark and historical equivalence

The paired study contains 240 conditions and 1,440 runs: four truth cases,
P0-P3, three noise levels, seeds 0-4, and six methods (Standard and Fixed
Physics at K=4/K=5, Active Diagnostic K=4, and Repeated Active Diagnostic K=5).
All methods used the same frozen ROM fingerprint and 625-candidate V3 domain.
Post-run oracle values were used only by the evaluator to calculate regret.

The 144 checked K=4 aggregate metrics match the frozen Active Diagnostic V1
artifact exactly within `1e-12`; maximum absolute error is 0.0.

## P0-P2: information cost and informative-prior retention

Values are mean final regret and mean best-seen regret over 20 episodes per
cell. `Phys` and `FR` are the K=5 Physics-selection and false-rejection rates.
K=4 selected Physics in all 180 informative-prior episodes and therefore had
zero false rejection throughout.

| Prior | Noise | K4 final | K4 best seen | K5 final | K5 best seen | K5 Phys | K5 FR |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 | zero | 0.000992 | 0.031109 | 0.000992 | 0.050102 | 100% | 0% |
| P0 | low | 0.005103 | 0.033018 | 0.006389 | 0.053415 | 100% | 0% |
| P0 | moderate | 0.019783 | 0.034769 | 0.035494 | 0.073237 | 100% | 0% |
| P1 | zero | 0.015277 | 0.068737 | 0.015972 | 0.075542 | 100% | 0% |
| P1 | low | 0.011749 | 0.059128 | 0.008514 | 0.054269 | 100% | 0% |
| P1 | moderate | 0.024436 | 0.051222 | 0.018422 | 0.042439 | 100% | 0% |
| P2 | zero | 0.009392 | 0.077635 | 0.020837 | 0.063735 | 25% | 75% |
| P2 | low | 0.010366 | 0.077913 | 0.033629 | 0.062190 | 25% | 75% |
| P2 | moderate | 0.028718 | 0.074952 | 0.039654 | 0.055587 | 40% | 60% |

The extra diagnostic did not merely remove an optimization trial: it changed
the expert decision for P2. P0 and P1 retained zero false rejection, while P2
incurred 42/60 false rejections. Across informative cells where Fixed Physics
K=5 beat Standard K=5, Repeated Diagnostic K=5 retained 71.10% of the summed
positive Physics benefit, below the preregistered 75% criterion. The maximum
cell false-rejection rate was 75%, above the 25% criterion. Thus the informative
prior cost is material even though several K=5 best-seen values improved.

## P3: misleading-prior identification

`W/T/L` compares each active method with Standard BO at the same budget.
Negative transfer is mean final regret minus the corresponding Standard mean;
lower values are better.

The budget-control mean final regrets were:

| Noise | Standard K4 | Standard K5 | Fixed Physics K4 | Fixed Physics K5 |
|---|---:|---:|---:|---:|
| zero | 0.126198 | 0.041376 | 0.177757 | 0.143868 |
| low | 0.107655 | 0.041517 | 0.180462 | 0.178268 |
| moderate | 0.111985 | 0.044873 | 0.200706 | 0.181081 |

| Noise | K4 reject / miss | K4 mean / median | K4 W/T/L | K4 transfer | K5 reject / miss | K5 mean / median | K5 W/T/L | K5 transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero | 75% / 5 | 0.037207 / 0.034757 | 15/0/5 | -0.088991 | 100% / 0 | 0.035795 / 0.036313 | 10/0/10 | -0.005581 |
| low | 70% / 6 | 0.134188 / 0.073651 | 11/0/9 | +0.026533 | 100% / 0 | 0.061932 / 0.030584 | 10/0/10 | +0.020415 |
| moderate | 60% / 8 | 0.386666 / 0.184275 | 5/0/15 | +0.274682 | 85% / 3 | 0.224730 / 0.200541 | 1/0/19 | +0.179857 |

K=5 increased total P3 rejection from 41/60 to 57/60 and reduced missed
rejection from 19/60 to 3/60. It eliminated misses at zero and low noise. At
moderate noise, rejection rose from 12/20 to 17/20 and misses fell from eight
to three. Mean final regret improved by 0.161936 relative to Active Diagnostic
K=4, a 41.87% reduction.

This identification improvement did not make K=5 competitive with its proper
optimization-budget control. At moderate noise, Repeated Diagnostic K=5 lost
19/20 comparisons against Standard K=5 and retained positive negative transfer
of 0.179857. The zero and low cells each split 10 wins and 10 losses against
Standard K=5.

## Moderate-noise missed-rejection audit

Of the eight K=4 missed-rejection episodes, K=5 changed five to Standard and
introduced no new miss among episodes K=4 had rejected correctly. Mean regret
on those same eight episodes fell from 0.688874 at K=4 to 0.359822 at K=5.
The three remaining K=5 misses had mean regret 0.494298; K=5 episodes selecting
Standard had mean regret 0.177159.

All three remaining misses were inconclusive decisions that defaulted to
Physics, rather than cumulative evidence crossing the positive Physics margin:

- one had conflicting evidence and `E_3 = +0.797379`;
- one had conflicting evidence and `E_3 = -2.240057`, just above `-ln(10)`;
- one had two Standard-favoring signs but `E_3 = -2.215263`, also just above
  `-ln(10)`.

The worst K=4 episode (regret 2.727778) remained a Physics selection but its K=5
regret fell to 0.092778 because the additional diagnostic changed the sampled
history. Another remaining miss retained regret 1.035671. The catastrophic
failure concentration was reduced, not eliminated.

## Evidence consistency

Counts below are `Trial 2 Physics/Standard`, `Trial 3 Physics/Standard`, and
`consistent Physics / consistent Standard / conflicting`, out of 20 per cell.
No formal run had invalid or neutral diagnostic evidence.

| Prior | Noise | Trial 2 P/S | Trial 3 P/S | Consistent P/S | Conflicting |
|---|---|---:|---:|---:|---:|
| P0 | zero | 20/0 | 10/10 | 10/0 | 10 |
| P0 | low | 20/0 | 10/10 | 10/0 | 10 |
| P0 | moderate | 20/0 | 17/3 | 17/0 | 3 |
| P1 | zero | 20/0 | 5/15 | 5/0 | 15 |
| P1 | low | 20/0 | 10/10 | 10/0 | 10 |
| P1 | moderate | 20/0 | 11/9 | 11/0 | 9 |
| P2 | zero | 20/0 | 0/20 | 0/0 | 20 |
| P2 | low | 20/0 | 0/20 | 0/0 | 20 |
| P2 | moderate | 20/0 | 5/15 | 5/0 | 15 |
| P3 | zero | 0/20 | 0/20 | 0/20 | 0 |
| P3 | low | 0/20 | 2/18 | 0/18 | 2 |
| P3 | moderate | 1/19 | 4/16 | 0/15 | 5 |

Trial 3 strongly reinforced Standard for P3, explaining the rejection gain.
However, it also systematically opposed Trial 2 for P2 and drove the 42/60
informative-prior false rejections. This is the central evidence conflict: the
extra observation improves P3 identification but is not reliably calibrated
across the full prior-quality range.

## Diagnostic candidate behavior and information gain

Trial 2 reproduced the frozen K=4 score. P0-P2 selected the four domain corners
uniformly; P3 primarily selected `(-0.0025, 0)` or `(0.0025, 0)`, with five
moderate-noise episodes selecting `(0.03, 0.03)`.

Trial 3 was always a distinct candidate and was recomputed from D2; there were
0 exact duplicate candidates in 240 episodes. Distinct does not imply a distant
or independent region. For P3, the median beta distance between diagnostics was
0.0025 at every noise level. Using the preregistered distance threshold 0.005,
the pair was highly similar in 20/20 zero-noise and 16/20 low- and
moderate-noise episodes. Typical pairs were `(+-0.0025, 0)` followed by
`(+-0.005, 0)` or an adjacent grid point.

| P3 noise | Mean divergence Trial 2 | Mean divergence Trial 3 | Mean change | Highly similar |
|---|---:|---:|---:|---:|
| zero | 10.208856 | 8.705950 | -1.502905 | 20/20 |
| low | 9.315633 | 4.945630 | -4.370002 | 16/20 |
| moderate | 4.944918 | 3.190972 | -1.753945 | 16/20 |

Thus Trial 3 supplies a new causal observation and often changes the evidence,
but for P3 it usually probes the same local disagreement region with lower
predicted divergence. Episode-level beta values and both divergences are
retained in `diagnostic_evidence.csv`; the complete grouped audit is in
`information_gain_summary.csv`.

## Scientific and freeze decision

The preregistered identification criteria were met: total P3 misses fell by
more than half, moderate misses fell from eight to three, and moderate P3 mean
regret improved by more than 25%. The budget cannot be frozen at K=5 because:

- summed informative-prior benefit retention was 71.10%, below 75%;
- the maximum informative-cell false rejection was 75%, above 25%;
- moderate P3 negative transfer versus Standard K=5 remained 0.179857, above
  the 0.05 acceptance limit.

The result therefore supports neither a clean K=4-information-limit conclusion
nor the claim that a second diagnostic is useless. The required conclusion is:

`EVIDENCE_MIXED`

Physics-prior reliability cannot be robustly identified under the tested
low-budget/noise conditions using the current architecture. Stop adding trust,
threshold, mixture, or failover heuristics at this stage. A later project-level
decision may choose simpler Standard BO, explicitly restrict Physics BO's
applicability, or reconsider the total trial budget; none is initiated here.

## Validation and artifacts

Semantic tests cover K=5 reference/diagnostic roles, D2-only Trial 3 selection,
non-duplication, both prequential scores, exact evidence accumulation,
post-Trial-3 arbitration, fixed expert use for Trials 4-5, exact K=5 budget,
invalid-observation semantics, frozen ROM, and absence of oracle/future or robot
imports. The formal artifact test additionally freezes the mixed conclusion,
historical K=4 equivalence, key P3 effects, candidate uniqueness, and readiness
decision.

Machine-readable results are in
`personalization/benchmarks/results_k4_vs_k5_information_budget_v1/`.
