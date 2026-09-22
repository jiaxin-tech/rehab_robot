# Adaptive Physics-Prior Trust Personalization V1

## Rule-freeze declaration

The primary trust rule below was specified before any adaptive-trust P0-P3
benchmark was run. Its machine-readable source is
`personalization/adaptive_trust_v1/PRIMARY_TRUST_RULE_V1.json`. Formal outputs
must record and verify that file's SHA-256. Benchmark results may be appended to
this document, but the primary rule and constants must not be changed in
response to those results.

Status at rule freeze:

```text
OFFLINE_ALGORITHM_DEVELOPMENT_ONLY
NOT_HUMAN_READY
NOT_ROBOT_APPROVED
PINN_NOT_JUSTIFIED
PINN_TRAINING = 0
```

## Why trust is needed

A useful gray-box prior can rank promising V3 coordination candidates before
four observations are available, which improves low-budget sample efficiency.
A directionally wrong prior can instead steer several of the four trials toward
poor candidates. Fixed-trust Physics-Informed BO has no mechanism to distinguish
those situations.

This stage adds one component after the gray-box prediction and before adaptive
prediction/acquisition:

```text
frozen SubjectROMProfile
  -> subject-specific V3 candidate
  -> executed EpisodeObservation
  -> compare observation with its pre-observation physics prediction
  -> PhysicsPriorTrustEstimator
  -> adaptive mixture prediction and LCB acquisition
```

The trust score is only `ALGORITHMIC_PHYSICS_PRIOR_TRUST_WEIGHT`. It is not the
probability that a model is biologically correct, patient confidence, clinical
confidence, safety confidence, or evidence about comfort or pain.

## Primary trust rule: frozen before benchmark

`PRIMARY_CAUSAL_OFFSET_INVARIANT_RANKING_TRUST_V1` is deterministic and uses
only valid, already executed trials. Let the physics prediction captured before
observing trial `i` be `m_i`, its endpoint be `y_i`, and its declared endpoint
uncertainty be `s_i`.

### Calibration disagreement

First remove a constant magnitude bias:

```text
e_i = y_i - m_i
b = median(e_i)
e_i_centered = e_i - b
```

For two or more valid observations:

```text
S = max(range(y), range(m), 4*median(s), 1e-6)
D = sqrt(mean(e_i_centered^2)) / S
C_cal = exp(-D)
```

With fewer than two valid observations, `C_cal=1`. Thus a single point cannot
identify ranking quality, and a constant offset does not by itself remove trust.

### Pairwise ranking consistency

For every pair of valid executed observations `(i,j)`, the pair is comparable
only when:

```text
abs(y_i-y_j) > 2*sqrt(s_i^2+s_j^2)
and
abs(m_i-m_j) > 1e-12
```

A comparable pair is concordant when:

```text
(y_i-y_j) * (m_i-m_j) > 0
```

With `n_pair` comparable and `n_concordant` concordant pairs:

```text
C_rank = (n_concordant + 0.5) / (n_pair + 1.0), if n_pair > 0
C_rank = 1.0,                                otherwise
```

The fixed half-count prevents one pair from producing an extreme 0 or 1
estimate under K=4.

### Trust score

The primary score is frozen as:

```text
lambda = clip(0.75*C_rank + 0.25*C_cal, 0, 1)
```

Ranking receives the larger weight because a prior with a constant magnitude
bias may still be useful for candidate ordering. No P0/P1/P2/P3 label, optimum,
unexecuted endpoint, future observation, or oracle value enters this rule.

Invalid observations keep the previous trust unchanged, add no evidence, and
retain the existing policy that the trial consumes K.

## Cold start and causal timing

The single fixed cold-start policy for every case is `lambda_1=1`. It means the
algorithm intentionally starts from its physics-prior design, not that current
subject evidence has established validity.

The timing convention is:

```text
lambda_1: fixed before Trial 1 (Trial 1 is beta=[0,0])
Trial k observation: compared only with the prediction captured before y_k
trust_after Trial k: lambda_(k+1), used to select Trial k+1
```

Therefore selection of Trial `k` uses only Trials `1...k-1`. After Trial 4, the
updated score is used for the final model recommendation but cannot alter any
already executed trial.

## Adaptive predictive model

Three options were considered before benchmarking.

- A weighted physics-prior mean GP is coherent only if every historical
  residual target is rebuilt whenever lambda changes. It does not automatically
  recover the existing centered Standard GP at lambda zero.
- A predictive mixture keeps the existing Standard BO and fixed
  Physics-Informed BO components internally unchanged and has exact endpoint
  behavior at lambda zero and one.
- A new complex trust-specific learner was rejected because K=4 supplies too
  little evidence.

The primary implementation is the predictive mixture. Fit, using the same
causal valid history:

```text
p_BO(y|beta) = Normal(mu_BO, sigma_BO^2)
p_PI(y|beta) = Normal(mu_PI, sigma_PI^2)

p_lambda(y|beta)
  = (1-lambda) p_BO(y|beta) + lambda p_PI(y|beta)
```

The exact mixture moments are:

```text
mu_lambda = (1-lambda) mu_BO + lambda mu_PI

sigma_lambda^2
  = (1-lambda) [sigma_BO^2 + (mu_BO-mu_lambda)^2]
    + lambda   [sigma_PI^2 + (mu_PI-mu_lambda)^2]
```

Candidate selection uses the existing deterministic lower-confidence-bound
rule on these moments:

```text
LCB_lambda = mu_lambda - 1.5*sigma_lambda
```

This is a moment-based acquisition from a finite predictive mixture. It is not
presented as a single exact Gaussian-process posterior.

At `lambda=0`, both moments and LCB equal Standard BO exactly. At `lambda=1`,
they equal fixed Physics-Informed BO exactly. The residual GP inside the fixed
physics-informed component continues to use `y_i-m_i`; its target definition is
not changed when trust changes.

`lambda < 0.5` is frozen as the reporting definition of a
standard-component-dominant fallback selection. It is a descriptive frequency,
not a safety or clinical threshold.

## ROM and ledger integration

Adaptive trust starts only after a frozen `SubjectROMProfile` and operates on
the subject-specific 625-member V3 domain. It cannot write or resize ROM. All
paired methods must carry one identical ROM fingerprint. A changed ROM closes
the old V2 episode and starts a new profile/domain/episode as already specified.

The existing `PERSONALIZATION_TRIAL_LEDGER` is extended per entry with:

```text
physics_prediction_before_observation
prediction_residual
physics_prior_trust_before
physics_prior_trust_after
trust_evidence_summary
```

The ROM calibration ledger remains separate. No parallel primary Stage 1
ledger is introduced.

## Frozen development evaluation plan

The formal benchmark will reuse the existing frozen analytic truth landscapes,
P0/P1/P2/P3 prior definitions, zero/low/moderate noise settings, deterministic
seeds `0...4`, 625 beta candidates, and K=4. Standard BO, fixed Physics-Informed
BO, and Adaptive-Trust Physics BO will be paired on identical truth, ROM,
candidate domain, seed, and canonical-candidate noise realization.

The following generic trust-semantic cases are also frozen before execution:

- magnitude bias: `m=y+2`, ordering unchanged;
- scaling bias: `m=1.8*y-0.7`, ordering unchanged;
- ranking inversion: `m=-y`, ordering reversed;
- one noisy point: one valid residual without pair evidence;
- invalid point: no trust evidence.

Primary success is not defined as beating every method. The question is whether
negative transfer from P3 is reduced while most informative-prior benefit is
retained under the same K=4 budget.

## Benchmark results

The frozen rule was evaluated only after its JSON artifact and unit tests were
in place. The formal run contains 240 paired conditions (4 analytic truth
landscapes x 4 prior qualities x 3 noise levels x 5 seeds), three methods per
condition, and exactly K=4 personalization trials per method. All methods used
the same frozen `ROM_MEDIUM` fingerprint and 625-member candidate domain within
each comparison. There were zero robot actions, zero human actions, zero oracle
access before post-run scoring, and zero PINN training steps.

The historical Standard BO and fixed Physics-Informed BO aggregate results were
recomputed through the subject-specific domain and matched the frozen V1
artifacts exactly: 96 compared aggregate metrics had maximum absolute error 0.

### Low-noise P0-P3 summary

Mean final regret is reported below. Benefit retention is
`(Standard - Adaptive) / (Standard - Fixed)`; it is descriptive when the fixed
prior is beneficial. W/T/L compares Adaptive against Standard over 20 paired
runs.

| Prior | Standard | Fixed physics | Adaptive trust | Final trust | Benefit retained | Adaptive vs Standard W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.107655 | 0.003625 | 0.006245 | 0.8931 | 97.48% | 20/0/0 |
| P1 | 0.107655 | 0.003248 | 0.003730 | 0.7830 | 99.54% | 20/0/0 |
| P2 | 0.107655 | 0.034941 | 0.043094 | 0.6733 | 88.79% | 12/0/8 |
| P3 | 0.107655 | 0.180462 | 0.163312 | 0.5537 | not interpreted as retained benefit | 6/0/14 |

Across zero/low/moderate noise, P1 retained 100.93%, 99.54%, and 88.46% of
the fixed-prior benefit; P2 retained 86.73%, 88.79%, and 112.77%. A value above
100% means Adaptive happened to outperform Fixed in that aggregate, not that a
new source of physical evidence was obtained.

### Negative-transfer audit

For P3, mean negative transfer relative to Standard BO changed as follows:

| Noise | Fixed physics | Adaptive trust | Reduction (`Fixed - Adaptive`) | Final trust |
|---|---:|---:|---:|---:|
| zero | +0.051559 | +0.017062 | +0.034497 | 0.5946 |
| low | +0.072807 | +0.055658 | +0.017149 | 0.5537 |
| moderate | +0.088721 | +0.094836 | -0.006115 | 0.5510 |

Thus the frozen rule reduces mean P3 negative transfer in zero and low noise,
but does not remove it and makes it worse in moderate noise. Its P3 low-noise
mean trust path is `[1.0000, 0.7580, 0.4925, 0.5537]`; with only four trials,
the estimate does not reliably become Standard-dominant.

Adaptive trust was worse than both Standard BO and fixed Physics BO in 17 of
240 paired conditions: P0/moderate (1), P1/moderate (3), P2/zero (5),
P2/low (4), P2/moderate (2), P3/low (1), and P3/moderate (1). This failure
inventory is retained as formal evidence and was not used to tune the rule.

The frozen semantic checks produced final trust 0.90625 for constant magnitude
bias, 0.86477 for scaling bias, 0.20424 for ranking inversion, 1.0 after one
noisy point without pair evidence, and 1.0 after an invalid point that supplied
no evidence. These match the intended distinction between calibration offset
and directional ranking failure.

## Formal answers

**Q1. What exact evidence updates physics trust?** Only valid, already executed
endpoint observations, their declared uncertainties, and the matching physics
predictions captured before those observations. The rule combines
offset-removed normalized prediction disagreement with uncertainty-filtered
pairwise ranking consistency. Invalid, future, unexecuted, oracle, and P0-P3
label information cannot update trust.

**Q2. What is the mathematical predictive model after adding trust?** It is the
finite predictive mixture
`p_lambda=(1-lambda)p_BO+lambda*p_PI`. Acquisition uses its exact mixture mean
and variance, including between-component disagreement, in the existing
`mean - 1.5*standard_deviation` LCB. It is not claimed to be one exact GP
posterior.

**Q3. Does trust=0 recover Standard BO behavior?** Yes. The implementation
short-circuits to the unchanged Standard component; candidate sequence and
final recommendation are exactly equal in the frozen limit test.

**Q4. Does trust=1 recover/approach fixed Physics BO?** It recovers it exactly,
including candidate sequence and final recommendation in the frozen limit
test.

**Q5. How does trust behave under magnitude bias only?** A constant +2 physics
prediction bias retains trust 0.90625 because median-offset removal finds zero
centered disagreement and all three comparable pairs have correct ordering.

**Q6. How does trust behave under ranking inversion?** It falls to 0.20424
after three observations because all three comparable pairs are discordant.

**Q7. Under P1/P2, how much Physics BO benefit is retained?** Under low noise,
P1 retains 99.54% and P2 retains 88.79%. Across zero/low/moderate noise the
fractions are P1: 100.93%/99.54%/88.46%; P2:
86.73%/88.79%/112.77%.

**Q8. Under P3, is negative transfer reduced?** Partly. Mean reduction is
0.03450 at zero noise and 0.01715 at low noise, but moderate-noise negative
transfer worsens by 0.00611. It is not eliminated at any noise level.

**Q9. Does adaptive trust ever perform worse than both baselines?** Yes, in 17
of 240 paired conditions, distributed across the seven prior/noise groups
listed in the negative-transfer audit above, including one P3 low-noise and one
P3 moderate-noise rotated-landscape run.

**Q10. Is it ready to freeze before a five-leg MuJoCo benchmark?** No. The
mechanism is implemented and causally audited, but P3 moderate noise regresses,
fallback is unreliable within K=4, and the 17 worse-than-both cases require a
new separately preregistered algorithm revision before escalation.

Formal conclusion:

```text
ADAPTIVE_PHYSICS_PRIOR_TRUST_PERSONALIZATION_V1_IMPLEMENTED_WITH_LIMITATIONS
READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK = false
```

Formal outputs are under
`personalization/benchmarks/results_adaptive_trust_v1/`; the checksum manifest
covers the JSON, CSV, and exactly three required figures.

## Boundaries

- No five-leg MuJoCo cohort is created in this stage.
- No MyoLeg model, V3 operator, ROM threshold, control, hardware, or safety code
  is modified.
- No robot or human is used.
- No PINN is trained.
- Existing V1/V2 benchmarks remain historical and unchanged.
