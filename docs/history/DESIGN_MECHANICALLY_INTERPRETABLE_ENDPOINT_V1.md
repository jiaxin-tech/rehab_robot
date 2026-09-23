# Mechanically interpretable endpoint design V1

## Scope and evidence boundary

This is an offline synthetic MuJoCo endpoint-design study. It preserves the
five frozen legs, five SubjectROMProfiles, V3 trajectory family and beta grid,
gray-box model, historical E0 definition, and all personalization algorithms.
No BO was run, no PINN was trained, and no robot interface was used.

All current joint-load features are derived from the frozen MuJoCo
inverse-dynamics model. The conclusions concern simulated mechanical endpoint
design only; they are not physical, robot, human, comfort, safety, or clinical
validation.

## Endpoint definitions fixed before characterization

No continuous weights were fitted or searched.

| ID | Fixed definition | Mechanical meaning |
|---|---|---|
| E0 | historical full-cycle dual-joint vector RMS | existing aggregate baseline |
| E1 | `max(relative full hip RMS, relative full knee RMS)` | minimize the worse relative joint RMS |
| E2 | `max(relative hip-flex RMS, hip-extend RMS, knee-flex RMS, knee-extend RMS)` | minimize the worse relative joint-by-branch RMS |
| E3 | `max(relative full hip peak, relative full knee peak)` | minimize the worse relative joint peak |
| E4 | maximum of four relative joint-by-branch peaks | minimize the worse relative joint-by-branch peak |

Every relative term uses the same leg's `beta=(0,0)` reference. Cohort-mean
normalization is not used.

The uncompressed `MECHANICAL_ENDPOINT_FEATURE_VECTOR_V1` is retained as:

```text
[
  r_hip_flex_RMS,
  r_hip_extend_RMS,
  r_knee_flex_RMS,
  r_knee_extend_RMS,
  r_hip_flex_peak,
  r_hip_extend_peak,
  r_knee_flex_peak,
  r_knee_extend_peak
]
```

Full hip/knee RMS and peak values are also retained separately.

## Q1. Where does E0 lose information?

All three mechanisms contribute, with joint and branch cancellation providing
the clearest scalarization evidence and time averaging further suppressing
localized effects.

- Across 5 x 625 cases, 2,331 candidates contain an opposing hip/knee RMS
  trade-off. Of these, 1,366 still lie within +/-0.1% of the E0 reference.
- Each leg contains 312 opposing flexion/extension trade-offs, for 1,560 total;
  1,173 remain within +/-0.1% of the E0 reference.
- The largest difference between E1's worst-joint result and E0's relative
  result is 0.57--1.61 percentage points across legs.
- The largest analogous E2 branch-aware gap is 1.05--1.77 percentage points.
- From the preceding discriminability audit, total mechanical separation stays
  above half maximum for only 7.5--14.2% of the trajectory; the most sensitive
  10% of samples contain 47.2--52.4% of integrated spread.

Joint cancellation is the most prevalent; branch cancellation produces the
largest representative minimax penalty; full-cycle time averaging dilutes both.

Representative Leg 1 example at `beta=(+0.03,-0.03)`:

- E0 changes by only +0.0189%;
- hip RMS improves by 0.0183%;
- knee RMS worsens by 1.1803%;
- knee-extension RMS worsens by 1.4589%;
- E1 reports 1.011803 and E2 reports 1.014589 instead of treating the case as
  mechanically unchanged.

## Q2. Does reference normalization remove magnitude scaling?

Yes for the intended comparison. E1--E4 all equal one at each leg's own
reference despite E0 reference loads ranging from about 32 to 320 N m. Thus
absolute leg magnitude cannot dominate their joint/branch comparison.

After normalization, trajectory-dependent cross-leg interaction becomes
visible:

| Endpoint | Normalized interaction RMS | Interaction energy fraction | Minimum pairwise Spearman |
|---|---:|---:|---:|
| E0 | 0.0128 | 0.36% | 0.9920 |
| E1 | 0.0671 | 8.15% | 0.4003 |
| E2 | 0.0677 | 5.23% | 0.8147 |
| E3 | 0.1313 | 8.77% | 0.5857 |
| E4 | 0.1313 | 8.77% | 0.5900 |

The E3/E4 interaction values are not accepted at face value because their
absolute beta ranges are extremely small and their rankings fail the
perturbation study.

## Q3. Does E1 preserve more candidate discrimination?

Yes, especially in the previously most compressed legs.

- E1 range/reference is 0.882--1.218% across all five legs, compared with
  E0's 0.036--2.166%.
- E1 coefficient of variation is 0.207--0.285%, compared with E0's
  0.0078--0.460%.
- E1 produces five distinct exact oracles and lowers minimum pairwise Spearman
  to 0.400, but these facts characterize interaction and are not the reason to
  select it.
- Its five-leg universal near-optimal counts are 49/477/610/625/625 at
  0.1/0.5/1/2/5%.

E1 directly exposes real simulated hip-knee RMS trade-offs and remains useful
as a joint-balanced diagnostic.

## Q4. Does E2 reveal additional flexion/extension structure?

Yes. E2 has the most direct anti-cancellation representation for the evidence
identified in the preceding audit.

- Range/reference is 1.121--1.598%; coefficient of variation is
  0.340--0.479%.
- Universal near-optimal counts are 8/228/500/625/625 at 0.1/0.5/1/2/5%.
- Legs 0--3 select the unchanged reference; Leg 4 selects
  `(-0.03,+0.0225)` with four numerically tied minima.
- Minimum pairwise Spearman is 0.815 and minimum normalized gradient cosine is
  0.875, showing decision-relevant but not arbitrary cross-leg shape changes.

The reference optimum in four legs has a clear meaning: within the current V3
family, moving away from reference improves some joint/branch channels only by
worsening another. E2 refuses that compensation. This is a valid conservative
mechanical result, not a failure to manufacture different oracles.

Of 975,000 within-leg candidate pairs, the count whose scalar endpoint differs
by at most 0.1% while at least one feature-vector element differs by at least
0.5% is:

- E0: 406,543;
- E1: 258,068;
- E2: 104,907;
- E3/E4: 817,965 each.

E2 loses the least information under this fixed descriptive comparison, though
the full feature vector remains more informative than any scalar.

## Q5. Do E3/E4 provide meaningful peak signal?

No for selection within the current V3 family.

- E3 and E4 range/reference is only 0.000124--0.0482%.
- Every candidate on every leg lies within 0.1% of the peak endpoint optimum.
- The fraction of branch samples at or above 99% of the joint peak is typically
  10.9--27.0%, so the underlying high-load regions are not merely one-sample
  numerical spikes.
- However, candidate-to-candidate peak differences are much smaller than the
  fixed perturbations. E3/E4 minimum baseline-versus-perturbed ranking Spearman
  is -0.802; exact oracle stability is only 60%/50%.

Thus peak regions are mechanically real, but V3 preserves the endpoint/peak
regions too strongly for peak scalarization to discriminate candidates. E3/E4
mostly add ranking sensitivity, not stable decision information.

## Robustness study

The fixed sensitivity study used two deterministic torque waveforms with
amplitude 0.25% of each leg reference joint peak and two deterministic
half-rate sampling grids. It is not a validated sensor-noise model.

| Endpoint | Minimum ranking Spearman | Median ranking Spearman | Exact oracle stability | Maximum endpoint change |
|---|---:|---:|---:|---:|
| E0 | 0.999987 | 0.999997 | 100% | 0.00736% |
| E1 | 0.999945 | 0.999995 | 95% | 0.00179% |
| E2 | 0.999250 | 0.999988 | 100% | 0.01852% |
| E3 | -0.802327 | 0.0725 | 60% | 0.04638% |
| E4 | -0.802327 | 0.0717 | 50% | 0.04638% |

E1's one exact oracle change has only `1.25e-5` baseline relative regret. E2
keeps every exact oracle under all 20 leg-by-perturbation cases.

## Q6. Which endpoint has the clearest mechanical interpretation?

E2. It asks one explicit question: what is the worst relative RMS load among
hip/knee and flexion/extension for this leg, relative to its own reference?
It directly matches the observed failure modes, does not average them away, and
uses no fitted weights. E1 is simpler but does not protect branch-specific
trade-offs.

## Q7. Is there decision-relevant subject-by-trajectory interaction?

Yes in the five frozen synthetic mechanical systems.

- E2 interaction energy rises from E0's 0.36% to 5.23%.
- E2 has two oracle structures rather than E0's universal boundary oracle.
- Pairwise ordering and gradient similarity remain substantial, but are no
  longer almost identical.
- E1 shows still stronger ordering variation, but oracle diversity is not used
  as the selection rule.

This supports reevaluating personalization necessity with the fixed E2
representation in a future, separate stage. It does not yet prove that a
personalization algorithm is necessary or effective.

## Q8. Is the interaction a mechanical trade-off or scalar-weight artifact?

It is a simulated mechanical trade-off. E2 is the maximum of four individually
observable model-derived ratios; it contains no fitted or continuous weights.
Representative candidates explicitly improve one joint/branch RMS while
worsening another, and E2 reports the worsening channel. The five endpoint
definitions were fixed before oracle characterization.

## Q9. Is a candidate justified as the next primary simulated endpoint?

Yes, with limitations: E2 is recommended as the next primary **simulated
mechanical** endpoint.

The recommendation is based on mechanical interpretation, joint-and-branch
anti-cancellation, uniform discriminability, deterministic perturbation
stability, and absence of outcome-driven weights. It is not based on maximizing
oracle diversity. E0 must remain as the historical baseline, E1 and the full
feature vector should remain diagnostic companions, and E3/E4 should not be
promoted.

Limitations:

- E2 is still a scalar and cannot preserve every feature-vector relationship.
- Five hundred candidates remain universally within 1% of the E2 optimum.
- This result covers only five frozen synthetic mechanical systems and V3.
- The perturbations are sensitivity probes, not measured sensor noise.
- E2 inputs are currently model-derived and lack physical validation.

## Real-measurement compatibility

| Quantity | Status | Boundary |
|---|---|---|
| Current E0--E4 and joint/branch load features | `MODEL_DERIVED` | derived from frozen MuJoCo inverse dynamics |
| Future calibrated robot wrench in a verified task direction | `DIRECTLY_MEASURABLE` | reference point, frame, timing and task-direction semantics still require physical validation |
| Wrench-to-joint or gray-box joint-load estimates | `MODEL_DERIVED` | require validation against independent physical load evidence |
| Contact pressure/distributed interface load | `FUTURE_SENSOR_REQUIRED` | tactile/pressure sensing unavailable in this benchmark |

None of the redesigned endpoints is real-world validated.

## Q10. Final conclusion

`MECHANICAL_ENDPOINT_REDESIGN_V1 = SUPPORTED_WITH_LIMITATIONS`

`RECOMMENDED_PRIMARY_ENDPOINT = E2`

`READY_TO_REEVALUATE_PERSONALIZATION_NECESSITY = true`

The readiness flag authorizes only a future offline reevaluation using the
fixed E2 definition. No BO, personalization algorithm, trajectory redesign,
model change, or robot action is started here.

## Outputs and test summary

- Machine-readable study: `lower_limb_sim/five_leg_mujoco_v1/results_endpoint_design_v1/study_summary.json`
- Preserved 5 x 625 feature vectors: `mechanical_endpoint_feature_vector_v1.csv`
- Eleven focused CSV tables cover discriminability, cross-leg structure,
  anti-cancellation, robustness, measurement compatibility, and selection.
- Targeted tests: `8 passed`
- Algorithm runs: `0`; robot actions: `0`; PINN training: `0`

DESIGN_MECHANICALLY_INTERPRETABLE_ENDPOINT_V1 = COMPLETE

MECHANICAL_ENDPOINT_REDESIGN_V1 = SUPPORTED_WITH_LIMITATIONS

RECOMMENDED_PRIMARY_ENDPOINT = E2

READY_TO_REEVALUATE_PERSONALIZATION_NECESSITY = true
