# FINAL_PERSONALIZATION_RESEARCH_MAINLINE_V1

## 1. Frozen mainline decision

```text
FINAL_SIMULATION_MAINLINE_FROZEN = true

SIMULATED_MECHANICAL_PERSONALIZATION_NECESSITY =
NOT_SUPPORTED

PERSONALIZATION_ARCHITECTURE =
EVIDENCE_GATED

NEXT_PRIMARY_STAGE =
REAL_MEASUREMENT_VALIDATION
```

This conclusion is limited to:

```text
current five-leg MuJoCo benchmark
current V3 fixed-ROM family
current E2 simulated mechanical endpoint
current passive mechanical formulation
```

It does not imply that real patients do not need personalization. It means only that the current frozen simulation does not provide a sufficiently strong, stable, decision-relevant target for coordination personalization.

## 2. Paper-ready research question

> Can subject-specific measurements reveal decision-relevant trajectory-dependent mechanical differences within a fixed-ROM rehabilitation task, and, when such evidence exists, can a low-budget model-informed sequential strategy identify a mechanically preferable coordination trajectory?

The research architecture does not assume that every subject must have a different optimum. A common/reference trajectory remains the default unless real measurements support a meaningful subject-by-trajectory interaction.

## 3. Frozen system layers

### 3.1 Stage 0: subject-specific ROM

The retained subject-specific layer is:

```text
subject-specific ROM determination
→ freeze ROM
```

ROM personalization and coordination personalization are distinct:

- **ROM personalization** determines the subject-appropriate task domain before trajectory comparison. It remains part of the main architecture.
- **Coordination personalization** chooses among trajectories inside the frozen ROM. It is currently inactive and requires real measurement evidence.

Freezing ROM prevents reachability and range differences from being silently treated as evidence for different coordination preferences. Retaining Stage 0 does not itself establish safety, clinical suitability, or an individualized mechanical optimum.

### 3.2 V3 trajectory family

`P4_BRANCH_AWARE_COORDINATION_FUNCTION_V3` is frozen as the fixed-ROM coordination family. No V4/V5 family will be created to manufacture oracle diversity.

V3 is retained because the frozen simulation shows that it produces measurable kinematic and model-derived mechanical differences. This makes it suitable for structured real trajectory-response probing. The same evidence also shows that V3 does not automatically create meaningful individual optimum diversity in simulation.

### 3.3 Simulated endpoints and representations

The frozen endpoint roles are:

| Item | Frozen role | Boundary |
|---|---|---|
| E2: `BRANCH_BALANCED_REFERENCE_NORMALIZED_RMS` | `PRIMARY_SIMULATED_MECHANICAL_ENDPOINT` | Model-derived and not validated as a real-measurement endpoint |
| E1 | Diagnostic endpoint | Retained for offline interpretation, not promoted to the primary endpoint |
| 8D feature vector | Diagnostic representation | Preserves joint, branch, RMS, and peak information for analysis |
| E0 | Historical baseline | Demonstrates compression from full-cycle aggregate RMS |

E2 remains useful because it prevents hip/knee and flexion/extension RMS cancellation and exposes branch-limiting behavior. Endpoint search stops here: no E5/E6 or further synthetic endpoint tuning will be introduced. A future real-robot endpoint must be separately grounded in validated, synchronized, repeatable measurements; E2 is not transferred directly by name alone.

### 3.4 Existing personalization methods

The following implementations are retained without further modification:

```text
Standard BO
Fixed Physics-Informed BO
Adaptive Mixture Trust V1
Predictive Failover V1
Active Diagnostic Arbitration V1
Repeated Diagnostic K5 V1
```

Their status is `OFFLINE METHODOLOGICAL IMPLEMENTATIONS`. Continuing to tune their selectors, priors, arbitration, trust, failover, or diagnostic budget is not scientifically justified while the target personalization benefit itself is unsupported.

## 4. Frozen simulation finding

The frozen evidence establishes the following sequence:

1. V3 creates trajectory-dependent kinematic and model-derived mechanical changes.
2. E0 strongly compresses joint-, branch-, and time-local differences.
3. E2 raises normalized leg-by-trajectory interaction energy to `5.228442%`.
4. Despite that increase, Legs 0–3 retain the reference oracle `(0,0)`.
5. Leg 4 selects `(-0.03, 0.0225)`, but the common reference costs it only `0.070684%` relative regret.
6. The cohort mean common-vs-individual regret is only `0.014137%`, with zero regret for four legs.

Therefore:

```text
SIMULATED_MECHANICAL_PERSONALIZATION_NECESSITY = NOT_SUPPORTED
```

The interaction is numerically present but does not produce a sufficiently consequential cohort-level decision difference. Synthetic oracle diversity will not be increased by adding legs or tuning parameters.

## 5. Evidence-gated personalization architecture

```text
Subject / limb
      ↓
Stage 0: subject-specific ROM determination
      ↓
freeze ROM
      ↓
V3 fixed-ROM trajectory probing
      ↓
real measured mechanical response
      ↓
measurement validity + repeatability + trajectory sensitivity
      ↓
Is decision-relevant subject × trajectory interaction supported?
      ↓
   YES                                      NO
    ↓                                        ↓
activate low-budget sequential         retain common/reference
trajectory personalization             trajectory
    ↓
gray-box model + GP/BO comparison
```

```text
PERSONALIZATION_IS_EVIDENCE_GATED = true
```

Gray-box models, Gaussian processes, and BO are downstream identification and search methods, not evidence that personalization is needed. They can be reactivated only after valid real data show repeatable trajectory sensitivity and decision-relevant cross-subject differences.

## 6. Next primary stage: real measurement validation

The next stage must answer four questions in order:

### A. Measurement validity

Are robot state and wrench signals physically credible, correctly interpreted, sufficiently synchronized, and repeatable? Until this is demonstrated, no measured mechanical endpoint is established.

### B. Episode repeatability

For repeated executions at the same frozen ROM and the same V3 beta, what is the within-condition response variability?

### C. Trajectory sensitivity

Are measured mechanical differences among selected V3 beta conditions materially larger than repeated-measurement variability and relevant uncertainty?

### D. Decision-relevant interaction

Across real subjects or limbs, do trajectory rankings, limiting branches, or common-vs-individual regret differ reproducibly enough to change the preferred trajectory decision?

Only positive evidence for D, built on A–C, can reactivate coordination-personalization algorithm comparison.

## 7. Dummy-leg role

The dummy leg is assigned only to:

```text
MEASUREMENT_PIPELINE_VALIDATION
TRAJECTORY_EXECUTION_VALIDATION
REPEATABILITY_TESTING
MECHANICAL_SIGNAL_SANITY_CHECK
```

It can reveal acquisition, synchronization, execution, repeatability, and basic mechanical-signal problems. It cannot establish patient personalization necessity, patient-specific optimum diversity, comfort, clinical safety, or treatment effectiveness.

## 8. MuJoCo and MyoLeg role

```text
MyoLeg / five-leg MuJoCo =
OFFLINE PHYSICS / METHOD DEVELOPMENT SUPPORT
```

They remain useful for mechanics reasoning, controlled method development, failure-mode exploration, and offline regression. They will not be expanded with additional synthetic subjects, retuned parameters, or manufactured oracle diversity to support a patient-personalization claim.

## 9. Current claim boundary

### Claims currently allowed

- V3 produces distinguishable kinematic and model-derived mechanical responses in the frozen simulation.
- E2 is a more mechanically interpretable simulated diagnostic endpoint than E0 for the stated branch-balanced purpose.
- E2 exposes more simulated leg-by-trajectory interaction than E0.
- The present five-leg passive MuJoCo benchmark does not support decision-relevant coordination personalization necessity.
- Stage 0 subject-specific ROM remains a separate and retained architectural layer.
- Existing gray-box/GP/BO methods are offline methodological implementations available for future evidence-triggered comparison.

### Claims currently not allowed

- Real patients do not need personalization.
- Real subjects share the same mechanically preferable trajectory.
- Current wrench/state measurements are physically valid, synchronized, or repeatable without real evidence.
- E2 is already a validated real-robot or clinical endpoint.
- Dummy-leg results prove patient personalization necessity.
- Synthetic parameters are physiological or tissue parameters.
- Lower simulated mechanical metrics imply comfort, safety, clinical benefit, or treatment effectiveness.
- Any existing personalization algorithm is ready or justified for deployment because it performs well on the synthetic benchmark.

## 10. Q1–Q10 final answers

**Q1. 当前 simulation 能证明什么？**

It shows that V3 produces model-observable trajectory differences, E0 compresses important structure, and E2 exposes a larger simulated interaction. Within the frozen five-leg passive benchmark, however, a common reference remains effectively optimal.

**Q2. 不能证明什么？**

It cannot establish real measurement validity, real-patient optimum diversity, clinical personalization necessity, comfort, safety, or treatment benefit. It also cannot establish that patients do not need personalization.

**Q3. 为什么停止继续开发 personalization algorithm？**

The present target is not decision-relevant enough: four legs gain nothing over the common trajectory and cohort mean gain is only `0.014137%`. Further selector or heuristic optimization would improve methods around an unsupported simulated need rather than answer the scientific question.

**Q4. Stage 0 ROM 为什么仍然保留？**

ROM is an independently justified subject-specific task boundary. Determining and freezing it before probing coordination separates reachability/range accommodation from within-ROM trajectory selection.

**Q5. V3 为什么仍然保留？**

V3 supplies a fixed, branch-aware family that produces distinguishable trajectories and can probe real trajectory-response sensitivity without changing ROM. Its role is experimental probing, not automatic personalization proof.

**Q6. E2 在未来项目里扮演什么角色？**

E2 remains the primary simulated mechanical diagnostic endpoint and a source of branch-specific hypotheses. It is not automatically the real-robot endpoint; real measurement features and endpoints require independent validity and repeatability evidence.

**Q7. Gray-box/GP/BO 什么时候重新启用？**

Only after valid real measurements show repeatable trajectory effects and decision-relevant subject-by-trajectory differences whose magnitude exceeds within-condition variability and uncertainty. Until then they remain frozen offline implementations.

**Q8. 假腿下一阶段具体负责验证什么？**

It validates the measurement pipeline, trajectory execution, repeated-episode variability, and basic mechanical-signal plausibility. It does not validate patient personalization.

**Q9. 哪类真实 evidence 才能重新支持 coordination personalization？**

The required evidence is reproducible cross-subject or cross-limb variation in trajectory ranking, limiting-branch response, or common-vs-individual regret, measured with a physically credible and synchronized pipeline, and clearly larger than repeatability noise and uncertainty.

**Q10. 当前最终论文/研究架构是什么？**

First determine and freeze subject-specific ROM. Then use frozen V3 trajectories to collect valid real mechanical responses and quantify repeatability, sensitivity, and subject-by-trajectory interaction. Retain a common/reference trajectory when that interaction is not decision-relevant; activate low-budget gray-box/GP/BO personalization only when real evidence supports it.

## 11. Stop decisions

```text
STOP_NEW_SYNTHETIC_PERSONALIZATION_EXPANSION = true
STOP_NEW_TRUST_FAILOVER_HEURISTICS = true
STOP_NEW_ENDPOINT_SEARCH = true
STOP_V4_V5_FOR_ORACLE_DIVERSITY = true
PINN_NOT_JUSTIFIED = true

FINAL_SIMULATION_MAINLINE_FROZEN = true
SIMULATED_MECHANICAL_PERSONALIZATION_NECESSITY = NOT_SUPPORTED
PERSONALIZATION_ARCHITECTURE = EVIDENCE_GATED
NEXT_PRIMARY_STAGE = REAL_MEASUREMENT_VALIDATION
```

This document closes the simulation-personalization mainline. It does not authorize robot motion or start the real-measurement stage.
