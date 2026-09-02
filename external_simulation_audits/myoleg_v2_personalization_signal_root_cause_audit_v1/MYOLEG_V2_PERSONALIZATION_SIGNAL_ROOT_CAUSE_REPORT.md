# MyoLeg V2 Personalization Signal Root-Cause Audit V1

## Overall conclusion

`PERSONALIZATION_SIGNAL_ROOT_CAUSE_IDENTIFIED`

This is a development-only offline mechanistic audit under the unchanged frozen MyoLeg-V2 truth semantics, candidate domain, reference normalization, and normalized RMS objective. It is not a new objective, learner, patient optimum, human result, comfort result, safety result, or clinical result. All eight held-out truth landscapes remained sealed.

## Hypothesis decisions

| Hypothesis | Decision | Core frozen metric |
|---|---:|---|
| H1 normalization cancellation | PARTIALLY_SUPPORTED | hip attenuation=0.595795; knee=0.196509 |
| H2 weak subject×trajectory interaction | SUPPORTED | J interaction variance=0.000331143; interaction/common=0.0182029 |
| H3 RMS objective compression | PARTIALLY_SUPPORTED | time/RMS interaction: hip=1.87254; knee=9.13953 |
| H4 candidate-domain monotonicity | SUPPORTED | boundary support={'hip': 1.0, 'knee': 1.0, 'phase': 1.0} |

## Q1 — Candidate main effect versus subject interaction

For J, candidate main effect accounts for **99.939077%** of centered landscape variance, subject main effect **0.027809%**, and subject×candidate interaction **0.033114%**. The interaction/common-effect RMS ratio is **0.0182029**.

## Q2 — Multiplicative or affine landscape relationship

`MULTIPLICATIVE_SUBJECT_SCALING_SUPPORTED`. Median affine R² is **0.999992065** for hip and **0.999940044** for knee. Median pure-proportional NRMSE is **0.000222862** and **0.000135531**, respectively.

## Q3 — Normalization attenuation

The preregistered dimensionless interaction attenuation ratio after subject-specific reference normalization is **0.595795** for hip and **0.196509** for knee. This diagnoses cancellation; it does not justify removing normalization.

## Q4 — Local descent around reference

All 24 development subjects share the same combined-J local descent direction around the reference: **hip +, knee +, phase −**, with sign agreement **1.000** in every dimension. Hip-only has the same three directions; knee-only instead prefers **hip −, knee +, phase −**, exposing a hip-axis joint trade-off that the combined J nevertheless resolves in the hip-positive direction. Boundary-step support at the frozen common oracle is hip **1.000**, knee **1.000**, and phase **1.000**. No out-of-domain derivative or optimum is claimed.

## Q5 — Boundary censoring

H4 is **SUPPORTED**. Pooled global majority-sign fractions are hip **1.000000**, knee **1.000000**, and phase **1.000000**. The scientifically allowed statement is that the frozen optimum is boundary-limited/censored if supported—not that an unknown true optimum lies outside the domain.

## Q6 — Hip/knee trade-off

Opposing hip/knee adjacent-transition fractions are hip-axis **1.000000**, knee-axis **0.000000**, and phase-axis **0.000000**. The frozen 5% diagnostic classifies these as: hip=False, knee=True, phase=True for `NO_MEANINGFUL_HIP_KNEE_OBJECTIVE_TRADEOFF`.

## Q7 — Time-resolved dynamics versus RMS

Time-resolved/RMS-summary interaction ratios are hip **1.87254** and knee **9.13953**. Sign-changing candidate-minus-reference waveform fractions are **1.000000** and **1.000000**. H3 is therefore **PARTIALLY_SUPPORTED** under the pre-frozen rule. This remains diagnostic and does not define a time-weighted objective.

## Q8 — Force-component mechanisms

The largest replay-subset subject-main fractions are `bias_gravity_knee`=0.998388, `bias_gravity_hip`=0.992116, and `zero_control_actuator_hip`=0.76112. The largest interaction/common-effect ratios are `bias_gravity_hip`=0.085655, `mass_hip`=0.0612786, and `mass_knee`=0.0607405. These are MuJoCo required-drive components, not physiological tissue contributions.

## Q9 — Supported hypotheses

- H1: **PARTIALLY_SUPPORTED**
- H2: **SUPPORTED**
- H3: **PARTIALLY_SUPPORTED**
- H4: **SUPPORTED**

All-candidate development rank-inversion rate has median **0.005138963**, P95 **0.011168629**, and maximum **0.016046054**. Exploratory parameter-mechanism associations with BH q<0.05: **7**; no predictive model was trained.

## Q10 — Defensible next branch

The next branch must follow the supported mechanisms rather than tune an objective to manufacture oracle diversity. A supported H4 motivates an independently preregistered **trajectory-parameterization/boundary audit**; a supported H1 or H3 motivates a separate **objective-formulation audit**; failure of meaningful interaction after both audits supports stopping personalization and reframing the study. No branch is executed here.

## Access and scope boundary

Exactly 24 development compact landscapes and the preregistered development replay subset were used. Held-out shards were only existence/size/stream-SHA checked. Held-out NPZ scientific array loads, J/oracle/rank/torque/component access: **0**. No Five-parameter model, NN, PINN, BO, objective redesign, normalization change, cohort/range change, candidate change, robot, or hardware operation occurred.
