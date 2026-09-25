# Physical interaction development pilot

Native MyoLeg prescribed-state torque plus a declared external spring/damper and fixed 2 Nm assistance.
This is a mechanical attachment sensitivity experiment, not a patient model, forward simulation or hardware measurement.
Candidate timing/share change assistance only. Actual dq in rad/s already includes duration scaling.
Each configuration uses its own unassisted duration=1 reference including the same external attachment.

Native bases: 3; declared attachments per base: 4; responses: 324.
The crossed configurations share native bases. Do not treat them or candidate rows as independent patients; no population CI is claimed.
Shared policy and oracle use the complete development landscape for screening only; neither is a tested learner recommendation.

| Native base / attachment | Feasible / 27 | Oracle | Shared-policy regret, with unassisted fallback |
| --- | ---: | --- | ---: |
| MYOLEG_VP_001::DAMPING | 11 | CONTROLLED_ACTUATION_V3:15 | 0.289159% |
| MYOLEG_VP_001::HIP_SPRING | 12 | CONTROLLED_ACTUATION_V3:24 | 0.284293% |
| MYOLEG_VP_001::KNEE_SPRING | 18 | CONTROLLED_ACTUATION_V3:24 | 0.466442% |
| MYOLEG_VP_001::ZERO_INTERACTION | 7 | CONTROLLED_ACTUATION_V3:8 | 0.000000% |
| MYOLEG_VP_002::DAMPING | 10 | CONTROLLED_ACTUATION_V3:7 | 0.198223% |
| MYOLEG_VP_002::HIP_SPRING | 10 | CONTROLLED_ACTUATION_V3:25 | 0.023424% |
| MYOLEG_VP_002::KNEE_SPRING | 17 | CONTROLLED_ACTUATION_V3:24 | 0.381205% |
| MYOLEG_VP_002::ZERO_INTERACTION | 9 | CONTROLLED_ACTUATION_V3:12 | 0.003589% |
| MYOLEG_VP_003::DAMPING | 10 | CONTROLLED_ACTUATION_V3:7 | 0.249538% |
| MYOLEG_VP_003::HIP_SPRING | 10 | CONTROLLED_ACTUATION_V3:7 | 0.241673% |
| MYOLEG_VP_003::KNEE_SPRING | 18 | CONTROLLED_ACTUATION_V3:24 | 0.457066% |
| MYOLEG_VP_003::ZERO_INTERACTION | 8 | CONTROLLED_ACTUATION_V3:8 | 0.000000% |

Maximum shared-policy relative regret: 0.466442%.
Overall screening decision: HOLD_NO_PRACTICAL_PERSONALIZATION_SIGNAL.
The predeclared practical gap is 0.5%, required in at least 20% of configurations, with a shared feasible candidate and distinct oracles.
Equivalent planar force mapping valid for all traces and baselines: True.
Force mapping uses the declared 0.42/0.30 m two-link geometry and a 500 N numerical screen; it is not measured cuff force or a clinical limit.

Separate native-base contrasts within each fixed attachment:
- DAMPING: max shared-policy regret 0.013843%; HOLD_NO_PRACTICAL_PERSONALIZATION_SIGNAL.
- HIP_SPRING: max shared-policy regret 0.266316%; HOLD_NO_PRACTICAL_PERSONALIZATION_SIGNAL.
- KNEE_SPRING: max shared-policy regret 0.000000%; HOLD_NO_PRACTICAL_PERSONALIZATION_SIGNAL.
- ZERO_INTERACTION: max shared-policy regret 0.003589%; HOLD_NO_PRACTICAL_PERSONALIZATION_SIGNAL.

No algorithm comparison was run: a useful mechanical signal is a prerequisite, not an assumed outcome.
No measurement noise or repeatability claim is made. Confirmatory data remain unused; V1 expansion paused; robot motion NO-GO.
