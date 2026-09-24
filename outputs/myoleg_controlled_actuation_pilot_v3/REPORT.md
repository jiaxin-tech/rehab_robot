# CONTROLLED_ACTUATION_V3 development pilot

Controlled torque subtraction on prescribed-state MyoLeg traces; no forward actuation, muscle recruitment, patient or hardware validation.
No adaptive algorithm was run. All endpoints use the same subject's unassisted duration=1 reference.

- subjects / assisted candidates: 3 / 27
- valid traces: 81
- unique constrained oracles: 2
- shared feasible candidates: 7
- E3 rank Spearman median / min: 0.981685 / 0.942613
- best shared assisted candidate: CONTROLLED_ACTUATION_V3:8
- shared-candidate relative regret median / max: 0 / 3.58864e-05
- shared policy including no-assistance fallback: CONTROLLED_ACTUATION_V3:8
- relative regret with no-assistance fallback median / max: 0 / 3.58864e-05
- decision: HOLD_NO_PRACTICAL_PERSONALIZATION_SIGNAL

The common candidate was selected with complete pilot truth, for diagnostic comparison only; it is not an independently fitted policy.
The previously documented reference-based rule is exploratory. V3 additionally requires a >=0.5% gap over the best shared feasible policy in >=20% of pilot subjects, allowing no assistance for both shared and individual choices.
A pilot pass permits further development, never immediate confirmation: algorithm comparison, null/positive controls and independent protocol freezing remain outstanding.

| Subject | Feasible | Oracle | Oracle E3 | Shared relative regret |
| --- | ---: | --- | ---: | ---: |
| MYOLEG_VP_001 | 7 | CONTROLLED_ACTUATION_V3:8 | 0.998397 | 0 |
| MYOLEG_VP_002 | 9 | CONTROLLED_ACTUATION_V3:12 | 0.999259 | 3.58864e-05 |
| MYOLEG_VP_003 | 8 | CONTROLLED_ACTUATION_V3:8 | 0.998796 | 0 |

Fixed strength means a 2 Nm total L1 peak, not fixed torque impulse. Joint impulse and signed work are reported in landscape.csv.
Assistance direction follows public reference movement, not the sign of hidden subject torque; it can increase required load.
V1 30-seed expansion remains paused. No sealed subjects were accessed.
