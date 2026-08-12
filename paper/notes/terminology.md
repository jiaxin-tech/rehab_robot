# Terminology control

## Preferred terms

| Use | Meaning in this manuscript |
|---|---|
| subject-specific equivalent dynamics | Task-local gray-box input–output model; not physiological ground truth. |
| equivalent passive stiffness | Constant linear coefficient estimated over the excited task region. |
| equivalent passive damping | Constant linear coefficient estimated over the excited task region. |
| passive lower-limb rehabilitation | Prescribed motion without a claim of active voluntary assistance. |
| supine hip–knee flexion | The frozen physical scenario. |
| reference trajectory | Prescribed closed rehabilitation path. |
| reference-local excitation | Small amplitude, phase, or timing variations around a reference. |
| task-local identification | Identification whose validity is limited to the excited state region. |
| candidate trajectory | Screened alternative near the reference; not automatically optimal. |
| mechanical interaction load | Modeled/observed endpoint force or generalized torque, with evidence level stated. |
| mechanical interaction residual | Difference between observed and model-predicted interaction; physical use remains TODO. |
| strap equivalent traction point | Point at which the planar robot-on-leg force is modeled to act. |
| model-based candidate screening | Offline rejection/ranking using the equivalent model and explicit constraints. |

## Fixed symbols and physical meaning

| Symbol | Meaning |
|---|---|
| `q_h` | Hip flexion angle. |
| `q_k` | Knee flexion angle. |
| `theta_s = q_h - q_k` | Absolute shank orientation; never replace with a sum convention. |
| `L1` | Thigh length. |
| `L2` | Knee-to-strap-equivalent-traction-point distance; not automatically ankle distance. |
| `F_{R->L}` | Point force exerted by robot on leg in the modeled sagittal frame. |
| `tau_meas = J^T F_{R->L}` | Generalized interaction torque reconstructed from the point force. |
| `q_ref(s)` | Prescribed closed reference; may be asymmetric. |
| `q_cand(s; alpha)` | Reference-local candidate; current continuous optimization is not finalized. |

## Prohibited or evidence-gated terms

Do not use the following as completed claims without new evidence:

- true patient dynamics / true patient stiffness / true tissue stiffness;
- optimal patient trajectory;
- comfortable trajectory / comfort score;
- clinically optimal / clinically safer;
- clinical effectiveness / improved rehabilitation outcome;
- real-time validated, robot-validated, dummy-validated, or human-validated;
- tactile–wrench fusion as a completed contribution.

Simulation force values and the 1000 N software anomaly bound are not robot
safety thresholds. Offline tests and fake adapters are not physical evidence.
