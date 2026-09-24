# MyoLeg candidate mechanism design V1

This document freezes the next candidate-space decision after the V1
candidate-space diagnostic. The purpose is to create subject--trajectory
interaction, not to increase the density of the existing beta timing grid.

## What has already been screened

The native `DURATION_COORDINATION_V2` family changes the total cycle duration
and applies a smooth knee coordination excursion. It contains 27 candidates:

| factor | frozen values | native interface |
|---|---|---|
| cycle duration scale | `0.8, 1.0, 1.2` | directly represented by `time_s`, `dq`, and `ddq` |
| coordination lag | `-0.12, 0, 0.12` phase units | directly represented by the knee trajectory |
| coordination amplitude | `-0.04, 0, 0.04` | directly represented by the knee trajectory |

The development screen found 2 unique constrained oracles among 24 subjects,
with median cross-subject E3 rank Spearman 1.000000. This is a weak signal and
does not justify a confirmatory personalization experiment by itself.

## Candidate families for the next pilot

The next pilot has two separate domains. They must not be silently merged,
because they have different physical meanings and different simulator
requirements.

### A. Native trajectory domain

Keep `DURATION_COORDINATION_V2` as the native trajectory pilot. If a second
native family is needed, vary the coordination pattern itself rather than
adding more values around the current lag/amplitude points:

| mechanism | candidate examples | purpose |
|---|---|---|
| duration | short, reference, long cycle | tests speed and motion duration |
| coordination | knee-leading, synchronous, hip-leading | tests the timing relationship between hip and knee |

Every candidate must pass the predeclared speed, acceleration, and ROM checks.
The candidate generator must remain independent of subject response values.

### B. Actuation/interaction domain

Assistance timing and load allocation cannot be evaluated by the native
prescribed-state interface: that interface returns the generalized torque
required to realize a prescribed trajectory and accepts no exogenous
assistance torque. They therefore require an explicit actuation adapter or a
controlled synthetic stress test.

The pilot should use the following three factors, with the assist magnitude
held fixed during the first comparison:

| factor | frozen pilot values | interpretation |
|---|---|---|
| assistance timing | early, middle, late within each flexion/extension branch (`0.25, 0.50, 0.75` phase) | when the assistance pulse is applied; this is not gait stance |
| hip/knee load share | hip-heavy, balanced, knee-heavy (`0.75, 0.50, 0.25` hip share) | which joint receives the assist torque |
| trajectory duration | short, reference, long (`0.9, 1.0, 1.1`) | interaction between speed and assistance |

The actuation adapter must expose only the requested candidate response. It
must record the assistance waveform, joint allocation, peak torque, and any
constraint violation. Its output must be labelled `CONTROLLED_ACTUATION` and
must not be reported as native MyoLeg physiology.

## Subject--trajectory interaction model

For an actuation candidate, let `tau_native(t)` be the native prescribed-state
torque and `a(t; z)` be a smooth assistance pulse with mechanism parameters
`z`. The controlled response is

```text
tau_net(t; z) = tau_native(t) - a(t; z)
```

The pulse is normalized to a fixed assist magnitude before comparing timing or
load share. This makes the pilot test *where* and *how* assistance is applied,
instead of allowing a larger assist magnitude to manufacture a difference.
The native trajectory and actuation domains are analysed separately and only
then combined in a predeclared interaction model.

## Go/no-go criteria

Advance a mechanism family to the frozen controlled comparison only if all of
the following hold on the development split:

1. every candidate has a valid trace and passes the declared kinematic or
   actuation limits;
2. at least two constrained oracle candidates occur across subjects;
3. the median pairwise E3 rank correlation is below `0.99`, or the
   subject-specific reference regret is at least `0.5%` of the reference E3
   in at least 20% of subjects;
4. the effect is reproduced by the held-out positive cohort and disappears in
   the common-optimum null cohort.

If these criteria fail, do not expand the grid. Keep the family as a null
mechanism diagnostic and design a different interaction factor.

## Scope boundary

The native screen supports claims about prescribed trajectories and required
generalized torque in the frozen MyoLeg model. The actuation pilot supports
claims about the evidence-gated algorithm under a declared interaction model.
Neither supports claims about patient comfort, treatment efficacy, or hardware
safety without a separate physical validation protocol.

## Implemented V3 pilot protocol

The [CONTROLLED_ACTUATION_V3 protocol](MYOLEG_CONTROLLED_ACTUATION_V3.md)
specifies the fixed 2 Nm peak budget, public movement-aligned pulse direction,
unassisted reference normalization, and development-only pilot. Before its
first run, it supplements the exploratory reference-regret criterion above
with the gap to the best shared feasible assisted candidate. This is needed
because assistance that helps all subjects equally does not establish a
personalization benefit. Pilot eligibility never opens the confirmatory split.
