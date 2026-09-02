# Task Direction Definition Audit

## Decision

`TASK_DIRECTION_REQUIRES_EXPERIMENTAL_VALIDATION`

Direction must be selected from mechanics and a prespecified validation, never from whichever direction produces the smallest RMS.

| Candidate | Mechanical meaning | Posture/measured-state dependence | Sign/noise | Relation to strap/task | Decision |
|---|---|---|---|---|
| A instantaneous TCP tangent | component along instantaneous executed path | posture-aware; computable after validated TCP timing | undefined near zero speed and reversals; differentiation is noisy | motion direction is not necessarily strap tension direction | retain diagnostic candidate, not primary definition |
| B strap/pull line of action | force along the physical load-transfer line | needs registered attachment endpoints or a validated direct direction measurement | stable if attachment is taut; sign can be fixed physically | closest to actual strap mechanics | preferred physical semantics, but evidence missing |
| C endpoint-to-hip direction | straight line from traction point to registered hip | posture-dependent; hip registration and traction point required | potentially stable away from zero length | plausible only if it represents the real taut strap | validation candidate |
| D fixed bed-plane axis | fixed component in a reviewed bed frame | independent of posture and easy to compute | stable and noise-robust | can miss changing line of action | secondary diagnostic only |
| E 2-DOF equivalent traction direction | line from modeled hip to `L2` strap-equivalent traction point | computable from q and frozen geometry after patient/robot registration | stable if registration valid | consistent with formal lower-limb model but not a measured ankle | validation candidate, never relabel `L2` as ankle |

The defensible target is B: the measured/registered strap pull line of action, with C/E usable only if an experiment proves that their registered geometry represents B throughout the task. Until that evidence exists, no production `d_task(t)` is frozen.

Future rules: express force and the unit direction in the same validated frame; normalize after checking finite nonzero norm; predefine direction orientation (for example hip-to-traction-point) and independently establish force sign. Log the signed projection. RMS is sign-invariant mathematically, but signed projection and resistive/assistive decomposition remain diagnostics and must not be created by an arbitrary absolute value.
