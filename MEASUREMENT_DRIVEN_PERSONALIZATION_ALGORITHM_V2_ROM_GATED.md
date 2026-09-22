# Measurement-Driven Personalization Algorithm V2: ROM-Gated

## Formal status and evidence boundary

`SUBJECT_SPECIFIC_ROM_GATED_SEQUENTIAL_PERSONALIZATION_V2_IMPLEMENTED_WITH_LIMITATIONS`

This implementation is `OFFLINE_ALGORITHM_DEVELOPMENT_ONLY`,
`NOT_HUMAN_READY`, and `NOT_ROBOT_APPROVED`. It is an architecture and
software-invariant result. It is not human, clinical, comfort, pain, safety,
or robot-effectiveness evidence.

Validated pressure, force, pain, and clinical-ROM thresholds are unavailable:

```text
P_limit = NOT_AVAILABLE
validated force threshold = NOT_AVAILABLE
safety_margin_policy = null
```

The only numeric boundary gate is `SyntheticThresholdGate`, which rejects
non-`SYNTHETIC_` policy IDs and is labelled `OFFLINE_ALGORITHM_TEST_ONLY` and
`NOT_A_HUMAN_SAFETY_MODEL`.

The V1 package, its benchmark outputs under `results_v1`, the authoritative
reference artifacts, the frozen 625-member V3 table, and the V3 operator source
were not edited by V2.

## Architecture

```text
        Subject enters
             |
             v
    Stage 0 ROM determination
             |
             v
      freeze ROM profile
             |
             v
  Subject-specific reference
             |
             v
     V3 fixed-ROM domain
             |
             v
            beta_1
             |
             v
       EpisodeObservation
             |
             v
      Gray-box physics
             |
             v
        Residual GP
             |
             v
      Physics-informed BO
             |
             v
          beta_(k+1)
             ^
             | loop through K
             v
      final coordination
```

`ROM determines the task envelope.`

`BO optimizes coordination inside that envelope.`

Stage 0 and Stage 1 use separate ledgers and budgets:

```text
ROM_CALIBRATION_LEDGER
    !=
PERSONALIZATION_TRIAL_LEDGER

ROM_DETERMINATION_BUDGET
    !=
PERSONALIZATION_ADAPTATION_BUDGET
```

The primary Stage 1 budget remains `K=4`; `K=3` and `K=5` remain sensitivity
settings. Stage 1 Trial 1 cannot start until `ROM_PROFILE_FROZEN=true` and is
always the subject-specific `[beta_flex,beta_extend]=[0,0]` reference.

## Stage 0: current task-specific allowed ROM

`SubjectROMProfile` is a frozen, versioned record containing:

- profile identity and version;
- independent hip/knee bounds for this first implementation;
- ROM status and provenance;
- boundary evidence, threshold-policy IDs, and calibration episode IDs;
- an optional safety-margin policy, with the current default `null`;
- extension points for a coupled-joint constraint, configuration-validity
  mask, and trajectory-specific constraint;
- immutable metadata and a content fingerprint.

Its semantic scope is `CURRENT_TASK_SPECIFIC_ALLOWED_ROM`. It does not represent
a person's complete ROM or any physiological ground truth.

`ROMBoundaryObservation` carries the observed joint configuration, hip/knee
angles, optional force/pressure/tracking features, validity, raw gate status,
external-stop request, threshold-policy identity, measurement quality, and
metadata. `ROMSafetyGate.evaluate(observation)` is the pluggable boundary.
Future validated pressure, force, robot/joint, tracking-validity, and manual
stop gates can implement this interface without changing Stage 1.

Tactile pressure is not a pain label, not a comfort label, and not an
automatically validated ROM limit. A future pressure gate is admissible only
after calibration, unit verification, repeatable placement, and independent
scientific/safety justification of its threshold.

`ROMDeterminationController` is a fail-closed offline state machine:

1. Evaluate only the current observation through the configured gate.
2. Retain a SAFE observation as `last_safe_configuration`.
3. On STOP, invalid measurement, missing synthetic feature, or external stop,
   record `first_stop_configuration` and close further expansion.
4. Use the first and last defensibly SAFE configurations as the observed lower
   and upper bounds, and freeze them without interpolation.

For `SAFE, SAFE, SAFE, STOP`, the STOP configuration is evidence but is never
adopted as an executable ROM boundary. After STOP, another observation is
rejected. `EXTERNAL_STOP` is a stop signal only; it is not converted to a
comfort label.

`UnavailableValidatedSafetyGate` always stops because validated real thresholds
do not exist. `RealROMDeterminationInterface` always raises
`REAL_ROM_DETERMINATION_DISABLED: NOT_ROBOT_APPROVED` and imports no motion or
hardware interface.

## Subject-specific reference construction

`SubjectSpecificReferenceAdapter` reads but does not edit the authoritative V3
parent reference. For joint `j`, it preserves normalized progression:

```text
u_j(t) = (q_j,source(t) - q_j,source,min)
         / (q_j,source,max - q_j,source,min)

q_j,subject(t) = q_j,subject,min
                 + u_j(t) (q_j,subject,max - q_j,subject,min)
```

Because each target span is positive, this affine mapping preserves source
branch timing, normalized progression, step signs, closure, and smoothness.
Velocity and acceleration are scaled by the same joint-specific positive scale.
Time, global phase, segment phase, phase rate, phase acceleration, and branch
labels are unchanged. The adapter audits extrema, closure, duration, normalized
progression, and the absence of pointwise clipping before returning an immutable
subject reference.

This is a task-envelope construction rule, not a claim that different subjects
execute identical absolute `q(t)`.

## Frozen V3 coordination domain inside the subject ROM

The existing operator is reused directly:

```text
w_b(s; beta_b) = s + beta_b * 64 * s^3 * (1-s)^3
```

The canonical beta grid remains 25 by 25, from -0.03 to +0.03 at 0.0025 steps,
for 625 candidates. The V3 source function is called on the subject-specific
reference; the original subject-independent absolute arrays are not blindly
reused.

Every generated candidate must pass, or the domain build fails without clipping:

- hip and knee minima/maxima equal the frozen subject ROM within 2e-5 rad;
- duration, endpoints, branch anchors, closure, and C2 anchor values are
  preserved;
- warp derivative is positive and neither branch folds;
- all arrays are finite and no pointwise clipping is used.

Candidate identity now has two levels. `canonical_beta_id` retains the frozen
V3 grid identity. `subject_candidate_id` hashes the ROM profile ID, profile
version and fingerprint, canonical beta identity and values, V3 operator
version, and subject-reference version. Equal beta values under different ROM
profiles therefore do not imply equal absolute trajectories or equal IDs.

## Stage 1: unchanged causal V1 core

Once ROM is frozen, `ROMGatedPersonalizationEpisode` delegates to the existing
V1 `run_sequential_personalization` implementation. The selector, standard GP,
residual Matern-5/2 GP, acquisition, observation validity policy, and causal
trial ledger are not rewritten:

```text
y(beta) = y_physics(beta; theta_hat) + r(beta)
r(beta) ~ GP
```

`SubjectSpecificFullDynamicsGrayBoxEndpointAdapter` preserves the existing five
effective gray-box parameter semantics and fitting rule. It changes only the
trajectory source: the physics prediction looks up the actual subject candidate
and evaluates its `q(t)`, `dq(t)`, and `ddq(t)`. These effective parameters are
not reinterpreted as physiological or tissue parameters.

Stage 1 has no ROM-amplitude optimization variable. It cannot change bounds in
response to candidate force or objective value. If ROM changes, the old episode
is closed as `CLOSED_ROM_PROFILE_CHANGED`; a higher-version frozen profile,
new subject domain, and new Stage 1 episode are required. Trial observations
from different task envelopes cannot silently share one ledger.

`assert_same_frozen_rom_comparison` rejects comparison results with more than
one profile fingerprint. Tests apply this rule to Reference, Random, Space
Filling, Model-Only Greedy, Standard BO, and Physics-Informed BO. The compact
recorded smoke runs Reference, Space Filling, Standard BO, and Physics-Informed
BO on that same domain. Selectors see only completed Stage 1 observations. The
Stage 0 gate is not given future BO performance, and the Stage 1 selector is not
given the ROM calibration provider or unexecuted endpoint truth.

## Offline smoke benchmark

The reproducible command is:

```bash
python3 -m personalization.benchmarks.run_rom_gated_v2_smoke
```

It covers four explicitly synthetic development cases (`ROM_SMALL`,
`ROM_MEDIUM`, `ROM_LARGE`, `ROM_ASYMMETRIC`), four same-ROM methods per case,
and `K=4`: 16 complete Stage 1 runs. Each case first executes the explicit
synthetic gate sequence `SAFE, SAFE, SAFE, STOP`, freezes the last-safe ROM,
builds and checks all 625 V3 candidates, and then starts Stage 1 at beta zero.

The recorded smoke result is:

| Check | Result |
|---|---:|
| ROM development cases | 4 |
| Same-ROM methods per case | 4 |
| Complete K=4 Stage 1 runs | 16/16 |
| Candidate domains with 625 members | 4/4 |
| Domains passing subject-ROM invariants | 4/4 |
| Maximum extrema error across domains | 5.166e-7 rad |
| Minimum warp derivative across domains | 0.896966 |
| Pointwise clipping events | 0 |
| Robot actions | 0 |
| Human actions | 0 |
| PINN training runs | 0 |

The endpoint is an offline inverse-dynamics torque-RMS development proxy plus a
declared synthetic residual. The benchmark verifies architecture compatibility;
it is not evidence that one method is better, safer, more comfortable, or more
effective for people or robots. Exact run ledgers and checksums are in
`personalization/benchmarks/results_v2_rom_gated/`.

## Limitations

- No pressure, force, pain, clinician, or clinical-ROM threshold is validated.
- The independent hip/knee rectangle is only the first schema realization;
  future validated coupled/configuration/trajectory constraints remain required.
- The four ROM cases and synthetic boundary proxy are algorithm-development
  fixtures, not a patient cohort or human safety model.
- The subject-specific reference mapping is audited software behavior; it has
  no current human or real-robot validation.
- The mechanical smoke endpoint is not the future validated measured endpoint.
- Four scalar trials cannot independently identify five effective gray-box
  parameters without the existing regularization assumptions.
- No poor-prior detector was added; V1's documented prior limitations remain.
- Real ROM determination and the real robot environment remain disabled.

## Required-question disposition

1. Stage 0 and Stage 1 are separate: **yes**.
2. Each development case receives its own frozen profile: **yes**.
3. ROM observations are excluded from K=4: **yes**; ledgers are distinct.
4. The V3 warp law and beta grid are preserved: **yes**.
5. Subject V3 trajectories preserve their subject ROM: **yes**, enforced by
   fail-on-build invariant checks with no clipping.
6. Future gates can be plugged in without hard-coded real thresholds: **yes**;
   real thresholds currently remain unavailable and fail closed.
7. A ROM change invalidates/closes the old episode and requires restart: **yes**.
8. Gray-box/residual GP/BO operate through the unchanged V1 core after freeze:
   **yes**, with a subject-trajectory gray-box adapter.
9. The complete offline ROM-to-V3-to-K=4 loop runs end-to-end: **yes**, for four
   development cases and 16 same-ROM method runs.
10. A future validated ROM provider can replace the synthetic Stage 0 provider
    without changing Stage 1: **yes at the software interface level**; that
    provider and its safety protocol are not implemented or validated here.
