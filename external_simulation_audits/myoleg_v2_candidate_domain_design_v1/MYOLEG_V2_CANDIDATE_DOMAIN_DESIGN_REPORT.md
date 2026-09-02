# MYOLEG_V2_CANDIDATE_DOMAIN_DESIGN_V1

## Final outcome

`MYOLEG_V2_CANDIDATE_DOMAIN_VALID_WITH_LIMITATIONS`

This is an offline simulator-domain design, not a human-safety or robot-motion approval.  No candidate objective, ranking, learner, BO acquisition, or full truth landscape was computed.

## Q1 - Frozen trusted simulator domain

- Native hip simulator range: `-30.000013` to `120.000281` deg.  The admitted candidate hip envelope is `28.909336` to `114.025439` deg, separated from the native boundaries.
- Upper trusted knee artifact bound: `120.0` deg.  The admitted candidate knee envelope is `18.320897` to `119.999933` deg.
- Selection used all 32 frozen subjects plus nominal at preregistered hip/low-speed boundary states.  Gates were absolute joint-limit contribution <= `0.005` Nm, relative contribution <= `0.0005`, equality residual <= `0.001`, finite state, zero warnings and zero contact constraints.

The lower diagnostic was non-monotonic at the exact native boundary: 0 deg was inactive, 0.1/0.25 deg crossed the preregistered gate, and >=0.5 deg passed.  This does not affect admission because every admitted trajectory remains above `18.320897` deg; consequently this report does not claim the entire native 0--120 deg interval is uniformly artifact-free.

These are `SIMULATOR_ARTIFACT_GATE` limits, not human or robot safety limits.

## Q2 - A_V2 size

`16675 / 21,025` original proposals remain.  Original proposal indices are retained; candidates were not renumbered.

## Q3 - Exclusions

```json
{
  "myoleg_knee_upper_trusted_bound": 4350
}
```

Reasons are deterministic and may overlap.  No `J_pred`, truth J, model coverage or performance filter was used.

## Q4 - Reference and immediate neighborhood

Reference proposal `MYOLEG_V2_P15012` (original index `15012`) is included: `True`.

```json
[
  {
    "direction": "HIP_NEGATIVE",
    "alpha": [
      -0.25,
      0.0,
      0.0
    ],
    "proposal_index": 14287,
    "candidate_id": "MYOLEG_V2_P14287",
    "included": true,
    "exclusion_reasons": ""
  },
  {
    "direction": "HIP_POSITIVE",
    "alpha": [
      0.25,
      0.0,
      0.0
    ],
    "proposal_index": 15737,
    "candidate_id": "MYOLEG_V2_P15737",
    "included": true,
    "exclusion_reasons": ""
  },
  {
    "direction": "KNEE_NEGATIVE",
    "alpha": [
      0.0,
      -0.25,
      0.0
    ],
    "proposal_index": 14987,
    "candidate_id": "MYOLEG_V2_P14987",
    "included": true,
    "exclusion_reasons": ""
  },
  {
    "direction": "KNEE_POSITIVE",
    "alpha": [
      0.0,
      0.25,
      0.0
    ],
    "proposal_index": 15037,
    "candidate_id": "MYOLEG_V2_P15037",
    "included": true,
    "exclusion_reasons": ""
  },
  {
    "direction": "PHASE_NEGATIVE",
    "alpha": [
      0.0,
      0.0,
      -0.0025
    ],
    "proposal_index": 15011,
    "candidate_id": "MYOLEG_V2_P15011",
    "included": true,
    "exclusion_reasons": ""
  },
  {
    "direction": "PHASE_POSITIVE",
    "alpha": [
      0.0,
      0.0,
      0.0025
    ],
    "proposal_index": 15013,
    "candidate_id": "MYOLEG_V2_P15013",
    "included": true,
    "exclusion_reasons": ""
  }
]
```

## Q5 - Positive knee exploration

Positive knee-amplitude exploration available: `True`.  This is determined only by the all-model trusted native-domain artifact gate.

## Q6 - Phase/C2/closure

All 25 phase values pass monotonic, branch, duration and C2 endpoint checks: `True`.  Every amplitude/phase proposal was independently checked for finite q/dq/ddq and q/dq/ddq closure.

## Q7 - Sparse MyoLeg validation

The preregistered `30` candidates were prescribed-replayed on `7` models (`SUBJECT_NOMINAL_CONTROL, MYOLEG_VP_031, MYOLEG_VP_011, MYOLEG_VP_009, MYOLEG_VP_004, MYOLEG_VP_032, MYOLEG_VP_028`), for `210` replay cases.  All passed: `True`.  No objective or rank was calculated.

## Q8 - Global set

One global candidate set applies to all 32 subjects.  There is no development/held-out or subject-specific candidate deletion.

## Q9 - Full landscape engineering estimate

- trajectories: `533,600`
- measured mean prescribed replay: `0.170007` s/candidate/subject
- serial: `25.199` h
- idealized 8-worker at 75% efficiency: `4.200` h
- torque-only float64 storage: `3.424` GB
- full retained replay schema estimate: `386.188` GB

These are engineering estimates only.

## Q10 - Next stage

Ready to design/execute `MYOLEG_V2_TRUTH_LANDSCAPE_GENERATION_V1`: `True` with the synthetic-cohort and sparse-validation limitations.  This stage did not start it.

Final candidate-domain manifest SHA-256: `0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7`.
