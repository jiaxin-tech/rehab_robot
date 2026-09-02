# Rehabilitation-task invariance audit

## Formal distinction

`TRAJECTORY PERSONALIZATION` should change the interior hip-knee coordination/path while the prescribed rehabilitation task remains fixed. `TASK / ROM DOSAGE MODIFICATION` changes a requested joint extreme or total excursion. Those are scientifically different questions even if both produce valid trajectories.

## What frozen V2 actually changes

- `delta_hip_amp` changes hip maximum by `7.000000` deg and hip ROM by `6.999999` deg across its admitted axis, while cycle duration remains fixed.
- `delta_knee_amp` changes knee maximum by `5.498975` deg and knee ROM by `5.498975` deg across its admitted axis, while cycle duration remains fixed.
- `knee_phase_shift` changes knee timing/path geometry but changes sampled knee maximum by only `0.000047040` deg across the full phase axis (below the frozen matched-ROM tolerance); joint extrema are mathematically fixed at branch endpoints.

Therefore:

`CURRENT_PARAMETERIZATION_CHANGES_REHABILITATION_TASK_AMPLITUDE = true`

The two amplitude variables are task/dose variables, not pure path-shape variables. The current matched-ROM grid status is `CURRENT_GRID_CANNOT_IDENTIFY_FIXED_ROM_PATH_EFFECT`: it supports a one-dimensional phase experiment, not a broad fixed-ROM path-shape experiment.

## Methodological recommendation

For the stated paper question, hip and knee extrema/ROM should be treated as **fixed prescribed-task constraints**. Future personalization should vary only smooth interior coordination/path quantities. This is a design recommendation; it does not modify V2, the active reference, the objective, or any robot setting.
