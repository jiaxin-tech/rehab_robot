# MyoLeg V2 Truth Landscape Generation V1

## Decision

`MYOLEG_V2_TRUTH_LANDSCAPE_VALID`

This artifact is a hidden offline virtual truth landscape. It is not a human result, robot-motion approval, clinical result, learner training set, or safety validation.

## Frozen identity and coverage

- Cohort manifest SHA-256: `31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057`.
- Candidate manifest SHA-256: `0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7`.
- Truth semantic: `MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1` / `TAU_MY0LEG_REQUIRED_DRIVE`.
- Evaluated pairs: `533,600` / `533,600`.
- Duplicate pairs: `0`.
- Reference candidate per subject: `MYOLEG_V2_P15012`.
- All subject reference objectives within 1e-12 of 1: `True`.

## Compact storage

The primary data are 2,144 deterministic compressed NPZ shards using 237 uncompressed schema bytes per pair. Shards occupy 36,738,870 bytes; checksum sidecars occupy 190,816 bytes. No bulk 401-sample replay schema was generated, avoiding the approximately 386 GB design.

## Runtime and integrity

- Formal process workers: `4`.
- Formal generation command wall time: `14225.315` s.
- All-pair lightweight integrity: `True`.
- Detailed prescribed-repeat pairs: `352`; pass: `True`.
- Controlled diagnostic pairs: `12` (24 joint rows); pass: `True`.

## Access and oracle ordering

Future algorithms must reveal executed candidates through `query(subject_id, candidate_id)`, which regenerates full prescribed replay arrays. The compact table is restricted to post-hoc evaluation/oracle/regret/personalization analysis. The chunk/data freeze SHA existed before minima were read. Oracle tie handling was preregistered as float64 minimum, 1e-12 equivalence, then lowest original proposal index. No personalization interpretation was performed here.

## Boundaries

No Five-parameter model, NN/PINN, BO, candidate-domain change, cohort change, robot/hardware connection, or human-ready claim was made. The next stage may audit personalization necessity only if the final decision above is valid or valid with limitations.
