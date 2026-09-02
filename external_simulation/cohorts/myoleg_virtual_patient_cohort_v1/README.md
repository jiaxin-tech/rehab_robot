# MYOLEG_VIRTUAL_PATIENT_COHORT_V1

This directory contains 32 frozen **heterogeneous musculoskeletal virtual
subjects** and one separate nominal control.  It is not a representative
patient cohort or physiological population sample.

Each subject is stored compactly as:

- `model_delta.json`: exact double-precision Scheme-A field changes relative to
  frozen base model SHA `20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d`;
- `metadata.json`: identity, split, hashes, denominators and integrity result;
- `reference_replay_truth.npz`: complete frozen P0/V2 prescribed and controlled
  replay arrays.

No copied MJB or upstream meshes are retained.  Reconstruct by loading the
frozen base XML, applying `model_delta.json` in factor order, and verifying the
compiled model fingerprint.  Do not resample or replace subjects.
