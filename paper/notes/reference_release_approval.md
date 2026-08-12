# Measured-Asymmetric Reference Release Approval Record

Date recorded: 2026-08-12 (Asia/Shanghai)

This repository record documents the experiment owner's report that the
advisor/laboratory lead approved the following three release decisions:

1. Use the measured natural cycle `5844 -> 5895 -> 5934` as the formal
   flexion-extension source cycle.
2. Accept the audited periodic-closure maximum deviations of `0.246 deg` at
   the hip, `0.189 deg` at the knee, and `2.256 mm` at the equivalent pull
   point.
3. Keep the nominal profile out of real-robot execution because its frozen
   local-identification-domain coverage is `66.334%`, below the unchanged
   `90%` gate. Only the slow profile remains active.

Released active trajectory:

```text
trajectory_id = reference_measured_asymmetric_closed_slow
SHA-256 = f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881
```

The SHA-256 is the byte digest of
`lower_limb_sim/data/reference_candidates/reference_measured_asymmetric_closed_slow.csv`.
Changing that file requires a new numerical audit, a new approval decision,
an updated pinned digest, and a new release checkpoint. Existing StartAnchor
files bound to an earlier trajectory identifier must not be relabeled and
reused.

This approval covers offline reference selection and data processing only. It
does not certify robot safety, validate xCoreSDK behavior, authorize human
motion, or replace the staged Windows, empty-load, mechanical-surrogate, and
institutional release gates in `REAL_ROBOT_EXPERIMENT.md`. No robot connection
or motion was performed to create this record.
