# Data-role audit

- DEVELOPMENT shadow outcomes: 15 cases (9 original P2 development + 6 rejected prospective cases already reclassified as post-rejection development).
- INDEPENDENT_CALIBRATION: 12 cases; used only for 324 one-step residual/direction observations and 216 direct endpoint residuals at each of 2/3/5 steps.
- The 12 calibration case IDs are not passed to the policy runner, final-J calculation, regret calculation, missed-improvement calculation, or false-improvement calculation.
- Calibration truth selects no semantic candidate and contributes no policy-performance row.
- The exact S0-S3 manifest is persisted and SHA-gated before initial identification or any development truth access.
- Development truth is attached only after each complete semantic path; it cannot alter gate definitions, percentile, 0.005, bundle lengths, or interpretation criteria.
- No new prospective cohort is generated. The immutable prior prospective conclusion remains `P2_V2_PROSPECTIVE_EVIDENCE_REJECTS_CURRENT_REVISION`.
- HELD_OUT_FINAL_TEST is not loaded or read.
- This task is synthetic offline research only; no hardware, control, collection, safety, or robot connector is imported by the semantics core/runner.
