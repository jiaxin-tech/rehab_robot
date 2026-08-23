# Calibration data split audit

The four data layers are strictly separated:

- `DEVELOPMENT`: the nine early P2 development cases plus the six revealed rejected-prospective cases. They are permanently development-used evidence.
- `INDEPENDENT_CALIBRATION`: the 12 new cases in `calibration_case_manifest.csv`; they are used only for decision-error residual estimation.
- `FUTURE_PROSPECTIVE`: `NOT_CREATED_IN_THIS_TASK`. No case or outcome is generated here.
- `HELD_OUT_FINAL_TEST`: `HELD_OUT_FINAL_TEST_NOT_READ`. It is not read, selected, enumerated for truth, or evaluated.

The calibration IDs do not overlap either protected historical group. Every calibration case has `reserved_for_future_prospective=false`, so a future prospective manifest must exclude all IDs recorded here. Case selection used a fixed synthetic grid, seed, exclusions by pre-existing identity/signature, and SHA ordering only; it used no truth optimum, residual, or subject-specificity result.
