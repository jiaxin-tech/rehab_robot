# Policy design data-role audit

- DEVELOPMENT policy-shadow cases: 15 (9 original P2 development + 6 rejected-prospective cases permanently reclassified as development).
- INDEPENDENT_CALIBRATION: 12 cases; residual distributions only. Their case IDs are never passed to the policy runner or outcome summarizer.
- FUTURE_PROSPECTIVE: not generated.
- HELD_OUT_FINAL_TEST: not read and no loader is called.
- Candidate manifest is frozen before initial identification or any development truth call.
- Development truth labels outcomes only after each policy path is complete and cannot alter candidate, percentile, bundle length, formula, or stopping K.
- The six rejected-prospective cases cannot support a new prospective success claim; immutable conclusion remains `P2_V2_PROSPECTIVE_EVIDENCE_REJECTS_CURRENT_REVISION`.
