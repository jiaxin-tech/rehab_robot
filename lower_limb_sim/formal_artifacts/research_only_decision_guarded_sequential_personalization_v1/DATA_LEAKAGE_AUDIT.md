# Data leakage audit

- Model fitting: initial TRAIN plus valid selected ADAPTATION_EXECUTED observations.
- Decision-uncertainty calibration: designated VALIDATION only.
- Held-out final test: not loaded, not fitted, not ranked, not calibrated, not used for stopping.
- Virtual truth: selection-token gated; proposal/ranking truth-call count remained unchanged: `true`.
- Exactly one selected virtual trajectory per iteration: `true`.
- Full-map truth: absent from policy inputs. Final local regret truth is a separately labelled post-policy evaluation.
- Exploration ranking: information metrics first; predicted J, truth, support distance are not the primary score.
- Support: `DATA_PROVENANCE_NOT_RELIABILITY_APPROVAL`.
- Human/robot approval created: `false`.
