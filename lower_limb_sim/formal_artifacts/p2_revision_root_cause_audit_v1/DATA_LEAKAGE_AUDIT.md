# Data leakage audit

- Virtual truth role: `POST_HOC_ROOT_CAUSE_AND_COUNTERFACTUAL_ONLY`.
- Truth landscapes and local finite differences were computed only after the frozen P2 implementation and reference/model definitions were fixed.
- Current P2 was replayed unchanged; truth was not supplied to proposal, candidate ranking, support, guard calibration, model fitting, exploration ranking, or stopping.
- Counterfactual G0 outcomes use truth only to label post-hoc true/false/missed outcomes.
- G1/G2 were not constructed from truth and remain unavailable because there is no designated validation pair on the formal local-alpha scale.
- Synthetic parameter scan role: `SYNTHETIC_PARAMETER_SENSITIVITY_ONLY`; it uses only parameter values already present in registered repository virtual subjects.
- Held-out final-test data were not loaded.
- Future stopping candidates contain no truth feature.
- No robot or human trajectory was executed; no human threshold or robot approval was created.
