# Data Leakage Audit

- Current P2 was not run or modified in `P2_REVISION_V2_DESIGN_ANALYSIS_V1`.
- The 341 local pairs are reconstructed historical decision opportunities; they are not designated validation.
- Their errors use post-hoc virtual truth. They may support design comparison, not threshold freezing.
- G1/G2 use leave-one-case-out bounds, so each evaluated case is excluded from its bound calculation. This reduces direct case reuse but does not create an independent study population.
- Held-out final-test data were not loaded.
- Truth was not used for model fitting, candidate proposal, current guard execution, exploration feature construction, or stopping input.
- `subject_specificity_gap.csv` uses truth only to classify post-hoc regret and possible cause categories.
- No human threshold, formal personalization, safety evidence, or robot approval was created.
