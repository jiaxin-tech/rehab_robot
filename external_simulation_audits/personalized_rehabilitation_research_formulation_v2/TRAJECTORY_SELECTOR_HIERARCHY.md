# Trajectory Selector Hierarchy

| Selector | Required observations | Personalization source |
|---|---|---|
| S0 fixed reference | none | none |
| S1 common/population trajectory | frozen non-personalized prior only | none for current subject |
| S2 random/space-filling | candidate geometry and previously executed set | subject outcomes are recorded but not modeled for selection |
| S3 model-only greedy | updated subject model predictions | subject's executed mechanical trials via M1/M2/M3 |
| S4 standard mechanical BO | measured mechanical endpoint and uncertainty | subject's executed endpoint observations |
| S5 physics/model-informed BO | physics/gray-box prior plus measured endpoint | subject's executed measurements update the prior/surrogate |
| S6 preference BO | direct rating/pairwise labels plus constraints | explicit human feedback only |

Mechanical BO optimizes one independently calibrated observable mechanical endpoint. Preference BO optimizes latent preference utility derived from direct labels. `BO is a selector, not the source of personalization.` The source is `subject-specific observations`.

## BO stop/go gate

`PERSONALIZED_BO_JUSTIFIED` requires a frozen candidate domain, observable endpoint/direct feedback, complete-trial semantics, independent safety constraints, and an equal-budget comparator. None is inferred from synthetic MyoLeg preference. Until the next design stage freezes the endpoint and physical domain, personalized BO remains not justified and is not run.
