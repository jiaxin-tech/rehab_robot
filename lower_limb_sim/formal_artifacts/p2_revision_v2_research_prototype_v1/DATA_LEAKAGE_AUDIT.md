# Data Leakage Audit

- Prototype status: `DEFAULT_OFF_RESEARCH_SHADOW_ONLY`; the existing P2 remains the default.
- No P2 policy run was executed by this runner and no current decision was changed.
- Local pairs and uncertainty metrics use retrospective virtual truth only as research calibration/counterfactual labels; they are not formal designated validation or frozen thresholds.
- G1/G2 use leave-one-case-out metrics. This reduces direct same-case reuse but is not an independent validation population.
- Exploration scoring uses already-executed history, model outputs, support, ranking, eligibility, and observed best change. It emits no stop action.
- The knee cumulative audit uses truth only after the frozen historical policy and never proposes, ranks, accepts, or executes a trajectory.
- Held-out final-test data were not loaded.
- No human threshold, hardware motion, robot approval, or formal personalization was created.
