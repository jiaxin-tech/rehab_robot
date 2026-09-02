# Option C — Measurement-Driven Personalization

## Recommendation

`PIVOT_TO_MEASUREMENT_DRIVEN_PERSONALIZATION`

This is the primary pivot because it preserves the original individual, multi-round research goal without claiming that synthetic MyoLeg parameters are patient preference truth. The pivot is scientifically plausible, not already validated: actual measured data may still show no learnable personalization signal.

## Mechanical-only primary question

Given a fixed rehabilitation task and the offline-feasibility-screened low-dimensional V3 coordination-path family, can a physics-informed subject model and low-budget adaptive exploration use only that subject's executed-trial force, torque, pressure, and tracking measurements to select a trajectory with a better prespecified **mechanical interaction metric** than fixed-reference, common-trajectory, random, and non-adaptive baselines under equal trial budget?

Required data: synchronized executed-trial interaction mechanics and state/tracking data. No comfort claim is permitted.

## Preference/comfort primary question

Given the same fixed task and feasible V3 family, can a low-budget preference-learning method use explicit per-subject ratings or pairwise trajectory choices, with mechanical feasibility constraints, to identify a trajectory the subject reports as preferable under equal trial budget?

Required data: direct rating, pairwise choice, or equivalent explicit human response, plus separate mechanical/safety monitoring. Pressure or torque alone is not preference truth.

## Conceptual architecture

1. **Fixed task/family:** retain validated `beta_flex, beta_extend`; do not change ROM to manufacture benefit.
2. **Physics prior:** analytical dynamics, MyoLeg, or a gray-box model provides feasibility and an initial mechanical prediction.
3. **Subject adaptation:** start with the existing five-parameter gray-box identification concept, refit only from executed trials and revalidate its adequacy for measured data. Consider a residual NN or physics-informed residual NN only if measured residual structure and sufficient data justify it.
4. **Low-budget selection:** compare random exploration, BO without a subject model, model-informed BO, and preference-based BO under the same trial budget.
5. **Feedback target:** mechanical measurements for mechanical personalization; direct feedback for preference/comfort; both remain constrained by independent feasibility and safety gates.

## Fair future validation philosophy

Replace “does a synthetic cohort contain different oracle trajectories?” with: “given actual subject-specific observations, does the adaptive method predict or select better than non-personalized and non-adaptive baselines under equal trial budget?”

Conceptual baselines: fixed reference, population/common trajectory, random exploration, model-only prediction, BO without subject model, gray-box plus BO, and—only when justified—residual/PINN plus BO.

This audit does not define a human protocol, trial count, safety threshold, or robot release gate. Those remain separate prerequisites.
