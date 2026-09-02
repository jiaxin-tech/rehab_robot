# Future Personalization Episode Protocol

This is a causal conceptual protocol, not hardware approval.

## Cold start and research budget

- Trial 1 starts at the frozen reference: `beta=[0,0]`.
- Primary adaptation-budget hypothesis: `K=4` complete trials, including the reference cold start.
- Sensitivity budgets: `K=3` and `K=5`.
- The budget is motivated by the two-dimensional family, one interpretable baseline plus a small number of updates, and subject-burden control. It is an experimental-design hypothesis, not an approved robot or human exposure.

## One sequential episode

1. Select `beta_k` from the future independently approved V3 subset/domain.
2. Execute one complete, separately approved rehabilitation trial.
3. Acquire time-qualified robot state and validated interaction measurements; tactile/direct feedback are included only if independently available.
4. Apply frozen quality gates and compute one versioned episode-feature record.
5. Update the subject-specific gray-box parameter/posterior state using trials `1..k` only.
6. If justified, update a residual/observation model using the same causal history only.
7. Update the selector/surrogate and select `beta_(k+1)`.
8. Repeat until the fixed K-trial adaptation budget is exhausted.

## Information boundary

Known before trial: fixed reference/task, V3 mathematical family, future approved domain, fixed physics prior, reviewed safety constraints, algorithm settings, and this subject's past valid trials only.

Measured during trial: synchronized state/tracking and validated mechanical interaction; optional tactile or direct feedback only when the corresponding protocol exists.

Updated after trial: episode features, effective gray-box parameters/posterior, predictive uncertainty, optional residual model state, BO/surrogate state, and the observed-candidate ledger.

Forbidden before execution: the current/future trial outcome, final-evaluation outcome, held-out subject outcome, synthetic MyoLeg oracle/preference, or extra observations unavailable to comparison baselines.

## Identification versus evaluation

The K adaptation trials cannot also be claimed as final performance evidence. After adaptation freezes the selected beta, a separate final-evaluation block must compare the selected trajectory with reference and common/non-personalized controls under a preregistered, counterbalanced and equal-observation protocol. Repetition count/order is deliberately left to the next data/endpoint design stage.
