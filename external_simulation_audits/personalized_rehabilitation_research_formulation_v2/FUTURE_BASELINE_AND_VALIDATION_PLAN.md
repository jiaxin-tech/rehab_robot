# Future Baseline and Validation Plan

## Equal-budget baseline hierarchy

Future comparison should include S0 reference, S1 common trajectory, S2 random/space-filling exploration, model-free mechanical BO, gray-box plus BO, and residual/PINN plus BO only if justified. Every adaptive method receives exactly `K=4` complete adaptation trials in the primary hypothesis; K=3 and K=5 are sensitivity budgets. No method receives oracle, future, held-out, simulator-preference or extra-trial information.

Non-adaptive baselines must receive a matched exposure/evaluation schedule defined before data collection; they cannot be advantaged or disadvantaged by silently changing the number or duration of executed trials.

## Primary future success form

Mechanical primary: after the equal K-trial adaptation phase, does the frozen subject-adaptive selection produce a lower independently evaluated, prespecified measured mechanical endpoint than reference, common and non-adaptive baselines?

Optional preference extension: with direct human labels, does equal-budget preference-aware selection produce higher independently evaluated direct preference than baselines while satisfying mechanical constraints?

## Validation hierarchy

1. Offline unit/integration and causal-information tests.
2. Sensor/state/wrench timing, frame, calibration and repeatability validation without a personalization claim.
3. Endpoint formulation and repeated-trial reliability study.
4. Equal-budget algorithm comparison on development participants only after independent robot/human approvals.
5. Locked confirmatory evaluation on new real subjects; the generalization unit is the `new real subject`, not a new MyoLeg parameter vector.

Adaptation and final evaluation data remain separate. Safety events, invalid data and constraint breaches are reported independently and never folded into an arbitrary reward weight.
