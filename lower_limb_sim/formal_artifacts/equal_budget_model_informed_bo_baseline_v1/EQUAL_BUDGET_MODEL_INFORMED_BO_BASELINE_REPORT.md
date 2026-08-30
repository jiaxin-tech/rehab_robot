# EQUAL_BUDGET_MODEL_INFORMED_BO_BASELINE_V1

- Protocol SHA-256: 88519ef162c2c5cff584e03ed1eade0b6aee2a1d2d719ee3fd85a91f8fa82aa0
- Frozen config SHA-256: 7f3a248eef5c253e99a11db3160b220071f5f6360e197235054fa6fa21f7267a
- BO_PROTOCOL_INTEGRITY = PASS
- OFFLINE_ONLY, NOT_HUMAN_READY, NOT_ROBOT_APPROVED

## Results

Mean regret: Model Top-1 0.002314; Frozen Top-3
0.002314; BO-A K=2 0.000029; BO-A K=3
0.000020. K=1 equals Model Top-1 in every case.

BO K=3 versus Top-1: 15 wins,
0 ties,
0 losses; mean gain
0.002294, bootstrap 95% CI
[0.001244,
0.003514].

BO K=3 versus Frozen Top-3: 15 wins,
0 ties,
0 losses; mean gain
0.002294, bootstrap 95% CI
[0.001244,
0.003514].

BO K=3 versus per-case Random-3 mean: 15
wins, 0 ties,
0 losses; mean gain
0.014779, bootstrap 95% CI
[0.013057,
0.016514].

Rescue count versus Top-1: 15.

## Direct answers

### Q1 Does adaptive BO provide measurable benefit beyond Model Top-1?

LOW_BUDGET_BO_PROVIDES_ADDITIONAL_VALUE.

### Q2 Does adaptive BO provide measurable benefit beyond Frozen Top-3?

ADAPTIVE_CANDIDATE_ADMISSION_OUTPERFORMS_FROZEN_SET.

### Q3 Does precommitting candidates impose an observable performance cost?

YES.

### Q4 At K=1/2/3, where does regret saturate?

Including K=5, primary BO saturates at K=5.

### Q5 Is the added complexity of BO empirically justified in the current cohort?

YES.

Prior formal conclusions remain unchanged. No robot, human, prospective cohort,
PINN, RL, MPC, or further optimizer stage was started.
