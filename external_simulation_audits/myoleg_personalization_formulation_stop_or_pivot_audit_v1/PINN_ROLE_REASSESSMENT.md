# PINN Role Reassessment

## Decision

`PINN_NOT_YET_JUSTIFIED`

There is currently no subject-specific measured dataset that gives a PINN a distinct scientific task. Training one now would add complexity without evidence that a physics residual is learnable or needed.

A PINN or physics-informed residual network becomes scientifically testable only after executed subject trials provide:

- synchronized inputs and mechanical targets;
- a prespecified train/validation separation;
- evidence that the gray-box physics baseline leaves systematic, repeatable residual structure;
- enough independent observations to compare against simpler residual and gray-box baselines.

Its future task would be `physics baseline + subject-specific residual`, not generation of synthetic patient preference truth. It must demonstrate equal-budget predictive or selection benefit over the simpler model.
