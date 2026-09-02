# Subject Model Hierarchy

| Model | Subject-specific state | Observations | Role and gate |
|---|---|---|---|
| M0 — no subject model | measured outcome ledger only | validated episode outcomes | direct/model-free baseline |
| M1 — gray-box | effective mass/stiffness/damping-like parameters or posterior; never physiological truth | q, dq, valid ddq, beta and measured mechanics from executed trials | primary low-data model candidate; must be revalidated for real measurements |
| M2 — data-driven residual | residual parameters/function on top of frozen physics baseline | causal executed-trial features and residual targets | allowed only after M1 residual is defined and train/validation separation exists |
| M3 — physics-informed/residual NN | learned subject-specific residual state | sufficient repeated measured trials | benchmark only after all PINN stop/go gates pass |

## Exact possible PINN task

`measured mechanical response = gray-box physics prediction + subject-specific residual`

Conceptual causal inputs: q, dq, validated ddq, beta/path descriptor, trial context, and past/current measured force state where temporal causality is preserved. Output: time-resolved interaction force/torque residual or a separately calibrated episode-endpoint residual.

The PINN does not infer comfort, generate a personalized objective, replace direct preference labels, or turn effective parameters into physiological truth.

## PINN stop/go gate

Enter a PINN benchmark only if: repeated measured trials exist; M1 is evaluated; a systematic residual exists; it is repeatable across repeated trials; data volume supports a learning split; and an equal-budget comparison against simpler models is frozen. Otherwise: `PINN_NOT_JUSTIFIED`.
