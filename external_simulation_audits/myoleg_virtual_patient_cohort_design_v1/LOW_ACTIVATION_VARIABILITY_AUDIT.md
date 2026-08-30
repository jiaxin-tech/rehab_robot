# Low-activation variability audit

The primary truth condition remains `P0` with zero control and zero initial activation. Low background activation is not a structural musculoskeletal identity by default; it is an episode-level nuisance describing incomplete relaxation during a passive-rehabilitation trial.

- Keep structural subject parameters fixed across every episode for that subject.
- If used, predefine low activation per episode and keep it separate from the subject manifest's structural factors.
- Only treat a fixed subject baseline as identity after independent physiological justification.
- Group-specific activation needs its own preregistered mapping and cannot be selected using downstream learner performance.

The current control domain is `[0,1]`, so a symmetric negative/positive smoke perturbation about zero is invalid. This design stage did not invent a negative activation or freeze a positive range.
