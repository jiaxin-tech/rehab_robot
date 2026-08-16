# Safeguard / Identification / Personalization Architecture

Protocol: `SAFEGUARDED_SEQUENTIAL_INITIAL_IDENTIFICATION_V1`

Evidence level: deterministic offline virtual-subject software validation only.
Nothing in this artifact is robot, human, clinical, comfort, safety, or
effectiveness validation.

## Layer 1 — HARD SAFEGUARD

Status: `NOT_DEFINED_NOT_APPROVED`.

This is an independent, future real-robot fail-closed protection layer.
Neither identification nor personalization may override it.  The global model
ROM (hip 0–120 deg, knee 5–145 deg) is not a universal patient-safe ROM.

## Layer 2 — SEQUENTIAL INITIAL IDENTIFICATION

One candidate is selected and virtually executed at a time.  Selection sees
only executed identification data, the current temporary five-parameter model,
global model constraints, and the current pre-supplied patient operational
envelope.  It ranks candidates lexicographically by constraint validity, rank,
minimum singular value, condition number, worst correlation, weakest-parameter
sensitivity, incremental state/regressor coverage, excursion, and stable ID.

After every trial, all five local equivalent dynamics parameters are audited.
The process stops immediately when a reviewed rule is met and never exceeds
5 trials.  Failure produces no `theta_hat_0`.
The repository currently has no approved complete stop rule, so the saved
virtual experiment uses an explicitly non-frozen illustrative comparator.  The
authoritative default remains `IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW`.

`excitation_duration_s` is independent of the unchanged 24 s rehabilitation
reference.  The tested durations are a research design range, not human-safety
limits.

## Layer 3 — SEQUENTIAL PERSONALIZATION (future interface only)

On successful identification only:

```text
theta_hat_0 + D_init + full prediction map + initial known region
    -> future EXPLOIT / EXPLORE
    -> execute one approved personalization trial
    -> theta_hat_(k+1) and updated support map
```

No personalization is implemented or executed by this task.  Unsupported
geometrically admissible points retain a calculated `J_pred` but are labelled
`UNSUPPORTED_EXTRAPOLATION`: can calculate does not mean can trust.

## Constraint ownership

| Layer | Meaning | May this module change it? |
|---|---|---|
| GLOBAL_MODEL_CONSTRAINTS | Model ROM, workspace, Jacobian, force mapping, C2, finite values | No; validate only |
| PATIENT_SPECIFIC_OPERATIONAL_ENVELOPE | Pre-supplied conservative local region | No; `AUTO_EXPAND_PATIENT_ENVELOPE=false` |
| REAL_ROBOT_HARD_SAFEGUARD | Future independent hardware protection | No; not defined or approved |
