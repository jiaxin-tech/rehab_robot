# Experiment and implementation TODO

Priorities reflect the frozen paper story, not the repository's software roadmap.
No item below authorizes robot motion or changes to control/safety code.

## P0 — Required for the paper's main claim

### P0.1 Regenerate task-local identification around the active asymmetric reference

- **Current state:** **Completed for formal offline matched-clean simulation on
  2026-08-11 (R-LOCAL-ACTIVE-001).** The runner's overwrite-protected result
  directory is
  `lower_limb_sim/data/reference_local_active_asymmetric/`.
- **Evidence obtained:** Frozen 12-trajectory active-reference neighbor set;
  six/two/four train/validation/test split; train-only 6-D domain; four-subject
  five-parameter fits; recomputed rank/singular-value/conditioning/correlation
  diagnostics; exact-active, nominal, within-domain, and boundary held-out
  predictions; generic comparison; retrospective legacy comparison; five
  scientific figures and hashed run metadata.
- **Release criterion:** **Passed.** Every new artifact records active ID
  `reference_measured_asymmetric_closed_slow` and SHA-256
  `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`.
- **Evidence boundary:** This closes the software P0.1 gap only. It does not
  provide physical, robot, human, clinical, or comfort evidence.

### P0.2 Finalize and implement subject-specific candidate selection

- **Current state:** Incomplete. Fixed C0--C8 candidate generation, feasibility
  screening, and Pareto reporting exist, but there is no finalized objective or
  deterministic subject-specific selector; all nine stored candidates are
  nondominated. P0.1 now supplies the active-reference-local identification and
  domain basis, but no selector was implemented in that work.
- **Required evidence:** A frozen mechanical interaction criterion, objective
  normalization/weights or another deterministic decision rule, train-only
  tuning, and a predeclared fallback when no candidate is feasible.
- **Release criterion:** The algorithm selects one candidate (or rejects all)
  without inspecting its final evaluation result.

### P0.3 Rerun candidate screening and comparison about the active reference

- **Current state:** Missing. Active-reference-local identification is now
  FORMAL, but existing C0--C8 candidate evidence remains centered on the
  superseded symmetric trajectory.
- **Required evidence:** Active-reference C0 baseline; selected candidate;
  closure, ROM, workspace, conditioning, local-domain, velocity, acceleration,
  and experiment-specific load-gate results; subject-wise and aggregate
  comparison on held-out model evaluation.
- **Release criterion:** A FORMAL reference-versus-selected result supports a
  task-local personalization claim without comfort or clinical language.

### P0.4 Acquire synchronized physical state/wrench evidence

- **Current state:** Missing. Read-only diagnostics and acquisition architecture
  exist, but no diagnostic result directory or reviewed physical episode was
  found; robot trajectory generation is blocked and physical status is NO-GO.
- **Required evidence:** Reviewed frame/sign/reference-point semantics, source
  timestamps, arrival timestamps, state age, wrench age, skew, validity and
  invalid reasons, query timing, and tracking data from a permitted dummy setup.
- **Release criterion:** Fail-closed preflight passes under the project's
  separate physical approval process; the paper records the exact episode and
  configuration provenance.

### P0.5 Demonstrate equivalent identification and unseen-path prediction on physical data

- **Current state:** Missing. `scripts/identify_real_episode.py` exists, but no
  qualifying episode/result was found.
- **Required evidence:** Train-only physical identification, repeated trials,
  validation selection, unseen task-local trajectory evaluation, generic versus
  identified comparison, uncertainty/repeatability, and failure reporting.
- **Release criterion:** A reviewed dummy-leg or approved physical-subject result
  replaces purely synthetic support for the enabling identification claim.

## P1 — Strongly recommended

### P1.1 Harmonize and freeze experiment-specific ROM conventions

- **Current state:** **Completed in software by `ROM_PROTOCOL_V2`.** The formal
  manifest freezes hip 0--120 deg, knee 5--145 deg and the subtractive shank
  convention for workspace, IK, reference, identification, candidate and
  robot-preview gates. Old 5--130 deg outputs remain inactive legacy evidence.
- **Remaining physical boundary:** Site-specific robot workspace, soft limits
  and safety thresholds still require human review; this migration does not
  approve them.

### P1.2 Subject-specific geometry calibration and sensitivity reporting

- **Current state:** Calibration/error-analysis software exists; no physical
  calibration result exists.
- **Needed:** Repeatable estimates of hip center, $L_1$, and strap-equivalent
  $L_2$, with uncertainty propagated into identification/prediction.

### P1.3 Physical dummy-leg validation before any human work

- **Current state:** Missing.
- **Needed:** Tracking error, force/torque consistency, timing/skew, repeated
  identification, candidate comparison, and all invalid/rejected samples under
  the fail-closed robot workflow.

### P1.4 Tactile calibration and acquisition integration

- **Current state:** Missing; no tactile sensor implementation, calibration, or
  quantitative file was found.
- **Needed:** Sensor geometry, calibration protocol, units, sample timestamps,
  spatial feature definitions, wrench/tactile alignment, uncertainty, and
  repeatability. Until then Section V-B stays framework/TODO only.

### P1.5 Freeze reproducibility manifests and manuscript tables

- **Current state:** P0.1 now has a self-contained configuration, split,
  provenance, figure manifest, checksums, and run summary. Older experiments
  remain distributed across many CSV/JSON files.
- **Needed:** Commit/hash, configuration, deterministic seeds, environment,
  exact run commands, checksums, aggregate scripts, and table-ready CSVs.

### P1.6 Report negative and outside-domain cases prominently

- **Current state:** Implemented and available, including strong-coupling
  failures, outside-domain trajectories, and the P0.1 10%-faster held-out case
  with 18.9526% of samples outside the train-fitted state box.
- **Needed:** Predefined acceptance/failure metrics and confidence intervals or
  replication across seeds/virtual populations, not only per-scenario averages.

## P2 — Optional or supplementary

### P2.1 Human-subject experiment

- **Current state:** Missing and not required for the immediate software/dummy
  validation milestone.
- **Prerequisites:** Ethics approval, clinical protocol, stopping rules,
  qualified supervision, calibrated sensing, physical-system release, and a
  separate statistical analysis plan. No present claim depends on it.

### P2.2 Compare richer equivalent model structures

- Cross-joint passive coupling, nonlinear stiffness/damping, or residual models
  may be compared only as prespecified ablations with identifiability controls.
  They should not displace the reference-centered paper story.

### P2.3 Online adaptation or control integration

- Not implemented and out of the current manuscript scope. MPC, PINN, and
  closed-loop personalized control should remain future work unless a separate
  validated implementation and evidence set is added.

## Requested-item audit matrix

| Item | Current state | Priority |
|---|---|---|
| Finalized continuous closed asymmetric reference | **Present (software):** active 24-s, 401-sample, periodic C2 slow reference | Done for software; physical validation P0 |
| Reference-local excitation design | **Present for current active reference:** 12 fixed trajectories with 6/2/4 train/validation/test split | P0.1 complete |
| Personalization algorithm implementation | **Partial:** fixed candidates and Pareto comparison, no final selector | P0 |
| Candidate feasibility screening | **Present but legacy-centered:** ROM/workspace/closure/Jacobian/force/domain gates | P0 regeneration |
| Tactile calibration | Absent | P1 |
| Tactile acquisition integration | Absent | P1 |
| Synchronized ROKAE state/wrench collection | Architecture/diagnostics present; qualifying result absent | P0 |
| Dummy-leg physical validation | Absent | P1 after P0.4 |
| Real-data equivalent parameter identification | Script present; qualifying data/result absent | P0 |
| Unseen-trajectory prediction | Present for the active reference in matched-clean simulation; absent on physical data | P0.1 complete; physical P0.5 |
| Reference vs personalized candidate comparison | Present only for legacy fixed candidates; no final selected active candidate | P0 |
| Human-subject experiment | Absent | P2, approval-dependent |
