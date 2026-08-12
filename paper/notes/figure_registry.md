# Figure and table registry

Only claim-bearing figures/tables are planned. Existing PNGs are audit inputs,
not automatically publication-ready artwork. Manuscript copies or wrappers may
be placed in `paper/figures/` and `paper/tables/` without modifying source data.

## Figures

### Fig. 1 — Reference-centered research pipeline

- **Research question:** How does a prescribed rehabilitation path lead to task-local identification, interaction evaluation, and constrained candidate selection?
- **Source data:** None; architecture derived from the implemented module/data flow and the frozen paper story.
- **Generating script:** Not present; create a vector schematic only after P0 selection interfaces are frozen.
- **Already exists?:** No manuscript-ready figure.
- **Needs regeneration?:** Yes.
- **Main manuscript claim:** Identification is an enabling component of reference-centered personalization; timing/mismatch are reliability checks rather than standalone contributions.
- **Current evidence status:** Framework grounded in code; final selection and physical branch incomplete.

### Fig. 2 — Two-link lower-limb geometry and coordinate convention

- **Research question:** What are $q_h$, $q_k$, $L_1$, $L_2$, the $x$--$z$ frame, and the strap equivalent traction point?
- **Source data:** Equations and conventions in `lower_limb_sim/kinematics.py`, `jacobian.py`, and `config.py`.
- **Generating script:** No dedicated manuscript script; `lower_limb_sim/workspace_atlas.py` generates related sample postures.
- **Already exists?:** `lower_limb_sim/data/workspace/sample_postures.png` exists but is not a complete coordinate schematic.
- **Needs regeneration?:** Yes, as vector artwork with $	heta_s=q_h-q_k$ made explicit.
- **Main manuscript claim:** The model uses a sagittal two-link chain whose endpoint is the strap equivalent traction point, not automatically the ankle.
- **Current evidence status:** Fully grounded analytical/software convention.

### Fig. 3 — Reachable workspace and active asymmetric reference

- **Research question:** Where does the active closed path lie relative to the implemented sagittal workspace and local-domain gate?
- **Source data:** `workspace_atlas.csv`, `reference_measured_asymmetric_closed_slow.csv`, `reference_measured_asymmetric_domain_coverage.csv`, and active-reference metadata.
- **Generating script:** `workspace_atlas.py` and `visualize_reference_measured_asymmetric.py`; a combined manuscript plotting wrapper is still needed.
- **Already exists?:** Separate workspace and active-reference path figures exist.
- **Needs regeneration?:** Yes, to combine them, state the exact ROM configuration, and avoid mixing legacy and active domains.
- **Main manuscript claim:** A closed asymmetric software reference can be retained inside its frozen software domain.
- **Current evidence status:** FORMAL software evidence (R-KIN-001, R-REF-ASYM-001); no physical reachability claim.

### Fig. 4 — Train-only equivalent-identification pipeline

- **Research question:** Which signals enter the estimator, where are invalid samples rejected, and how are training, validation, and test roles separated?
- **Source data:** `identification_dataset.py`, `parameter_estimator.py`, `identifiability_analysis.py`, and `run_identification.py`.
- **Generating script:** Not present; create a vector/data-flow schematic from the audited interfaces.
- **Already exists?:** No single pipeline figure; result plots exist per identification case.
- **Needs regeneration?:** Yes.
- **Main manuscript claim:** The estimator uses only valid $q,\dot q,\ddot q,F$ observations, rebuilds $J^T F$, fits on training data, and evaluates held-out data afterward.
- **Current evidence status:** Fully grounded implementation; physical input acquisition remains unvalidated.

### Fig. 5 — Excitation and local identifiability

- **Research question:** How do excitation breadth, sensitivity singular values, parameter correlations, and task-local state coverage relate?
- **Source data:** `lower_limb_sim/data/reference_local_active_asymmetric/` files `excitation_metadata.csv`, `domain_coverage.csv`, `state_domain_bounds.json`, `identifiability_summary.csv`, `parameter_correlations.csv`, and `sensitivity_singular_values.csv`.
- **Generating script:** `lower_limb_sim/visualize_reference_local_active_asymmetric.py`, invoked by `run_reference_local_active_asymmetric.py`.
- **Already exists?:** Yes. Candidate panels are `active_reference_local_excitation_family.png`, `active_reference_local_domain_coverage.png`, and `active_reference_local_identifiability.png`; `figure_manifest.json` records each panel's question, sources, and interpretation. Held-out torque and generic-comparison companion figures are also present.
- **Needs regeneration?:** Only a publication-layout composite may be needed; the active-reference source panels and data are now frozen.
- **Main manuscript claim:** Held-out prediction must be accompanied by excitation-domain and identifiability evidence.
- **Current evidence status:** FORMAL offline synthetic evidence (R-ID-002, R-LOCAL-ACTIVE-001). The 10%-faster boundary panel must retain its 81.0474% coverage warning.

### Fig. 6 — State--wrench timing misalignment and causal alignment

- **Research question:** How do fixed and variable delays affect equivalent identification, and what is recovered by offline versus causal methods?
- **Source data:** `all_delay_summary.csv`, `all_variable_delay_summary.csv`, selected delay-search curves, tracking histories, and validity summaries.
- **Generating script:** `run_delay_compensation_experiment.py`, `run_variable_delay_experiment.py`, and `visualize_variable_delay.py`.
- **Already exists?:** Yes, many separate before/after and per-case figures.
- **Needs regeneration?:** Yes, with one fixed-delay accuracy panel, one torque-error panel, and one causal variable-delay example; offline/causal semantics must be visually distinct.
- **Main manuscript claim:** Temporal misalignment can strongly bias identification, while validated timestamps and causal matching can reduce the synthetic bias.
- **Current evidence status:** FORMAL offline synthetic evidence (R-TIME-001, R-TIME-002); no real timestamp validation.

### Fig. 7 — Model-mismatch task-local generalization and failure boundary

- **Research question:** When does the identified equivalent model outperform a generic baseline, and where does it fail?
- **Source data:** `model_mismatch/summaries/generic_vs_identified_comparison.csv` plus per-scenario split and residual-correlation CSV files.
- **Generating script:** `run_model_mismatch_experiment.py` and `visualize_model_mismatch.py`.
- **Already exists?:** Per-scenario comparison plots exist.
- **Needs regeneration?:** Yes, to show combined-mild improvement and strong-coupling negative cases in the same scale, separated by interpolation/boundary/outside domain.
- **Main manuscript claim:** Equivalent fitting is useful under some mild mismatch but is not guaranteed under strong structural mismatch or extrapolation.
- **Current evidence status:** FORMAL offline synthetic evidence (R-MM-001).

### Fig. 8 — Active reference versus selected personalized candidate

- **Research question:** Does a deterministically selected task-local candidate improve the prespecified mechanical criterion while satisfying all constraints?
- **Source data:** Future active-reference candidate metrics, feasibility report, selector trace, subject-wise comparison, and held-out evaluation.
- **Generating script:** Existing `run_reference_candidate_evaluation.py` and `visualize_reference_candidates.py` are starting points; selector and active-reference rerun are missing.
- **Already exists?:** Legacy C0--C8 plots exist, but no valid active-reference selected-candidate figure exists.
- **Needs regeneration?:** Yes, after P0.1--P0.3.
- **Main manuscript claim:** The main personalization claim; must compare unchanged active reference with one preselected candidate and report constraint margins/failures.
- **Current evidence status:** INCOMPLETE. Current evidence supports only legacy model-based candidate screening (R-CAND-001).

## Tables

### Table I — Notation, frames, geometry, and modeled quantities

- **Research question:** Are all coordinates, signs, units, and equivalent-parameter meanings unambiguous?
- **Source data:** `config.py`, `kinematics.py`, `jacobian.py`, `full_dynamics.py`, and manuscript equation registry.
- **Generating script:** None; LaTeX wrapper to be authored from audited constants/interfaces.
- **Already exists?:** No.
- **Needs regeneration?:** Yes, after ROM harmonization; keep knee ROM experiment-specific.
- **Main manuscript claim:** Reproducible modeling convention, including $	heta_s=q_h-q_k$ and strap-equivalent $L_2$.
- **Current evidence status:** Grounded; one repository knee-ROM conflict is disclosed.

### Table II — Identification design and estimator configuration

- **Research question:** What trajectories, splits, parameters, bounds, loss, scaling, and solver settings define the estimator?
- **Source data:** `identification_dataset.py`, `reference_local_excitation.py`, `parameter_estimator.py`, and per-run metadata.
- **Generating script:** Manuscript table wrapper; values are code/config derived.
- **Already exists?:** No manuscript table.
- **Needs regeneration?:** Yes, but the active-reference local design is now frozen and table-ready.
- **Main manuscript claim:** Fixed train-only five-parameter estimation with held-out validation/test evaluation.
- **Current evidence status:** Global and active-reference-local designs are grounded (R-ID-001, R-LOCAL-ACTIVE-001).

### Table III — Identification accuracy and identifiability

- **Research question:** What prediction/parameter errors and sensitivity diagnostics are obtained across virtual subjects and observation conditions?
- **Source data:** `identification_summary.json`, `dataset_metrics.csv`, per-case global summaries, and `reference_local_active_asymmetric/{identifiability_summary,parameter_errors,prediction_metrics,generic_vs_identified,domain_coverage}.csv`.
- **Generating script:** `run_identification.py` and `run_reference_local_active_asymmetric.py`; a paper aggregation wrapper is still needed.
- **Already exists?:** Source summaries exist; publication table does not.
- **Needs regeneration?:** Yes, to separate matched self-consistency, noisy robustness, current active-local evidence, and limited legacy comparison.
- **Main manuscript claim:** Equivalent prediction can remain accurate despite parameter bias; low residual and identifiability are distinct evidence.
- **Current evidence status:** FORMAL offline synthetic (R-ID-001, R-ID-002, R-LOCAL-ACTIVE-001); R-LOCAL-001 is retained only as legacy provenance.

### Table IV — Timing-robustness summary

- **Research question:** How do uncorrected, offline fixed-delay, global fixed, and causal buffered methods compare?
- **Source data:** `all_delay_summary.csv`, `all_variable_delay_summary.csv`, and per-case `method_comparison.csv`.
- **Generating script:** Timing experiment scripts plus a new read-only table wrapper.
- **Already exists?:** Aggregate CSVs exist; no manuscript table.
- **Needs regeneration?:** Yes, with explicit timestamp-availability and causality columns.
- **Main manuscript claim:** Timing semantics materially affect identification reliability.
- **Current evidence status:** FORMAL offline synthetic (R-TIME-001, R-TIME-002).

### Table V — Model-mismatch and domain-generalization summary

- **Research question:** Across mismatch scenarios, where does identification improve or degrade predictions relative to a generic baseline?
- **Source data:** `model_mismatch/summaries/generic_vs_identified_comparison.csv` and per-scenario metadata.
- **Generating script:** `run_model_mismatch_experiment.py`; paper aggregation wrapper needed.
- **Already exists?:** Aggregate CSV exists.
- **Needs regeneration?:** Yes, to report interpolation, boundary, and outside-domain separately and retain negative improvements.
- **Main manuscript claim:** Task-local utility has explicit structural and domain boundaries.
- **Current evidence status:** FORMAL offline synthetic (R-MM-001).

### Table VI — Candidate feasibility and selected-candidate comparison

- **Research question:** Which active-reference candidates pass each gate, and what does the predeclared selector choose for each subject?
- **Source data:** Future active-reference feasibility, metrics, Pareto/selector trace, and held-out comparison CSVs.
- **Generating script:** Extend/reuse `run_reference_candidate_evaluation.py` after P0 selector implementation.
- **Already exists?:** Legacy feasibility/metrics/Pareto CSVs exist; required active selected-candidate table does not.
- **Needs regeneration?:** Yes.
- **Main manuscript claim:** One constrained, task-local candidate improves the declared mechanical objective without violating implemented gates.
- **Current evidence status:** INCOMPLETE; legacy screening only (R-CAND-001).

### Table VII — Evidence ladder and physical release status

- **Research question:** Which claims are analytical, simulated, software-validated, physical, or human-validated?
- **Source data:** `results_registry.md`, robot export metadata, root README, and `CURRENT_ARCHITECTURE.md`.
- **Generating script:** None; manuscript/reproducibility table derived from registry status.
- **Already exists?:** No.
- **Needs regeneration?:** Yes whenever a release gate changes.
- **Main manuscript claim:** Prevent simulation or tests from being presented as robot/human evidence.
- **Current evidence status:** Current physical state is NO-GO; no human evidence.
