# Results registry

This file is the quantitative source of truth for the manuscript. `FORMAL`
means that a versioned repository result and a generating implementation were
found; it does not mean physical or clinical validation. Paths are relative to
the repository root. Quantities reported below were audited on 2026-08-11.

## R-KIN-001 — Reachable sagittal workspace

- **Stage / Module:** Workspace and kinematics
- **Scientific question:** What sagittal traction-point region satisfies the implemented ROM and above-bed geometry checks?
- **Experiment type:** software validation
- **Claim supported:** The implemented two-link geometry produces a finite, filterable workspace atlas under formal `ROM_PROTOCOL_V2`.
- **Numerical evidence:** 17,061 sampled configurations; 11,993 retained as reachable. The newly admitted knee-above-130-deg region contains 1,815 samples, including 1,127 geometrically reachable samples; all 1,815 pass the existing Jacobian numerical gate.
- **Primary source file:** `lower_limb_sim/formal_artifacts/rom_protocol_v2/workspace/workspace_atlas.csv`
- **Generating script:** `lower_limb_sim/workspace_atlas.py`
- **Input dataset:** None; deterministic grid from `lower_limb_sim/config.py`.
- **Figure/table source:** `lower_limb_sim/formal_artifacts/rom_protocol_v2/workspace/workspace_hip_angle.png`, `workspace_knee_angle.png`, and `sample_postures.png`.
- **Allowed interpretation:** Software evidence for geometry, workspace and Jacobian filters under hip 0--120 deg and knee 5--145 deg.
- **Interpretation NOT supported:** Clinical ROM, collision avoidance, robot reachability, or participant safety.
- **Candidate manuscript section:** VI-B; Fig. 3.
- **Status:** FORMAL

## R-DYN-001 — Quasi-static force-map audit

- **Stage / Module:** Quasi-static equivalent dynamics and force mapping
- **Scientific question:** How do virtual-subject parameters change the modeled quasi-static endpoint force over the implemented workspace?
- **Experiment type:** simulation
- **Claim supported:** The quasi-static model and Jacobian pseudoinverse produce subject-dependent software force maps and explicitly reject anomalous mappings.
- **Numerical evidence:** Four subject maps over the workspace; the hip-stiff map reaches 998.08 N among valid samples and has 14 samples rejected above the 1000-N software anomaly threshold.
- **Primary source file:** `lower_limb_sim/formal_artifacts/rom_protocol_v2/force_maps/virtual_subject_comparison.csv`
- **Generating script:** `lower_limb_sim/build_force_map.py`
- **Input dataset:** Formal V2 workspace atlas and virtual-subject definitions.
- **Figure/table source:** `lower_limb_sim/formal_artifacts/rom_protocol_v2/force_maps/virtual_subject_comparison.png` and per-subject force-map CSV files.
- **Allowed interpretation:** Numerical behavior of the quasi-static model and its software validity gates.
- **Interpretation NOT supported:** A physical interaction-load limit, safe contact force, or measured human/robot force.
- **Candidate manuscript section:** VI-B; supplementary model audit.
- **Status:** FORMAL

## R-DYN-002 — Dynamic speed-profile audit

- **Stage / Module:** Full inverse dynamics
- **Scientific question:** Does time scaling separate inertial and damping loads while retaining a common joint path?
- **Experiment type:** simulation
- **Claim supported:** The implemented dynamic model exhibits the expected speed dependence and valid force mapping for the audited virtual trajectories.
- **Numerical evidence:** All 12 subject-by-speed trajectories are valid. For the baseline slow/nominal/fast profiles, peak endpoint force is 183.81/184.55/189.13 N, peak inertial torque is 0.092/0.368/1.472 N m, and peak damping torque is 0.682/1.364/2.727 N m. The largest audited endpoint force is 243.04 N for the knee-stiff fast case.
- **Primary source file:** `lower_limb_sim/dynamic_force_audit.md`
- **Generating script:** `lower_limb_sim/simulate_dynamic_trajectory.py` and `lower_limb_sim/compare_speed_profiles.py`
- **Input dataset:** Deterministic trajectory profiles and virtual-subject definitions.
- **Figure/table source:** `lower_limb_sim/data/dynamic_trajectories/*/speed_profile_comparison.csv` and `.png`.
- **Allowed interpretation:** Internal consistency and speed sensitivity of the software inverse-dynamics model.
- **Interpretation NOT supported:** Robot tracking performance or physically measured loads.
- **Candidate manuscript section:** VI-B; Table III or supplement.
- **Status:** FORMAL

## R-ID-001 — Global equivalent-parameter identification

- **Stage / Module:** Global identification and observation-noise audit
- **Scientific question:** Can the five-parameter estimator recover equivalent parameters and predict held-out torques under matched and noisy synthetic observations?
- **Experiment type:** offline synthetic experiment
- **Claim supported:** The bounded train-only estimator is numerically self-consistent in matched data and remains predictive, with parameter bias, under the audited realistic synthetic observation combination.
- **Numerical evidence:** In clean matched data, the maximum parameter relative error is approximately 0.000001% and maximum test torque RMSE is 5.21e-9 N m. Across combined-realistic cases, mean parameter error is 12.64%, maximum parameter error is 87.01%, mean test RMSE is 0.400 N m, and maximum test RMSE is 0.421 N m.
- **Primary source file:** `lower_limb_sim/data/identification/identification_summary.json`
- **Generating script:** `lower_limb_sim/run_identification.py`
- **Input dataset:** `lower_limb_sim/identification_dataset.py` outputs for four virtual subjects, three motion families, three speeds, and configured observation scenarios.
- **Figure/table source:** `lower_limb_sim/data/identification/dataset_metrics.csv`, `parameter_estimates.csv`, and per-case prediction/parameter plots.
- **Allowed interpretation:** Offline synthetic recovery/prediction performance for the implemented model, splits, bounds, and noise scenarios.
- **Interpretation NOT supported:** Real-human parameter accuracy, unique physiology, or robot performance.
- **Candidate manuscript section:** VI-C.
- **Status:** FORMAL

## R-ID-002 — Local identifiability audit

- **Stage / Module:** Identifiability analysis
- **Scientific question:** Does the selected excitation set provide locally independent sensitivity to all five estimated parameters?
- **Experiment type:** analytical
- **Claim supported:** The audited sensitivity matrices are numerically full rank, while their conditioning remains subject- and excitation-dependent.
- **Numerical evidence:** All clean full-excitation cases have numerical rank 5; condition numbers for the all-family/all-speed set range from 40.19 to 108.04; maximum absolute off-diagonal parameter correlations are approximately 0.405--0.409, with no pair at or above 0.9.
- **Primary source file:** `lower_limb_sim/data/identification/*/clean/identifiability_summary.csv`
- **Generating script:** `lower_limb_sim/run_identification.py` using `lower_limb_sim/identifiability_analysis.py`
- **Input dataset:** Clean synthetic identification datasets and train-fitted parameter estimates.
- **Figure/table source:** Per-subject `sensitivity_singular_values.png` and `parameter_correlation_heatmap.png`.
- **Allowed interpretation:** Local numerical sensitivity/conditioning for the chosen equivalent model and excitations.
- **Interpretation NOT supported:** Global structural identifiability, physiological uniqueness, or uncertainty calibrated on physical data.
- **Candidate manuscript section:** VI-C; Fig. 5 and Table III.
- **Status:** FORMAL

## R-TIME-001 — Fixed-delay compensation audit

- **Stage / Module:** State--wrench timing alignment
- **Scientific question:** How much does a fixed state--force delay bias identification, and can train/validation grid selection correct it without test leakage?
- **Experiment type:** offline synthetic experiment
- **Claim supported:** On the injected grid, common-support validation selection recovers the fixed delay and removes most delay-induced identification error.
- **Numerical evidence:** All 24 combinations (four virtual subjects by six delays from 0 to 40 ms) are recovered exactly at 1-ms resolution, with zero boundary selections. Worst test torque RMSE decreases from 0.48621 to 0.000635 N m; worst parameter error decreases from 28.643% to 0.000767%.
- **Primary source file:** `lower_limb_sim/data/delay_compensation/all_delay_summary.csv`
- **Generating script:** `lower_limb_sim/run_delay_compensation_experiment.py`
- **Input dataset:** Synthetic identification trajectories with injected fixed force delays.
- **Figure/table source:** `delay_estimation_accuracy.png`, `test_rmse_before_after.png`, and `damping_error_before_after.png` in the same directory.
- **Allowed interpretation:** Offline sensitivity and recoverability for delays located exactly on the tested synthetic grid.
- **Interpretation NOT supported:** Sub-millisecond accuracy, arbitrary real latency, or online robot compensation.
- **Candidate manuscript section:** VI-D; Fig. 6 and Table IV.
- **Status:** FORMAL

## R-TIME-002 — Variable-delay causal alignment audit

- **Stage / Module:** Buffered causal sample matching and variable-delay tracking
- **Scientific question:** Can causal history matching preserve identification accuracy under time-varying synthetic delay, jitter, dropout, and stale data?
- **Experiment type:** offline synthetic experiment
- **Claim supported:** The causal buffered method performs accurately in the core synthetic suite when reliable sample-time semantics are available and rejects invalid timing events explicitly.
- **Numerical evidence:** In the core causal-buffered results, mean/max test torque RMSE is approximately 4.3e-5/7.2e-5 N m, mean/max delay MAE is 0.645/1.878 ms, mean/min acceptance is 0.9917/0.9669, and maximum parameter error is 0.000719%.
- **Primary source file:** `lower_limb_sim/data/variable_delay/all_variable_delay_summary.csv`
- **Generating script:** `lower_limb_sim/run_variable_delay_experiment.py`
- **Input dataset:** Synthetic asynchronous state/wrench streams from `variable_delay_models.py`.
- **Figure/table source:** Per-case `true_vs_estimated_delay.png`, `state_match_error_vs_time.png`, `valid_rejected_samples.png`, and `method_comparison.csv`.
- **Allowed interpretation:** Software evidence for causal buffering, timestamp validity handling, and synthetic delay tracking.
- **Interpretation NOT supported:** Verified ROKAE timestamp semantics, real-time deadline performance, or physical synchronization accuracy.
- **Candidate manuscript section:** VI-D; Fig. 6 and Table IV.
- **Status:** FORMAL

## R-MM-001 — Structured model-mismatch and generalization audit

- **Stage / Module:** Equivalent-model mismatch/generalization
- **Scientific question:** Does the fitted five-parameter equivalent model improve task-local predictions when the generator violates its structure?
- **Experiment type:** offline synthetic experiment
- **Claim supported:** The fitted equivalent model can improve predictions under mild structured mismatch, but strong mismatch produces clear trajectory-specific failures.
- **Numerical evidence:** For combined-mild mismatch, interpolation-path mean torque RMSE is 1.2973 N m for the generic model and 0.46168 N m after identification, a mean improvement of 63.55% (56.90--72.86% by interpolation path); near-boundary improvement is 53.80%. Under strong hip--knee coupling, mean interpolation improvement is -23.61%, with a minimum of -190.38%.
- **Primary source file:** `lower_limb_sim/data/model_mismatch/summaries/generic_vs_identified_comparison.csv`
- **Generating script:** `lower_limb_sim/run_model_mismatch_experiment.py`
- **Input dataset:** Nine mismatch scenarios generated by `mismatch_dynamics.py`, `mismatch_scenarios.py`, and held-out paths from `generalization_trajectories.py`.
- **Figure/table source:** Per-scenario `generic_vs_identified_rmse.png`, `nrmse_by_split.png`, and `residual_feature_correlations.png`.
- **Allowed interpretation:** Task-local predictive benefits and failure boundaries of an equivalent model under specified synthetic mismatch.
- **Interpretation NOT supported:** Recovery of nonlinear generator coefficients, real physiology, or guaranteed improvement outside the training domain.
- **Candidate manuscript section:** VI-D; Fig. 7 and Table V.
- **Status:** FORMAL

## R-GEO-001 — Geometry and observation-model error audit

- **Stage / Module:** Geometry calibration and kinematic observation error
- **Scientific question:** How do link-length, hip-center, neutral-angle, and angle-observation errors propagate into equivalent identification?
- **Experiment type:** offline synthetic experiment
- **Claim supported:** Equivalent-parameter and prediction errors depend materially on the assumed geometry and observation mode, with some identified cases worse than the generic baseline.
- **Numerical evidence:** Scenario-level estimates, generic-versus-identified errors, observation-mode comparisons, and $q_0$--stiffness correlations are stored for matched, signed, mild, strong, and angle-noise cases; no single aggregate is promoted because effects are mode- and scenario-dependent.
- **Primary source file:** `lower_limb_sim/geometry_error_audit.md`
- **Generating script:** `lower_limb_sim/run_geometry_error_experiment.py`
- **Input dataset:** Synthetic trajectories and geometry-error definitions from `geometry_error_scenarios.py` and `kinematic_observation.py`.
- **Figure/table source:** `lower_limb_sim/data/geometry_error/summaries/` plus per-scenario comparison plots.
- **Allowed interpretation:** A controlled software stress test that motivates subject-specific geometry calibration and explicit observation assumptions.
- **Interpretation NOT supported:** Validated calibration accuracy on a person or robot, or a universal error bound.
- **Candidate manuscript section:** VI-D or supplement.
- **Status:** FORMAL

## R-REF-ASYM-001 — Active closed asymmetric reference

- **Stage / Module:** Reference trajectory construction
- **Scientific question:** Can a measured flexion--extension cycle be closed smoothly without erasing its asymmetry and remain inside the frozen software domain?
- **Experiment type:** software validation
- **Claim supported:** The active slow reference is a 24-s, 401-sample, asymmetric, periodic $C^2$ software trajectory with complete frozen-domain coverage.
- **Numerical evidence:** Selected raw cycle: 91 rows and 4.507-mm natural closure error. Periodic correction maxima: 0.2462 deg hip, 0.1891 deg knee, and 2.2558 mm traction point. Asymmetry-retention ratios exceed 0.99998. Slow profile: 13.6-s flexion plus 10.4-s extension and 100% domain coverage. The 12-s nominal profile has only 66.334% coverage and is fail-closed.
- **Primary source file:** `reference_release/reference_release_manifest.json`; frozen CSV SHA-256 `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`.
- **Generating script:** `lower_limb_sim/run_reference_measured_asymmetric.py`
- **Input dataset:** Natural cycle `5844 -> 5895 -> 5934`; source skeleton SHA and reconstruction paths are frozen in `reference_release/source_reference_information.json`.
- **Figure/table source:** `measured_flexion_vs_extension.png`, `raw_vs_periodic_closed.png`, `asymmetry_preservation.png`, and `new_reference_pull_path.png`.
- **Allowed interpretation:** A frozen software reference satisfying the implemented continuity/correction/domain gates.
- **Interpretation NOT supported:** Robot executability, participant-specific rehabilitation prescription, or clinical validity.
- **Candidate manuscript section:** VI-B and VI-E; Fig. 3.
- **Status:** FORMAL

## R-LOCAL-001 — Reference-local identification around legacy reference

- **Stage / Module:** Reference-local excitation and identification
- **Scientific question:** Can limited neighboring trajectories identify the five equivalent parameters and predict held-out local perturbations?
- **Experiment type:** offline synthetic experiment
- **Claim supported:** In matched clean simulation, the fixed six-trajectory local design is numerically sufficient for the legacy symmetric reference.
- **Numerical evidence:** Four training trajectories provide 1,604 samples. Across four virtual subjects, maximum parameter relative error is 2.037e-6% and maximum held-out torque RMSE is 9.421e-9 N m. Validation/test state-box coverage is 94.264%/94.514%.
- **Primary source file:** `lower_limb_sim/data/reference_candidates/reference_local_parameter_estimates.csv`
- **Generating script:** `lower_limb_sim/reference_local_excitation.py` and `lower_limb_sim/run_reference_candidate_evaluation.py`
- **Input dataset:** `reference_local_identification_dataset.csv`, centered on `reference_closed_c2_slow.csv`/`nominal.csv`.
- **Figure/table source:** `local_excitation_trajectories.png` and `local_identification_domain.png`.
- **Allowed interpretation:** Matched-model, task-local numerical feasibility for the legacy symmetric reference.
- **Interpretation NOT supported:** Identification around the active asymmetric reference or physical subject identification.
- **Candidate manuscript section:** VI-C and VI-E, only with explicit legacy qualification.
- **Status:** FORMAL

## R-LOCAL-ACTIVE-001 — Active-asymmetric reference-local identification

- **Stage / Module:** Active-reference-local excitation, identification, and held-out validation
- **Scientific question:** Does the existing five-parameter train-only pipeline transfer to the current active continuous closed asymmetric reference with adequate local support and numerical identifiability?
- **Experiment type:** formal offline matched-clean synthetic experiment
- **Claim supported:** The conservative active-reference-local design provides full-rank local numerical identifiability for the adopted five-parameter equivalent model and accurate held-out prediction under matched clean virtual-subject dynamics.
- **Numerical evidence:** Twelve fixed 401-sample trajectories were prespecified: six train, two validation, and four test. The exact active slow reference was held out and has 99.7506% coverage in the train-fitted axis-aligned 6-D $q/\dot q/\ddot q$ box; the relevant nominal profile has 98.7531%, the within-domain phase-delay test has 100%, and the 10%-faster boundary profile has 81.0474%. Across the four frozen virtual subjects, every sensitivity matrix has rank 5, condition numbers range from 36.448 to 90.131, maximum absolute parameter correlation is 0.3085, and no pair reaches 0.9. Maximum matched-clean parameter relative error is $3.654\times10^{-6}\%$; maximum held-out combined torque RMSE is $1.874\times10^{-8}$ N m. Mean generic-to-identified held-out RMSE improvement is 99.99999993%, which is numerical self-consistency evidence under matched clean equations, not a realistic effect-size estimate.
- **Primary source file:** `lower_limb_sim/data/reference_local_active_asymmetric/run_summary.json`
- **Generating script:** `lower_limb_sim/run_reference_local_active_asymmetric.py` using `reference_local_active_asymmetric.py`
- **Input dataset:** Current active `reference_measured_asymmetric_closed_slow.csv` (SHA-256 `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`), its relevant nominal profile, and the four prespecified virtual subjects.
- **Figure/table source:** Five `active_reference_*.png` figures, `domain_coverage.csv`, `identifiability_summary.csv`, `parameter_errors.csv`, `prediction_metrics.csv`, and `generic_vs_identified.csv` in the same result directory.
- **Allowed interpretation:** Transfer of task-local equivalent identification from the legacy symmetric setting to the current active asymmetric task, with local numerical identifiability and held-out matched-clean prediction for the adopted model.
- **Interpretation NOT supported:** True human parameter recovery, global dynamics identification, clinical/comfort benefit, physical safety, robot execution, or human validation. The 18.9526% outside-domain fraction on the 10%-faster boundary case remains a failure warning despite its low matched-model prediction error.
- **Candidate manuscript section:** IV-A/IV-C and VI-C/VI-E; Fig. 5 and Table III.
- **Status:** FORMAL

## R-CAND-001 — Fixed candidate feasibility and Pareto screening

- **Stage / Module:** Reference-local candidate evaluation
- **Scientific question:** Do fixed amplitude, phase, and duration variants pass implemented software gates and expose mechanical tradeoffs?
- **Experiment type:** offline synthetic experiment
- **Claim supported:** The legacy symmetric-reference evaluator can screen nine fixed candidates and compute subject-dependent equivalent torque/force/smoothness tradeoffs.
- **Numerical evidence:** All C0--C8 candidates are feasible and all are nondominated under the unweighted objectives. For the worst-subject aggregate, C0 peak knee torque/RMS combined torque is 64.1178/50.1203 N m; C4 is 61.2643/48.7294 N m. C7 reduces torque-rate and jerk metrics from 11.664/16.832 to 9.381/6.764 while extending duration from 24.0 to 28.8 s. Software endpoint-force peaks are approximately 259.76 N.
- **Primary source file:** `lower_limb_sim/data/reference_candidates/reference_candidate_metrics.csv`
- **Generating script:** `lower_limb_sim/run_reference_candidate_evaluation.py`
- **Input dataset:** Legacy symmetric reference, `reference_local_parameter_estimates.csv`, and fixed C0--C8 definitions.
- **Figure/table source:** `candidate_joint_paths.png`, `candidate_torque_comparison.png`, `candidate_subject_comparison.png`, and `candidate_pareto.png`.
- **Allowed interpretation:** Implemented model-based candidate generation, gate evaluation, and multi-objective comparison around the legacy reference.
- **Interpretation NOT supported:** A finalized subject-specific selector, active-reference personalization, comfort improvement, physical safety, or clinical optimality.
- **Candidate manuscript section:** VI-E; Fig. 8 and Table VI after active-reference regeneration.
- **Status:** FORMAL

## R-REF-IMPORT-001 — Imported 3-D source motion quality audit

- **Stage / Module:** Reference-trajectory import
- **Scientific question:** Is the imported motion source immediately suitable as a planar, timed rehabilitation reference?
- **Experiment type:** software validation
- **Claim supported:** The raw import exposes timing, planarity, angle, and joint-limit issues that require processing before use.
- **Numerical evidence:** Source frame rate is absent; maximum out-of-plane deviation is 0.09294 m; 54 angle rows and 105 joint-limit rows are invalid under the import audit.
- **Primary source file:** `lower_limb_sim/data/reference_trajectories/processed/metadata.json`
- **Generating script:** `lower_limb_sim/run_reference_trajectory.py`
- **Input dataset:** Repository-provided raw reference-trajectory source.
- **Figure/table source:** Processed import CSV/metadata and import visualization outputs.
- **Allowed interpretation:** A data-quality warning that justifies later cycle selection and closure processing.
- **Interpretation NOT supported:** Valid source timing, planar ground truth, or an executable reference.
- **Candidate manuscript section:** VI-A or supplement.
- **Status:** PRELIMINARY

## R-ROBOT-READINESS-001 — Physical-evidence readiness audit

- **Stage / Module:** Robot trajectory export and acquisition architecture
- **Scientific question:** Does the current repository contain an approved trajectory export and reviewed physical episode?
- **Experiment type:** software validation
- **Claim supported:** The codebase contains fail-closed acquisition/preflight architecture, but current physical evidence remains NO-GO.
- **Numerical evidence:** `generation_status=blocked`, `robot_execution_approved=false`; no generated robot trajectory CSV, diagnostics result directory, or physical episode/result directory was found in the 2026-08-11 audit.
- **Primary source file:** `lower_limb_sim/data/robot_trajectories/metadata.json`
- **Generating script:** `lower_limb_sim/run_robot_trajectory_export.py`
- **Input dataset:** Active asymmetric reference plus robot calibration/preflight configuration.
- **Figure/table source:** None.
- **Allowed interpretation:** Current evidence boundary and release-gate state.
- **Interpretation NOT supported:** Robot execution, dummy-leg validation, real-subject data, or safe deployment.
- **Candidate manuscript section:** VI-F as an explicit TODO/evidence boundary, not a result claim.
- **Status:** PRELIMINARY

## R-SW-TEST-001 — Offline regression suite

- **Stage / Module:** Repository-wide software tests
- **Scientific question:** Do repository unit/integration tests pass in an offline environment?
- **Experiment type:** software validation
- **Claim supported:** The audited working tree passes the broad offline software regression suite.
- **Numerical evidence:** 649 passed and 5 skipped in 103.30 s on 2026-08-11 using `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider`.
- **Primary source file:** `lower_limb_sim/data/reference_local_active_asymmetric/run_summary.json` (verification record)
- **Generating script:** `pytest -q`
- **Input dataset:** Test fixtures and fake adapters.
- **Figure/table source:** None.
- **Allowed interpretation:** Software regression/debug status for the audited working tree only.
- **Interpretation NOT supported:** Experimental validation, real-time behavior, real-robot behavior, or safety.
- **Candidate manuscript section:** Reproducibility supplement only.
- **Status:** DEBUG

## R-REF-RETIME-OLD-001 — Early imported-reference retiming

- **Stage / Module:** Legacy reference retiming
- **Scientific question:** Could the initially imported path be retimed directly?
- **Experiment type:** software validation
- **Claim supported:** Historical only; the direct retiming did not establish a closed cycle.
- **Numerical evidence:** Endpoint closure error is 0.1999756 m and source frame rate is unknown.
- **Primary source file:** `lower_limb_sim/data/reference_trajectories/retimed/metadata.json`
- **Generating script:** `lower_limb_sim/run_reference_retiming.py`
- **Input dataset:** Early processed import.
- **Figure/table source:** Legacy retiming visualizations.
- **Allowed interpretation:** Development history explaining why a new cycle-boundary/closure pipeline was needed.
- **Interpretation NOT supported:** Current reference continuity or executability.
- **Candidate manuscript section:** None.
- **Status:** SUPERSEDED

## R-REF-C2-LEGACY-001 — Legacy symmetric closed reference

- **Stage / Module:** Legacy periodic reference construction
- **Scientific question:** Could a manually shaped symmetric path be represented with periodic $C^2$ continuity?
- **Experiment type:** software validation
- **Claim supported:** Historical software construction used by the stored local-identification/candidate experiments.
- **Numerical evidence:** 24-s slow and 12-s nominal profiles, each with 401 samples; exact current values remain in the version manifest and C2 metadata.
- **Primary source file:** `lower_limb_sim/data/reference_candidates/reference_c2_metadata.json`
- **Generating script:** `lower_limb_sim/run_reference_c2.py`
- **Input dataset:** Legacy symmetric reference definition.
- **Figure/table source:** `reference_c2_joint_comparison.png`, `reference_c2_acceleration_comparison.png`, and `reference_c2_pull_path_comparison.png`.
- **Allowed interpretation:** Provenance for legacy task-local experiments.
- **Interpretation NOT supported:** Current active-reference status or preservation of measured flexion--extension asymmetry.
- **Candidate manuscript section:** Methods provenance/supplement only.
- **Status:** SUPERSEDED

## Status count

- **FORMAL:** 13
- **PRELIMINARY:** 2
- **DEBUG:** 1
- **TEMPORARY:** 0
- **SUPERSEDED:** 2
