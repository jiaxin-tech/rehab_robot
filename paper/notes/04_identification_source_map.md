# Section IV source map

Every prose paragraph is identified by the `% Trace:` marker in
`paper/sections/04_identification.tex`. Equations are mapped separately.

## Paragraph map

### IV-A-P1

- **Manuscript subsection:** IV-A, Identification Data and Excitation Design
- **Claim/equation:** Estimator observations are $q,\dot q,\ddot q,F,t,$ validity; invalid samples are excluded; torque is rebuilt from $J^T F$; generator parameters/torques are not estimator inputs.
- **Implementation source:** `identification_dataset.py`, `parameter_estimator.py`.
- **Relevant function/class:** `validate_identification_dataset`, `valid_observations`, `_reject_leakage_columns`, `measured_joint_torque`.
- **Relevant configuration:** Explicit observation whitelist and validity fields.
- **Relevant result file:** Per-case identification datasets/metadata; R-ID-001.
- **Confidence:** HIGH for software; LOW for physical signal semantics.
- **Notes / assumptions:** Real observations require frame/sign/timestamp validation before reuse.

### IV-A-P2

- **Manuscript subsection:** IV-A
- **Claim/equation:** General suite has three motion families by three speeds, split 4 train/2 validation/3 test before fitting, to vary state derivatives and hip--knee coupling.
- **Implementation source:** `config.py`, `trajectory_profiles.py`, `identification_dataset.py`.
- **Relevant function/class:** Trajectory profile generators, `build_identification_dataset`, `split_identification_dataset`.
- **Relevant configuration:** Family/speed definitions and split map.
- **Relevant result file:** `data/identification/dataset_metrics.csv` and per-case metadata (R-ID-001/R-ID-002).
- **Confidence:** HIGH.
- **Notes / assumptions:** Excitation intent is supported by identifiability diagnostics, not a universal optimal-design claim.

### IV-A-P3

- **Manuscript subsection:** IV-A
- **Claim/equation:** Current local suite uses 12 fixed 401-sample active-reference neighbors: six train, two validation, and four held out; the exact active slow reference never enters fitting; the local domain is a train-fitted six-dimensional axis-aligned box.
- **Implementation source:** `reference_local_active_asymmetric.py`, generalized helpers in `reference_local_excitation.py`, and `run_reference_local_active_asymmetric.py`.
- **Relevant function/class:** Active-reference loader/hash gate; conservative trajectory builder; local dataset builder; train-only estimator/domain fitting; domain coverage and held-out prediction.
- **Relevant configuration:** Train: hip/knee -3 deg and knee phase +/-3% across slow/nominal timing. Validation: -2-deg hip and +2% knee phase at 18 s. Test: -2% knee phase at 18 s, exact active slow, exact relevant nominal, and 10%-faster nominal boundary. Every trajectory has 401 samples.
- **Relevant result file:** `data/reference_local_active_asymmetric/experiment_config.json`, `domain_coverage.csv`, and `run_summary.json` (R-LOCAL-ACTIVE-001).
- **Confidence:** HIGH.
- **Notes / assumptions:** The box is a software support heuristic, not a confidence region.

### IV-A-P4

- **Manuscript subsection:** IV-A
- **Claim/equation:** General and current active-reference-local suites answer different questions; the exact active slow/nominal and within/boundary paths remain held out, while legacy symmetric results are used only for a same-definition retrospective transfer check.
- **Implementation source:** `reference_version_manifest.csv`, `reference_local_active_asymmetric.py`, `reference_local_excitation.py`, and `reference_measured_asymmetric.py`.
- **Relevant function/class:** Manifest/hash verification; split declarations; active local experiment; limited legacy recomputation.
- **Relevant configuration:** Active ID `reference_measured_asymmetric_closed_slow`; six/two/four split; legacy C2 source explicitly labeled retrospective.
- **Relevant result file:** R-REF-ASYM-001, R-LOCAL-ACTIVE-001, and R-LOCAL-001 (legacy only).
- **Confidence:** HIGH.
- **Notes / assumptions:** The comparison supports framework transfer, not superiority of asymmetric geometry.

### IV-B-P1

- **Manuscript subsection:** IV-B, Equivalent Parameter Estimation
- **Claim/equation:** Five parameters comprise common mass/inertia scale and hip/knee K/B; COM, neutral angles, gravity fixed; prediction uses exact Section III inverse dynamics.
- **Implementation source:** `parameter_estimator.py`, `full_dynamics.py`.
- **Relevant function/class:** `_parameter_vector`, `candidate_subject_from_parameters`, `predict_joint_torque`.
- **Relevant configuration:** `BaselineSubjectTemplate` and subject geometry.
- **Relevant result file:** `parameter_estimates.csv` and per-case `estimated_parameters.json` (R-ID-001).
- **Confidence:** HIGH.
- **Notes / assumptions:** Scaling both masses and both inertias is exact current behavior.

### IV-B-P2

- **Manuscript subsection:** IV-B
- **Claim/equation:** Only training observations enter a joint-scaled soft-l1 bounded fit; each torque scale has a 1-N m floor.
- **Implementation source:** `parameter_estimator.py`.
- **Relevant function/class:** `_torque_scales`, `estimate_subject_parameters`.
- **Relevant configuration:** `identification_loss="soft_l1"`; train-only caller/data split.
- **Relevant result file:** Per-case metadata/estimates (R-ID-001).
- **Confidence:** HIGH.
- **Notes / assumptions:** LaTeX cost is a concise representation of SciPy's standard soft-l1 loss applied to the scaled residual vector.

### IV-B-P3

- **Manuscript subsection:** IV-B
- **Claim/equation:** Parameter bounds, initial vector, x-scale, 500-evaluation limit, and post-fit metrics are as implemented.
- **Implementation source:** `config.py`, `parameter_estimator.py`, `run_identification.py`.
- **Relevant function/class:** `estimate_subject_parameters`, `compute_torque_metrics`.
- **Relevant configuration:** Bounds [0.6,1.6], [0,60] K, [0,10] B; initial [1,10,10,1,1]; x-scale [1,20,20,3,3]; `max_nfev=500`.
- **Relevant result file:** Per-case identification summaries and aggregate `dataset_metrics.csv`.
- **Confidence:** HIGH.
- **Notes / assumptions:** Physical justification of bounds remains a future citation/experiment item.

### IV-B-P4

- **Manuscript subsection:** IV-B
- **Claim/equation:** Method is bounded nonlinear least squares, not PINN/Bayesian/MPC/comfort prediction; parameters remain equivalent under mismatch.
- **Implementation source:** `parameter_estimator.py`; absence audit across repository; `mismatch_dynamics.py`.
- **Relevant function/class:** SciPy `least_squares` call; fixed estimator model.
- **Relevant configuration:** Five-parameter template and soft-l1 loss.
- **Relevant result file:** R-MM-001 and R-GEO-001.
- **Confidence:** HIGH.
- **Notes / assumptions:** Negative implementation claim was verified by repository search.

### IV-C-P1

- **Manuscript subsection:** IV-C, Parameter Identifiability
- **Claim/equation:** Scaled residual sensitivity is computed with central differences where bounds permit and one-sided differences otherwise, using common torque scales across excitation comparisons.
- **Implementation source:** `identifiability_analysis.py`.
- **Relevant function/class:** `numerical_sensitivity_matrix`, `_common_torque_scales`, `compare_excitation_sets`.
- **Relevant configuration:** Parameter scaling and numerical perturbation settings.
- **Relevant result file:** Per-subject clean `sensitivity_singular_values.csv` (R-ID-002).
- **Confidence:** HIGH.
- **Notes / assumptions:** Diagnostic is local at the evaluated parameter vector.

### IV-C-P2

- **Manuscript subsection:** IV-C
- **Claim/equation:** Singular values, rank, condition number, information-shape pseudoinverse, correlations, four excitation sets, and force quartiles are computed.
- **Implementation source:** `identifiability_analysis.py`, `run_identification.py`.
- **Relevant function/class:** `analyze_identifiability`, `compare_excitation_sets`, `force_amplitude_sensitivity_analysis`, `save_identifiability_outputs`.
- **Relevant configuration:** Coupled nominal; coupled all speeds; all families/speeds; all after highest 5% Jacobian-condition removal.
- **Relevant result file:** `identifiability_summary.csv`, `parameter_correlation_matrix.csv`, force-quartile outputs (R-ID-002).
- **Confidence:** HIGH.
- **Notes / assumptions:** $I^\dagger$ is a relative covariance-shape proxy, not calibrated posterior uncertainty.

### IV-C-P3

- **Manuscript subsection:** IV-C
- **Claim/equation:** Low prediction error does not imply unique physical recovery; rank, conditioning, correlation, held-out error, and bound activity must be reported together.
- **Implementation source:** Identifiability diagnostics plus estimator bound/metric outputs.
- **Relevant function/class:** `analyze_identifiability`, `compute_torque_metrics`, parameter result object.
- **Relevant configuration:** Same fixed model/excitation conditioning.
- **Relevant result file:** R-ID-001, R-ID-002, R-MM-001.
- **Confidence:** HIGH.
- **Notes / assumptions:** Future system-identification literature citation is required for the general inference.

### IV-D-P1

- **Manuscript subsection:** IV-D, Timing Alignment and Delay Robustness
- **Claim/equation:** Positive fixed delay means $F_{obs}(t)=F_{clean}(t-\delta)$; search is -50--50 ms by 1 ms; each candidate trains parameters, validation chooses delay, test remains hidden.
- **Implementation source:** `timestamp_alignment.py`, `delay_estimator.py`, `run_delay_compensation_experiment.py`.
- **Relevant function/class:** Delay injection/alignment; `estimate_wrench_delay`.
- **Relevant configuration:** `delay_search_*`; train/validation-only estimator signature.
- **Relevant result file:** `all_delay_summary.csv` and per-case search curves (R-TIME-001).
- **Confidence:** HIGH for synthetic protocol.
- **Notes / assumptions:** Hardware delay sign/time bases are not verified.

### IV-D-P2

- **Manuscript subsection:** IV-D
- **Claim/equation:** Delay candidates use common support with 50-ms margin, >=80% coverage, <=20-ms interpolation gap; boundary estimates are flagged; offline and causal modes are distinct.
- **Implementation source:** `delay_estimator.py`, `timestamp_alignment.py`, `causal_sample_matcher.py`.
- **Relevant function/class:** `_common_candidate_support`, `_apply_common_support`, alignment/matcher functions.
- **Relevant configuration:** Common margin/support/gap constants.
- **Relevant result file:** Per-case delay metadata and R-TIME-001.
- **Confidence:** HIGH.
- **Notes / assumptions:** Offline bidirectional interpolation may consume future samples and is never presented as online behavior.

### IV-D-P3

- **Manuscript subsection:** IV-D
- **Claim/equation:** Variable-delay pipeline distinguishes state source, wrench sample, and wrench arrival times; reliable sample time is primary, arrival minus estimate is fallback; causal buffer rejects invalid temporal cases.
- **Implementation source:** `state_history_buffer.py`, `causal_sample_matcher.py`, `run_variable_delay_experiment.py`.
- **Relevant function/class:** `StateHistoryBuffer`, causal sample matcher/result types.
- **Relevant configuration:** 2-s buffer, 20-ms interpolation gap, 5-ms match error, simulated 100-ms wrench age.
- **Relevant result file:** `all_variable_delay_summary.csv`, per-case matched datasets/validity plots (R-TIME-002).
- **Confidence:** HIGH software; LOW physical timestamps.
- **Notes / assumptions:** Reliable synthetic wrench sample timestamps make the fallback less exercised in primary runs.

### IV-D-P4

- **Manuscript subsection:** IV-D
- **Claim/equation:** Windowed tracker settings and invalid-update behavior are exact; timing studies support sensitivity analysis, not a headline or robot real-time claim.
- **Implementation source:** `windowed_delay_tracker.py`, `config.py`, `run_variable_delay_experiment.py`.
- **Relevant function/class:** Windowed tracker update/search methods.
- **Relevant configuration:** 2 s, 0.5 s, [-50,80] ms/1 ms, alpha 0.5, max change 8 ms, min 25, excitation 0.05.
- **Relevant result file:** Tracking histories and R-TIME-002.
- **Confidence:** HIGH for software.
- **Notes / assumptions:** Query deadlines and source-update cadence require separate real-robot diagnostics.

### IV-E-P1

- **Manuscript subsection:** IV-E, Model-Mismatch and Generalization Analysis
- **Claim/equation:** Synthetic generator adds cubic stiffness, coupling-potential torque, odd quadratic damping, and deterministic structured residual while estimator stays five-parameter and blind to generator fields.
- **Implementation source:** `mismatch_dynamics.py`, `mismatch_subject.py`, `parameter_estimator.py`, `run_model_mismatch_experiment.py`.
- **Relevant function/class:** Mismatch torque components/generator; explicit estimator whitelist.
- **Relevant configuration:** Scenario coefficient definitions.
- **Relevant result file:** Per-scenario `generator_parameters.json`, `metadata.json` (R-MM-001).
- **Confidence:** HIGH.
- **Notes / assumptions:** Extra terms are synthetic stressors only.

### IV-E-P2

- **Manuscript subsection:** IV-E
- **Claim/equation:** Nine implemented mismatch cases and held-out interpolation/boundary/outside-domain paths are evaluated after train-only fitting.
- **Implementation source:** `mismatch_scenarios.py`, `generalization_trajectories.py`, `run_model_mismatch_experiment.py`.
- **Relevant function/class:** Scenario registry; generalization trajectory builders; experiment runner.
- **Relevant configuration:** Matched; mild/strong stiffness; mild/strong coupling; mild damping; residual; mild/strong combined.
- **Relevant result file:** `model_mismatch/summaries/generic_vs_identified_comparison.csv` (R-MM-001).
- **Confidence:** HIGH.
- **Notes / assumptions:** Outside-domain paths are explicitly labeled and excluded from task-local validity claims.

### IV-E-P3

- **Manuscript subsection:** IV-E
- **Claim/equation:** Identified and generic models use identical valid samples; torque/force/peak/NRMSE metrics and residual-feature correlations are recorded without assigning physiology.
- **Implementation source:** `mismatch_metrics.py`, `run_model_mismatch_experiment.py`.
- **Relevant function/class:** Comparison metric and residual-diagnostic functions.
- **Relevant configuration:** Common validity masks and domain labels.
- **Relevant result file:** Per-scenario `generic_vs_identified_comparison.csv`, `residual_feature_correlations.csv` (R-MM-001).
- **Confidence:** HIGH.
- **Notes / assumptions:** Correlation is a structural warning, not causal mechanism evidence.

### IV-E-P4

- **Manuscript subsection:** IV-E
- **Claim/equation:** Under mismatch, fitted parameters are equivalent predictive coefficients; useful prediction can coexist with bias, and strong mismatch may underperform generic baseline.
- **Implementation source:** Fixed estimator versus richer generator and common evaluation pipeline.
- **Relevant function/class:** `estimate_subject_parameters`; mismatch experiment comparison.
- **Relevant configuration:** Train-only fitting and interpolation/boundary/outside splits.
- **Relevant result file:** R-MM-001, including negative strong-coupling improvements.
- **Confidence:** HIGH.
- **Notes / assumptions:** Failure cases and domain membership are required whenever averages are reported.

## Equation map

### `eq:identification_signals`

- **Manuscript subsection:** IV-A
- **Claim/equation:** Observation tuple $\{q,\dot q,\ddot q,F,t,v\}$.
- **Implementation source:** `identification_dataset.py`, `parameter_estimator.py`.
- **Relevant function/class:** Dataset schema/validation and `valid_observations`.
- **Relevant configuration:** Explicit observation whitelist.
- **Relevant result file:** R-ID-001 datasets.
- **Confidence:** HIGH software.
- **Notes / assumptions:** Physical signal provenance remains P0.

### `eq:parameter_vector`

- **Manuscript subsection:** IV-B
- **Claim/equation:** $\vartheta=[\beta_m,k_h,k_k,b_h,b_k]^T$.
- **Implementation source:** `parameter_estimator.py`.
- **Relevant function/class:** `_parameter_vector`, `candidate_subject_from_parameters`.
- **Relevant configuration:** Baseline template.
- **Relevant result file:** R-ID-001 parameter estimates.
- **Confidence:** HIGH.
- **Notes / assumptions:** $\beta_m$ scales both masses and inertias.

### `eq:torque_prediction`

- **Manuscript subsection:** IV-B
- **Claim/equation:** Predicted torque is Section III inverse dynamics at observed state and candidate parameters.
- **Implementation source:** `parameter_estimator.py`, `full_dynamics.py`.
- **Relevant function/class:** `predict_joint_torque`, `inverse_dynamics`.
- **Relevant configuration:** Fixed geometry/template fields.
- **Relevant result file:** R-ID-001 prediction metrics.
- **Confidence:** HIGH.
- **Notes / assumptions:** No learned residual model.

### `eq:estimation_objective`

- **Manuscript subsection:** IV-B
- **Claim/equation:** Train-only, joint-scaled soft-l1 robust objective.
- **Implementation source:** `parameter_estimator.py`.
- **Relevant function/class:** `_torque_scales`, `estimate_subject_parameters`.
- **Relevant configuration:** `identification_loss`, torque-scale floor 1 N m.
- **Relevant result file:** R-ID-001.
- **Confidence:** HIGH.
- **Notes / assumptions:** Compact mathematical form follows SciPy soft-l1 semantics.

### `eq:parameter_bounds`

- **Manuscript subsection:** IV-B
- **Claim/equation:** Box bounds for mass scale, K, and B.
- **Implementation source:** `config.py`, `parameter_estimator.py`.
- **Relevant function/class:** Estimator bounds passed to `least_squares`.
- **Relevant configuration:** [0.6,1.6], [0,60], [0,10].
- **Relevant result file:** Per-fit metadata/estimates.
- **Confidence:** HIGH.
- **Notes / assumptions:** Mixed units apply by parameter block.

### `eq:sensitivity_matrix`

- **Manuscript subsection:** IV-C
- **Claim/equation:** $H=(\partial r/\partial\vartheta)S_\vartheta$.
- **Implementation source:** `identifiability_analysis.py`.
- **Relevant function/class:** `numerical_sensitivity_matrix`.
- **Relevant configuration:** Finite-difference/scaling settings.
- **Relevant result file:** R-ID-002 singular values.
- **Confidence:** HIGH.
- **Notes / assumptions:** Numerical local approximation.

### `eq:information_matrix`

- **Manuscript subsection:** IV-C
- **Claim/equation:** $I=H^TH$, covariance shape proportional to $I^\dagger$.
- **Implementation source:** `identifiability_analysis.py`.
- **Relevant function/class:** `analyze_identifiability`.
- **Relevant configuration:** Pseudoinverse tolerance/numerical rank handling.
- **Relevant result file:** R-ID-002 correlations/summary.
- **Confidence:** HIGH diagnostic; MEDIUM statistical interpretation.
- **Notes / assumptions:** Not a calibrated Bayesian covariance.

### `eq:fixed_delay_definition`

- **Manuscript subsection:** IV-D
- **Claim/equation:** Positive delay convention $F_{obs}(t)=F_{clean}(t-\delta)$.
- **Implementation source:** `timestamp_alignment.py`, fixed-delay metadata.
- **Relevant function/class:** Delay injection and alignment functions.
- **Relevant configuration:** Known delays 0,8,16,24,32,40 ms.
- **Relevant result file:** R-TIME-001.
- **Confidence:** HIGH synthetic.
- **Notes / assumptions:** Hardware clocks/sign remain unverified.

### `eq:variable_delay_target`

- **Manuscript subsection:** IV-D
- **Claim/equation:** Arrival-only fallback target $t_q^*=t_{F,arr}-\hat\delta(t)$.
- **Implementation source:** `causal_sample_matcher.py`, `run_variable_delay_experiment.py`.
- **Relevant function/class:** Causal target selection and state-buffer query.
- **Relevant configuration:** Tracker/matcher settings.
- **Relevant result file:** R-TIME-002.
- **Confidence:** HIGH implementation; MEDIUM primary-experiment coverage.
- **Notes / assumptions:** Reliable sample timestamp takes precedence when available.

### `eq:mismatch_generator`

- **Manuscript subsection:** IV-E
- **Claim/equation:** $\tau_{gen}=\tau_{5p}+\tau_{K3}+\tau_{Kc}+\tau_{B2}+\tau_r$.
- **Implementation source:** `mismatch_dynamics.py`, `mismatch_subject.py`.
- **Relevant function/class:** Mismatch component and total-torque generators.
- **Relevant configuration:** `mismatch_scenarios.py` coefficients.
- **Relevant result file:** R-MM-001 generator metadata/comparisons.
- **Confidence:** HIGH.
- **Notes / assumptions:** Components are hidden from the estimator and are not physiological ground truth.
