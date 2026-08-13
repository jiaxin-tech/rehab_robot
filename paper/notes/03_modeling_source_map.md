# Section III source map

Every prose paragraph is identified by the `% Trace:` marker in
`paper/sections/03_modeling.tex`. Equations are mapped separately so that a
paragraph-level source cannot conceal an unsupported mathematical statement.

## Paragraph map

### III-A-P1

- **Manuscript subsection:** III-A, Rehabilitation Scenario and Coordinate Definition
- **Claim/equation:** Supine passive hip--knee flexion is represented by a sagittal two-link lower limb with a hip-origin $x$--$z$ frame and strap-point interaction; exclusions are stated.
- **Implementation source:** `lower_limb_sim/README.md`, `lower_limb_sim/kinematics.py`, repository task/config descriptions.
- **Relevant function/class:** `forward_kinematics`.
- **Relevant configuration:** Project task assumption; $x$ along bed toward foot and $z$ upward.
- **Relevant result file:** R-KIN-001 workspace atlas is consistent with this plane; no physical result.
- **Confidence:** HIGH for project convention; MEDIUM for external biomechanical adequacy.
- **Notes / assumptions:** Sagittal/passive approximation needs a future literature citation; no active muscle, pressure distribution, out-of-plane, or robot-link model is claimed.

### III-A-P2

- **Manuscript subsection:** III-A
- **Claim/equation:** Coordinates are $[q_h,q_k]^T$ and absolute shank orientation is $q_h-q_k$; radians are internal.
- **Implementation source:** `kinematics.py`, `jacobian.py`, `full_dynamics.py`.
- **Relevant function/class:** `forward_kinematics`, `leg_jacobian`, `center_of_mass_positions`.
- **Relevant configuration:** Function argument ordering and radian-valued arrays.
- **Relevant result file:** Unit tests `test_kinematics.py`, `test_jacobian.py`, `test_full_dynamics.py` (software checks only).
- **Confidence:** HIGH.
- **Notes / assumptions:** This is an explicit implementation convention, not a generic two-link formula.

### III-A-P3

- **Manuscript subsection:** III-A
- **Claim/equation:** $L_1$ is thigh length, $L_2$ is knee-to-strap-equivalent-point length, current values are 0.42/0.30 m, and formal `ROM_PROTOCOL_V2` is hip 0--120 deg / knee 5--145 deg.
- **Implementation source:** `config.py`, `reference_measured_asymmetric.py`, `reference_closed_c2.py`.
- **Relevant function/class:** Configuration constants; active-reference ROM audit.
- **Relevant configuration:** `L1`, `L2`, `formal_experiment_manifest.json`, `FORMAL_HIP_ROM_DEG`, `FORMAL_KNEE_ROM_DEG`.
- **Relevant result file:** `reference_measured_asymmetric_metadata.json`; `workspace_atlas.csv`.
- **Confidence:** HIGH.
- **Notes / assumptions:** Earlier 5--130 deg files are legacy provenance and are not formal active inputs.

### III-B-P1

- **Manuscript subsection:** III-B, Two-Link Lower-Limb Kinematics
- **Claim/equation:** Exact implemented forward kinematics and sign explanation.
- **Implementation source:** `kinematics.py`.
- **Relevant function/class:** `forward_kinematics`.
- **Relevant configuration:** $L_1,L_2>0$; radians.
- **Relevant result file:** `workspace_atlas.csv`; software tests only for formula checks.
- **Confidence:** HIGH.
- **Notes / assumptions:** Endpoint is the strap equivalent traction point.

### III-B-P2

- **Manuscript subsection:** III-B
- **Claim/equation:** Workspace filters require $z_K,z_P,x_P\ge0$; IK uses the flexion branch and rejects configured-ROM violations.
- **Implementation source:** `workspace_atlas.py`, `kinematics.py`.
- **Relevant function/class:** Workspace sampling/filtering; `inverse_kinematics`.
- **Relevant configuration:** Legacy ROM in `config.py`.
- **Relevant result file:** `workspace_atlas.csv` (R-KIN-001).
- **Confidence:** HIGH for software behavior.
- **Notes / assumptions:** Geometric feasibility is not a clinical or robot safety guarantee.

### III-C-P1

- **Manuscript subsection:** III-C, Jacobian and Mechanical Interaction Mapping
- **Claim/equation:** Exact analytic Jacobian for the strap point.
- **Implementation source:** `jacobian.py`.
- **Relevant function/class:** `leg_jacobian`, `jacobian_diagnostics`.
- **Relevant configuration:** $L_1,L_2$ from `config.py`.
- **Relevant result file:** `test_jacobian.py` software checks; force-map results use this Jacobian.
- **Confidence:** HIGH.
- **Notes / assumptions:** Frame is the modeled sagittal rehabilitation frame.

### III-C-P2

- **Manuscript subsection:** III-C
- **Claim/equation:** Virtual work maps robot-on-leg point force to generalized torque with $J^T F$; inverse mapping uses $(J^T)^\dagger$.
- **Implementation source:** `parameter_estimator.py`, `force_mapping.py`.
- **Relevant function/class:** `measured_joint_torque`, `endpoint_force_from_joint_torque`.
- **Relevant configuration:** Point-force assumption and manuscript force-direction convention.
- **Relevant result file:** Force maps (R-DYN-001), dynamic trajectories (R-DYN-002), identification results (R-ID-001).
- **Confidence:** HIGH analytically/software; LOW for unverified physical wrench semantics.
- **Notes / assumptions:** A physical wrench must first be expressed at the modeled point and in the modeled frame with correct sign.

### III-C-P3

- **Manuscript subsection:** III-C
- **Claim/equation:** Nonfinite, determinant, condition-number, and 1000-N anomaly gates define mapping validity; physical wrench semantics remain a separate gate.
- **Implementation source:** `config.py`, `jacobian.py`, `force_mapping.py`, `identification_dataset.py`.
- **Relevant function/class:** `jacobian_diagnostics`, `endpoint_force_from_joint_torque`, `_recompute_observation_fields`.
- **Relevant configuration:** determinant $10^{-4}$, condition 100, force anomaly 1000 N.
- **Relevant result file:** Force-map audit (R-DYN-001); robot readiness (R-ROBOT-READINESS-001).
- **Confidence:** HIGH for software gates; LOW for physical meaning.
- **Notes / assumptions:** 1000 N is explicitly not a safety limit.

### III-D-P1

- **Manuscript subsection:** III-D, Subject-Specific Equivalent Dynamic Model
- **Claim/equation:** Inverse dynamics contains inertia, Coriolis/centrifugal, gravity, diagonal damping, and diagonal stiffness; exact mass matrix uses $a,b,d$.
- **Implementation source:** `full_dynamics.py`.
- **Relevant function/class:** `inverse_dynamics`, `mass_matrix`.
- **Relevant configuration:** `DynamicSubject` baseline/virtual-subject fields; $L_1$.
- **Relevant result file:** Dynamic audit R-DYN-002 and identification R-ID-001.
- **Confidence:** HIGH.
- **Notes / assumptions:** Negative off-diagonal terms follow the selected knee coordinate.

### III-D-P2

- **Manuscript subsection:** III-D
- **Claim/equation:** Exact Coriolis/centrifugal and gravity vectors, including negative knee gravity component.
- **Implementation source:** `full_dynamics.py`, `quasi_static_dynamics.py`.
- **Relevant function/class:** `coriolis_vector`, `gravity_vector`, `gravity_torque`.
- **Relevant configuration:** Gravity and virtual-subject inertial properties.
- **Relevant result file:** R-DYN-001 and R-DYN-002.
- **Confidence:** HIGH.
- **Notes / assumptions:** Support/contact forces beyond the modeled endpoint force are not modeled.

### III-D-P3

- **Manuscript subsection:** III-D
- **Claim/equation:** Passive terms are diagonal linear $K,B$; estimator varies only common mass/inertia scale plus four coefficients.
- **Implementation source:** `full_dynamics.py`, `parameter_estimator.py`.
- **Relevant function/class:** `damping_torque`, `stiffness_torque`, `candidate_subject_from_parameters`.
- **Relevant configuration:** Parameter template, neutral angles, bounds, and fixed COM proportions.
- **Relevant result file:** R-ID-001, R-LOCAL-001.
- **Confidence:** HIGH.
- **Notes / assumptions:** Cross-joint and nonlinear terms appear only in mismatch generators, never in the five-parameter estimator.

### III-D-P4

- **Manuscript subsection:** III-D
- **Claim/equation:** Estimated coefficients are task-local equivalent input--output summaries rather than unique physiology.
- **Implementation source:** Simplified estimator structure in `parameter_estimator.py`; stress tests in `mismatch_dynamics.py` and `geometry_error_scenarios.py`.
- **Relevant function/class:** `estimate_subject_parameters`; mismatch generator/evaluator.
- **Relevant configuration:** Five-parameter estimator remains fixed while generator/observations vary.
- **Relevant result file:** R-MM-001, R-GEO-001, R-ID-002.
- **Confidence:** HIGH as an evidence-bound interpretation.
- **Notes / assumptions:** Future citations should support general gray-box/equivalent-model language.

### III-E-P1

- **Manuscript subsection:** III-E, Reference-Trajectory-Centered Personalization Problem
- **Claim/equation:** Only endpoint closure is required; active path is closed, periodic, $C^2$, and asymmetric, with a small periodic spline correction.
- **Implementation source:** `reference_measured_asymmetric.py`, `run_reference_measured_asymmetric.py`.
- **Relevant function/class:** `fit_measured_asymmetric_periodic_reference`, `retime_measured_asymmetric_periodic_reference`, continuity/asymmetry audits.
- **Relevant configuration:** Correction gates and active slow timing.
- **Relevant result file:** `reference_measured_asymmetric_metadata.json` (R-REF-ASYM-001).
- **Confidence:** HIGH for software reference.
- **Notes / assumptions:** Symmetry $q(s)=q(1-s)$ is not required and is not claimed.

### III-E-P2

- **Manuscript subsection:** III-E
- **Claim/equation:** Generic endpoint-preserving local deformation abstracts implemented fixed amplitude, phase, and duration variants; continuous optimization is absent.
- **Implementation source:** `reference_local_excitation.py`, `run_reference_candidate_evaluation.py`.
- **Relevant function/class:** Fixed local-trajectory and C0--C8 candidate constructors.
- **Relevant configuration:** 3-deg amplitude perturbations, 3% phase variants, fixed duration candidates.
- **Relevant result file:** Legacy local/candidate data (R-LOCAL-001, R-CAND-001).
- **Confidence:** HIGH for fixed variants; LOW for a continuous optimizer.
- **Notes / assumptions:** Equation is labeled formulation, not a finalized algorithm.

### III-E-P3

- **Manuscript subsection:** III-E
- **Claim/equation:** Intended constrained selection minimizes a mechanical criterion; current feasibility set implements ROM, above-bed geometry, closure, force/Jacobian validity, and >=90% train-fitted local-box coverage.
- **Implementation source:** `run_reference_candidate_evaluation.py` and local-domain code in `reference_local_excitation.py`.
- **Relevant function/class:** Candidate evaluation, feasibility row construction, Pareto evaluation.
- **Relevant configuration:** Condition limit 100; local coverage threshold 0.90; ROM/workspace/closure gates.
- **Relevant result file:** `reference_candidate_feasibility.csv`, `reference_candidate_metrics.csv`, `reference_candidate_pareto.csv` (R-CAND-001).
- **Confidence:** HIGH for fixed screening; LOW for finalized subject-specific selection.
- **Notes / assumptions:** Objective $\mathcal L_{int}$ is intentionally abstract until P0 selector is frozen.

### III-E-P4

- **Manuscript subsection:** III-E
- **Claim/equation:** Velocity, acceleration, physical load, tactile, and final selection evidence are incomplete; current claim is model-based screening only.
- **Implementation source:** Candidate evaluator; separate `collection/safety_guard.py`, `collection/trajectory.py`, and robot preflight configuration.
- **Relevant function/class:** Candidate feasibility evaluation; physical preflight is separate from personalization.
- **Relevant configuration:** `experiment_safety.json`, `real_identification_config.json` are fail-closed and not imported as validated candidate limits.
- **Relevant result file:** R-CAND-001 and R-ROBOT-READINESS-001; no tactile result.
- **Confidence:** HIGH.
- **Notes / assumptions:** This negative claim follows the repository audit; no comfort/clinical conclusion is permitted.

## Equation map

### `eq:generalized_coordinates`

- **Manuscript subsection:** III-A
- **Claim/equation:** $q=[q_h,q_k]^T$.
- **Implementation source:** `kinematics.py`, `full_dynamics.py`.
- **Relevant function/class:** Public function argument ordering.
- **Relevant configuration:** Radians.
- **Relevant result file:** N/A; code convention.
- **Confidence:** HIGH.
- **Notes / assumptions:** Hip precedes knee everywhere audited.

### `eq:shank_angle`

- **Manuscript subsection:** III-A
- **Claim/equation:** $\theta_s=q_h-q_k$.
- **Implementation source:** `kinematics.py`, `jacobian.py`, `full_dynamics.py`.
- **Relevant function/class:** `forward_kinematics`, `leg_jacobian`, `center_of_mass_positions`.
- **Relevant configuration:** Flexion sign convention.
- **Relevant result file:** N/A; code convention and software tests.
- **Confidence:** HIGH.
- **Notes / assumptions:** Non-negotiable project convention.

### `eq:forward_kinematics`

- **Manuscript subsection:** III-B
- **Claim/equation:** Exact knee/strap-point forward kinematics.
- **Implementation source:** `kinematics.py`.
- **Relevant function/class:** `forward_kinematics`.
- **Relevant configuration:** `L1=0.42`, `L2=0.30` m.
- **Relevant result file:** R-KIN-001.
- **Confidence:** HIGH.
- **Notes / assumptions:** $L_2$ ends at strap equivalent traction point.

### `eq:inverse_kinematics`

- **Manuscript subsection:** III-B
- **Claim/equation:** Flexion-branch analytic IK.
- **Implementation source:** `kinematics.py`.
- **Relevant function/class:** `inverse_kinematics`.
- **Relevant configuration:** Pipeline-specific ROM.
- **Relevant result file:** R-KIN-001.
- **Confidence:** HIGH.
- **Notes / assumptions:** Reachability and ROM checks surround the analytic branch.

### `eq:jacobian`

- **Manuscript subsection:** III-C
- **Claim/equation:** $\dot p=J(q)\dot q$ with the displayed entries.
- **Implementation source:** `jacobian.py`.
- **Relevant function/class:** `leg_jacobian`.
- **Relevant configuration:** `L1`, `L2`.
- **Relevant result file:** R-DYN-001/R-DYN-002 use it.
- **Confidence:** HIGH.
- **Notes / assumptions:** Analytic derivative of `forward_kinematics`.

### `eq:force_to_torque`

- **Manuscript subsection:** III-C
- **Claim/equation:** $\tau_{meas}=J^T F_{R\to L}$.
- **Implementation source:** `parameter_estimator.py`.
- **Relevant function/class:** `measured_joint_torque`.
- **Relevant configuration:** Manuscript force sign convention.
- **Relevant result file:** R-ID-001.
- **Confidence:** HIGH software; LOW physical.
- **Notes / assumptions:** Requires verified force expression frame/reference point.

### `eq:torque_to_force`

- **Manuscript subsection:** III-C
- **Claim/equation:** $F=(J^T)^\dagger\tau_{req}$.
- **Implementation source:** `force_mapping.py`.
- **Relevant function/class:** `endpoint_force_from_joint_torque`.
- **Relevant configuration:** Determinant/condition/force gates.
- **Relevant result file:** R-DYN-001, R-DYN-002.
- **Confidence:** HIGH software.
- **Notes / assumptions:** Pseudoinverse point-force model only.

### `eq:inverse_dynamics`

- **Manuscript subsection:** III-D
- **Claim/equation:** $\tau=M\ddot q+c+g+B\dot q+K(q-q_0)$.
- **Implementation source:** `full_dynamics.py`.
- **Relevant function/class:** `inverse_dynamics`.
- **Relevant configuration:** `DynamicSubject` and baseline template.
- **Relevant result file:** R-DYN-002, R-ID-001.
- **Confidence:** HIGH.
- **Notes / assumptions:** Equivalent lower-limb model; robot/bed dynamics excluded.

### `eq:dynamic_abbreviations`

- **Manuscript subsection:** III-D
- **Claim/equation:** Definitions of $a,b,d$.
- **Implementation source:** `full_dynamics.py`.
- **Relevant function/class:** `mass_matrix` local terms.
- **Relevant configuration:** Subject masses/COM/inertias and $L_1$.
- **Relevant result file:** N/A; implemented algebra.
- **Confidence:** HIGH.
- **Notes / assumptions:** Abbreviations only, not fitted independently.

### `eq:mass_matrix`

- **Manuscript subsection:** III-D
- **Claim/equation:** Exact symmetric inertia matrix.
- **Implementation source:** `full_dynamics.py`.
- **Relevant function/class:** `mass_matrix`.
- **Relevant configuration:** Subject inertial template.
- **Relevant result file:** R-DYN-002.
- **Confidence:** HIGH.
- **Notes / assumptions:** Off-diagonal signs follow $q_h-q_k$.

### `eq:coriolis_vector`

- **Manuscript subsection:** III-D
- **Claim/equation:** Exact Coriolis/centrifugal vector.
- **Implementation source:** `full_dynamics.py`.
- **Relevant function/class:** `coriolis_vector`.
- **Relevant configuration:** Subject inertial template.
- **Relevant result file:** R-DYN-002.
- **Confidence:** HIGH.
- **Notes / assumptions:** Software analytical convention.

### `eq:gravity_vector`

- **Manuscript subsection:** III-D
- **Claim/equation:** Exact gravity vector.
- **Implementation source:** `full_dynamics.py`, `quasi_static_dynamics.py`.
- **Relevant function/class:** `gravity_vector`, `gravity_torque`.
- **Relevant configuration:** Gravity and subject template.
- **Relevant result file:** R-DYN-001/R-DYN-002.
- **Confidence:** HIGH.
- **Notes / assumptions:** Knee component sign follows coordinate convention.

### `eq:passive_terms`

- **Manuscript subsection:** III-D
- **Claim/equation:** $B=\mathrm{diag}(b_h,b_k)$ and $K=\mathrm{diag}(k_h,k_k)$.
- **Implementation source:** `full_dynamics.py`.
- **Relevant function/class:** `damping_torque`, `stiffness_torque`.
- **Relevant configuration:** Subject passive coefficients and neutral angles.
- **Relevant result file:** R-ID-001.
- **Confidence:** HIGH.
- **Notes / assumptions:** Equivalent linear terms only.

### `eq:reference_closure`

- **Manuscript subsection:** III-E
- **Claim/equation:** $q_{ref}(0)=q_{ref}(1)$ without symmetry.
- **Implementation source:** `reference_measured_asymmetric.py`.
- **Relevant function/class:** Periodic fit and continuity audit.
- **Relevant configuration:** Cubic periodic spline/correction gates.
- **Relevant result file:** R-REF-ASYM-001.
- **Confidence:** HIGH software.
- **Notes / assumptions:** Position/velocity/acceleration continuity audited in the active profile.

### `eq:candidate_parameterization`

- **Manuscript subsection:** III-E
- **Claim/equation:** $q_{cand}=q_{ref}+\Delta q$ with endpoint-preserving deformation.
- **Implementation source:** `reference_local_excitation.py`, `run_reference_candidate_evaluation.py`.
- **Relevant function/class:** Fixed amplitude, phase-warp, duration candidate generators.
- **Relevant configuration:** C0--C8 and six local trajectories.
- **Relevant result file:** R-LOCAL-001, R-CAND-001.
- **Confidence:** MEDIUM.
- **Notes / assumptions:** Abstraction covers fixed variants; continuous $\alpha$ optimization is not implemented.

### `eq:personalization_problem`

- **Manuscript subsection:** III-E
- **Claim/equation:** Constrained mechanical-interaction candidate selection.
- **Implementation source:** `run_reference_candidate_evaluation.py`.
- **Relevant function/class:** Feasibility, metric, and Pareto computations.
- **Relevant configuration:** Implemented gates and local coverage 0.90.
- **Relevant result file:** R-CAND-001.
- **Confidence:** LOW for final selection; HIGH for screening formulation.
- **Notes / assumptions:** $\mathcal L_{int}$ and deterministic subject-specific selection remain P0 work.
