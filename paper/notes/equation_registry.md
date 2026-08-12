# Equation registry

`IMPLEMENTED` denotes an equation directly evaluated by current code.
`FORMULATION` denotes a manuscript-level problem statement consistent with, but
not fully realized by, the current fixed-candidate software. Equation labels
refer to `paper/sections/03_modeling.tex` and `04_identification.tex`.

| Equation label | Purpose | Status | Implementation/project source | Sign and unit audit | Outstanding issue |
|---|---|---|---|---|---|
| `eq:generalized_coordinates` | Define $q=[q_h,q_k]^T$ | IMPLEMENTED convention | `kinematics.py`, `full_dynamics.py` argument order | rad internally; hip then knee | Add literature citation only for model scope, not convention |
| `eq:shank_angle` | Define absolute shank orientation | IMPLEMENTED | `kinematics.forward_kinematics`, `jacobian.leg_jacobian`, `full_dynamics.center_of_mass_positions` | $	heta_s=q_h-q_k$ checked; never use sum | None |
| `eq:forward_kinematics` | Knee and strap-point positions | IMPLEMENTED | `kinematics.forward_kinematics` | $L$ in m; output m; signs match shank convention | Physical hip/frame registration unvalidated |
| `eq:inverse_kinematics` | Flexion-branch IK | IMPLEMENTED | `kinematics.inverse_kinematics` | $D$ dimensionless; `acos` returns nonnegative flexion; rad | Knee ROM differs between legacy and active pipelines |
| `eq:jacobian` | Map joint velocity to strap-point velocity | IMPLEMENTED | `jacobian.leg_jacobian` | J units m/rad; analytic signs match finite-difference tests | Physical point/frame unvalidated |
| `eq:force_to_torque` | Reconstruct generalized interaction torque | IMPLEMENTED | `parameter_estimator.measured_joint_torque` | $J^T F$ gives N m; $F$ is robot-on-leg by manuscript convention | Real wrench sign, frame, reference point, compensation, timing unverified |
| `eq:torque_to_force` | Map required torque to endpoint force | IMPLEMENTED | `force_mapping.endpoint_force_from_joint_torque` | pseudoinverse of $J^T$; output N | Point force is nonunique/fragile near singularities; physical load limit absent |
| `eq:inverse_dynamics` | Five-parameter equivalent inverse dynamics | IMPLEMENTED | `full_dynamics.inverse_dynamics` | Mddq, c, g, Bdq, K(q-q0) all N m | Robot/strap dynamics excluded; model-mismatch limits must accompany results |
| `eq:dynamic_abbreviations` | Compact two-link inertia terms | IMPLEMENTED algebra | `full_dynamics.mass_matrix` | $a,b,d$ in kg m2 | Not independent estimated parameters |
| `eq:mass_matrix` | Coupled inertia matrix | IMPLEMENTED | `full_dynamics.mass_matrix` | symmetric; negative off-diagonal follows $q_h-q_k$ | Numerical positive-definiteness tested in software only |
| `eq:coriolis_vector` | Coriolis/centrifugal torque | IMPLEMENTED | `full_dynamics.coriolis_vector` | N m; signs checked against code/energy convention | No independent physical validation |
| `eq:gravity_vector` | Gravity torque | IMPLEMENTED | `full_dynamics.gravity_vector`, `quasi_static_dynamics.gravity_torque` | N m; knee term negative under coordinate convention | Bed/support/contact contributions excluded |
| `eq:passive_terms` | Diagonal linear equivalent stiffness/damping | IMPLEMENTED | `full_dynamics.damping_torque`, `stiffness_torque` | K: N m rad-1; B: N m s rad-1 | No cross-joint or nonlinear terms in estimator |
| `eq:reference_closure` | Permit closed asymmetric cycles | IMPLEMENTED invariant | `reference_measured_asymmetric.fit_measured_asymmetric_periodic_reference` | q in rad; endpoint value equality | Physical tracking closure unvalidated |
| `eq:candidate_parameterization` | Abstract reference-local deformation | FORMULATION with fixed instances implemented | `reference_local_excitation.py`, `run_reference_candidate_evaluation.py` | additive angle deformation in rad; endpoint-preserving phase/amplitude variants | No continuous optimizer about active asymmetric reference |
| `eq:personalization_problem` | State intended constrained selection | FORMULATION | Fixed feasibility/Pareto logic in `run_reference_candidate_evaluation.py` | objective terms require explicit normalization/units | Final objective, selector, active-reference rerun, and physical limits missing |
| `eq:identification_signals` | Define estimator observation tuple and validity | IMPLEMENTED interface | `identification_dataset.py`, `parameter_estimator.valid_observations` | q rad, dq rad/s, ddq rad/s2, F N, t s | Real timestamp semantics and force frame unverified |
| `eq:parameter_vector` | Define five estimated coefficients | IMPLEMENTED | `parameter_estimator._parameter_vector`, `candidate_subject_from_parameters` | dimensionless mass scale; K/B units as above | Coefficients are equivalent, not physiological |
| `eq:torque_prediction` | Define model-predicted torque | IMPLEMENTED | `parameter_estimator.predict_joint_torque` | output N m | Baseline geometry/template treated as fixed |
| `eq:estimation_objective` | Train-only joint-scaled robust fit | IMPLEMENTED via SciPy residual interface | `parameter_estimator.estimate_subject_parameters` | residual channels dimensionless after >=1 N m scaling | Equation expresses soft-l1 cost conceptually; SciPy applies its standard loss to residuals |
| `eq:parameter_bounds` | Bound mass scale, K, and B | IMPLEMENTED | `config.py` estimator bounds; `parameter_estimator.estimate_subject_parameters` | Cartesian product notation carries mixed units by block | Bounds need physical-experiment justification/citation |
| `eq:sensitivity_matrix` | Define scaled local numerical sensitivity | IMPLEMENTED numerical approximation | `identifiability_analysis.numerical_sensitivity_matrix` | normalized torque residual per scaled parameter; dimensionless columns | Local and estimate-dependent only |
| `eq:information_matrix` | Define information shape and correlation proxy | IMPLEMENTED diagnostic | `identifiability_analysis.analyze_identifiability` | $H^T H$ dimensionless under scaling | Pseudoinverse is not a calibrated posterior covariance |
| `eq:fixed_delay_definition` | Fix positive-delay sign convention | IMPLEMENTED synthetic convention | `timestamp_alignment.py`, delay experiment metadata | seconds internally; positive means observed force lags source | Must verify sign/time semantics on hardware |
| `eq:variable_delay_target` | Choose target state time for arrival-only wrench | IMPLEMENTED fallback | `causal_sample_matcher.py`, `state_history_buffer.py`, `run_variable_delay_experiment.py` | all host/source times in seconds | Primary synthetic runs often have reliable wrench sample timestamps, so fallback evidence is mainly tests |
| `eq:mismatch_generator` | Separate estimator model from richer virtual generator | IMPLEMENTED generator, excluded from estimator | `mismatch_dynamics.py`, `mismatch_subject.py`, `run_model_mismatch_experiment.py` | all additive terms are generalized torque in N m | Scenario coefficients are synthetic stressors, not physiology |

## Gate constants used around equations

- Jacobian validity: $|\det J|\ge10^{-4}$ and
  $\kappa_2(J)\le100$ (`config.py`, `jacobian.py`, `force_mapping.py`).
- Software force anomaly: endpoint-force magnitude no greater than 1000 N;
  explicitly not a safety threshold.
- Fixed-delay search: -50 to 50 ms in 1-ms steps, 50-ms common margin,
  minimum 80% common support, maximum 20-ms interpolation gap.
- Causal state matching: 2-s state buffer, maximum 20-ms interpolation gap,
  maximum 5-ms match error; simulation additionally audits 100-ms wrench age.
- Delay tracker: 2-s window, 0.5-s update interval, -50 to 80 ms in 1-ms
  steps, smoothing 0.5, maximum 8-ms update, minimum 25 samples, excitation
  threshold 0.05.
