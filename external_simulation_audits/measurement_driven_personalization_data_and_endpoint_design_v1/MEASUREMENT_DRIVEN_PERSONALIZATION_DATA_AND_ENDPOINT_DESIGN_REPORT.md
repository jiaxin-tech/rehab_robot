# Measurement-Driven Personalization Data and Endpoint Design V1

## Formal outcome

`PRIMARY_MECHANICAL_ENDPOINT_DEFINITION_INCOMPLETE`

`WRENCH_FRAME_SEMANTICS_NOT_VERIFIED`  
`TASK_DIRECTION_REQUIRES_EXPERIMENTAL_VALIDATION`  
`BASE_WRENCH_ROTATION_VERIFIED = false`  
`FILTER_NOT_YET_FROZEN`

This was a source/document design audit only: zero robot connection, zero motion, zero human collection, zero model/PINN training and zero BO.

## Q1. Channels sufficiently defined for future research use

The software schema and provenance paths for q, host-derived dq, TCP, raw wrench/joint-torque arrays and timing are sufficiently specified to plan future validation. None is currently a formally validated physical research outcome. q/TCP are future state/context candidates after setup validation; dq/ddq are derived candidates. Cartesian force is blocked as primary endpoint input. Joint/cartesian torque remain secondary candidates. Tactile is only a placeholder.

## Q2. Exact current wrench semantics

The local SDK defines 3 Cartesian force values in N, 3 Cartesian torque values in N*m, 6 measured joint torques and 6 controller-model-derived external joint torques in N*m. `getEndTorque` accepts documented world/flange/tool requests and current code requests world. Sign, compensation/bias, exact moment point for all frames, source timestamp/cadence and synchronization are not proved.

## Q3. Base rotation

Not justified. Offline math verifies only a convention, not the physical SDK convention. Known-direction, multiple-orientation force tests and known-lever-arm moment tests are required. Force projection can omit moment translation only if moment is excluded; all force semantics still need validation.

## Q4. Task direction

The physical strap/pull line of action is the most defensible. A registered equivalent traction-point-to-hip line can approximate it only after experimental geometry validation. TCP tangent and fixed bed axes remain diagnostics. No direction is selected from lower RMS.

## Q5. Mathematical candidate

For valid samples in a common validated frame, `F_task(t_i)=dot(F_interaction(t_i), d_task(t_i))`, with unit `d_task`; then `J_force=sqrt(sum(w_i F_task(t_i)^2)/sum(w_i))`, in N. Signed projection is retained, while RMS is naturally sign-invariant. Exact mask, quadrature, transient, bias and filter remain unfinished dependencies.

## Q6. Bias/filter/delay/synchronization

Always retain raw data. Validate an unloaded pre-episode zero candidate against static, pose-dependent and drift behavior before selection; do not call strap preload zero. No filter cutoff is frozen. Host `perf_counter_ns` is the master clock; retain all query bounds/ages/skews, never zero-fill, and do not reuse simulation delay. Numeric age/gap/skew/missing limits remain null pending real evidence and review.

## Q7. Episode validity

A 24 s episode must complete normally, have no invalidating safety event, valid operation/tracking, finite wrench/state, monotonic timing, and pass future frozen freshness/skew/gap/missing rules plus validated wrench/task-direction semantics. Failure produces a null endpoint and no gray-box/BO observation.

## Q8. Future evidence

Preregister repeated identical approved trials to estimate mean/SD/CV, drift and design-appropriate ICC, then test prespecified small V3 perturbations against repeatability noise. Repeat count and thresholds are not invented here.

## Q9. Tactile integration

Use a nullable timestamped raw/calibrated matrix stream with per-cell missing/saturation masks, calibration and placement provenance. Pressure features remain secondary and pressure is not comfort.

## Q10. Single next stage

`WRENCH_FRAME_AND_TASK_DIRECTION_RESOLUTION_V1`

Definition is not ready for endpoint validation until frame/sign/point and physical task direction are resolved. Status remains `NOT_HUMAN_READY / NOT_ROBOT_APPROVED`; the next stage was not executed.
