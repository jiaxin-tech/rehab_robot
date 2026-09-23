# REAL_MEASUREMENT_VALIDATION_ANALYSIS_V1

## 1. Implementation status and boundary

```text
REAL_MEASUREMENT_VALIDATION_ANALYSIS_V1 =
IMPLEMENTED_WITH_LIMITATIONS

MEASUREMENT_VALIDITY = INSUFFICIENT
SAME_TRAJECTORY_REPEATABILITY = INSUFFICIENT
TRAJECTORY_SENSITIVITY = INSUFFICIENT
```

The three analysis levels, evidence hierarchy, normalized input interface, offline runner, JSON/CSV outputs, and targeted tests are implemented. The current three evidence results remain `INSUFFICIENT` because this stage used only a deterministic synthetic demo and did not consume real robot or dummy-leg logs.

The implementation is offline-only. It does not import a robot adapter, connect to hardware, issue motion, change safety/control logic, run personalization algorithms, or train a PINN. It does not hard-code unverified physical or safety thresholds.

## 2. Three-level hierarchy

```text
Level 1: MEASUREMENT_VALIDITY
        ↓ must be SUPPORTED
Level 2: SAME_TRAJECTORY_REPEATABILITY
        ↓ must be SUPPORTED
Level 3: TRAJECTORY_SENSITIVITY
        ↓ must be SUPPORTED
Future subject × trajectory evaluation
```

Each level computes descriptive results even when upstream evidence is incomplete, but it cannot publicly declare support while an earlier level is not `SUPPORTED`. Decision thresholds must be supplied by a reviewed experiment protocol; absent criteria produce `INSUFFICIENT`, and failed supplied criteria produce `FAILED`.

The framework exposes only:

```text
SUPPORTED / INSUFFICIENT / FAILED
```

for the three requested levels. It never outputs `PERSONALIZATION_SUPPORTED`.

## 3. Normalized input interface

The runner accepts one `REAL_MEASUREMENT_VALIDATION_INPUT_V1` JSON document. Static records may be embedded or referenced through `static_csv_paths`; trajectory episodes may be embedded or referenced through `episode_json_paths`.

Minimal structure:

```json
{
  "schema_version": "REAL_MEASUREMENT_VALIDATION_INPUT_V1",
  "evidence_classification": "REAL_MEASUREMENT",
  "static_csv_paths": ["static_validation_labels.csv"],
  "episodes": [
    {
      "episode_id": "episode_001",
      "entity_id": "dummy_leg_01",
      "entity_kind": "DUMMY_LEG",
      "beta_flex": 0.0,
      "beta_extend": 0.0,
      "rom_profile_id": "rom_01",
      "rom_profile_version": "v1",
      "hardware_setup_id": "setup_01",
      "measurement_source": "recorded_robot_log",
      "metadata": {},
      "samples": []
    }
  ],
  "task_direction": null,
  "criteria": {
    "level_1": {},
    "level_2": {},
    "level_3": {}
  }
}
```

Required provenance separates `entity_id`, beta, ROM identity/version, hardware setup, measurement source, and episode identity. Tool, workobject, TCP, payload, robot, and controller metadata are compared within repeated conditions when available. A changed setup is therefore not silently pooled as trajectory variability.

## 4. Level 1 — measurement validity

### Inputs

- PRE/LOAD/POST phase and repeat identity;
- raw `Fx/Fy/Fz` and optional `Mx/My/Mz`;
- query start/end or query duration;
- source and robot-state validity plus invalid reasons;
- robot-state timestamp, joint state, and TCP state when available;
- known reference-load vector when available;
- pose, direction, load level, measurement source, tool/TCP/payload, robot, and controller provenance.

### Analysis

For every static repeat cell, the implementation retains raw PRE, LOAD, and POST samples and computes:

\[
\Delta F=\bar F_{LOAD}-\frac{\bar F_{PRE}+\bar F_{POST}}{2}.
\]

It also reports baseline bias, baseline standard deviation, POST-minus-PRE drift, optional torque counterparts, query timing, missing/invalid counts, known-load sign agreement, vector-direction cosine, and across-repeat load-contrast variability.

Missing or nonfinite values remain `None`/blank and are excluded with counts; no absent measurement is replaced with zero.

## 5. Task direction

Raw Cartesian `Fx/Fy/Fz` are always retained. The analyzer never assumes that X is the task direction.

`task_direction_projection` is created only when the input explicitly supplies a finite nonzero direction vector. Its source, frame, and `geometry_validated` state remain in the output. An unvalidated projection may be inspected diagnostically but is not eligible as a decision feature. The current project therefore keeps task-direction projection unavailable for decision use until physical geometry is verified.

## 6. Level 2 — same-trajectory repeatability

Episodes are grouped only when all of the following match:

```text
entity
beta_flex / beta_extend
ROM profile and version
hardware setup ID
measurement source
```

Within each group, the analyzer reports:

- valid episode count and fraction;
- setup-metadata consistency;
- episode-level feature mean, standard deviation, range, and absolute coefficient of variation;
- phase-aligned pairwise time-series correlation;
- pairwise time-series RMS difference;
- full-cycle RMS, flexion RMS, extension RMS, and absolute peak for each available channel;
- branch- and joint/component-specific variability when those channels are available.

Branch metrics require explicit sample branch labels. The analyzer does not infer flexion/extension from an assumed phase convention.

## 7. Level 3 — trajectory sensitivity

Only the small set of beta conditions supplied in the real experiment is compared; the analyzer does not generate or scan the 625-point V3 grid.

For each predeclared feature, it calculates condition means, the largest pairwise between-trajectory effect, and pooled within-trajectory standard deviation:

\[
SNR_{trajectory}=
\frac{|\Delta_{between}|}{\sigma_{within}}.
\]

No fixed SNR pass number is built into the implementation. The experiment protocol must provide the decision feature IDs and threshold before interpretation.

In parallel, phase-normalized curves are retained. The analyzer reports condition-mean curves, maximum local difference, its phase, the responsible beta pair, local within-condition variability, and time-local SNR. This prevents full-cycle RMS from being the only result and makes short response windows visible.

## 8. Measurement provenance and E2 compatibility

| Quantity | Framework provenance |
|---|---|
| Raw Cartesian force/torque log channels | `ROBOT_REPORTED_RAW_CHANNEL` |
| Robot-reported measured joint torque | `ROBOT_REPORTED_RAW_CHANNEL` |
| External joint torque | `MODEL_DERIVED` |
| Hip/knee mechanical decomposition | `MODEL_DERIVED` |
| Explicit task-direction projection | `DERIVED_FROM_EXPLICIT_TASK_DIRECTION` |

“Robot-reported raw” means that the value is copied from the robot data path without being synthesized by this analyzer. It does not establish force reference point, compensation, frame correctness, synchronization, or physical validity; those remain Level 1 questions.

The transferable parts of the simulated E2 work are representational: preserve flexion/extension separation, component separation, RMS, peaks, time-local responses, and a subject's own reference when normalization later becomes justified. The current real framework does not hard-code the E2 max scalar, simulated hip/knee decomposition, or simulated reference normalization as a real endpoint.

## 9. Offline demo and outputs

Run the deterministic demo:

```bash
PYTHONPATH=. python3 -m scripts.run_real_measurement_validation_analysis \
  --demo \
  --output-dir /private/tmp/real_measurement_validation_demo_v1
```

The demo exercises 3 PRE/LOAD/POST repeats and 9 episodes covering 3 beta conditions with 3 repeats each. Its supplied demo criteria pass internally, proving that all three calculation paths execute. Its public evidence statuses remain `INSUFFICIENT` because the input is explicitly classified `OFFLINE_SYNTHETIC_DEMO_ONLY`; the future subject×trajectory gate remains closed.

Outputs:

- `validation_summary.json`;
- `level_1_static_cells.csv`;
- `episode_features.csv`;
- `level_2_repeatability_groups.csv`;
- `level_3_trajectory_effects.csv`;
- `level_3_time_local_effects.csv`.

Source logs are read-only and are not rewritten.

## 10. Q1–Q10

**Q1. Level 1–3 分别需要什么输入？**

Level 1 needs phase-labelled static wrench/state records, validity, timing, provenance, and optional known loads. Level 2 needs repeated full episodes with episode ID, beta, frozen ROM identity/version, hardware setup, time series, validity, branch labels, and provenance. Level 3 needs at least two preselected beta conditions with repeated valid episodes and protocol-supplied decision criteria, after Levels 1 and 2 are supported.

**Q2. 当前哪些量可以直接来自机器人？**

The log can directly carry robot-reported Cartesian force/torque, measured joint torque, RT joint position, TCP pose, robot/status fields, plus host-recorded query and sample timestamps. These are direct data-path fields, not yet proof of correct physical semantics or synchronization.

**Q3. 哪些量仍然是 model-derived？**

Hip/knee mechanical decomposition and external joint torque are model/controller-derived. Task-direction projection is geometry-derived and remains decision-ineligible until its direction/frame is physically validated. E2 normalization, branch maximum, and any scalar real mechanical preference endpoint are also not yet validated real measurements.

**Q4. 如何计算 same-trajectory repeatability？**

Group episodes only when entity, beta, ROM, hardware setup, and measurement source match; verify setup metadata; then compute feature standard deviation/CV and pairwise phase-aligned correlation/RMS difference, including branch and component features where available.

**Q5. 如何比较 trajectory effect 与 repeatability noise？**

For each predeclared feature, compare the beta-condition mean difference with the pooled within-beta episode standard deviation and report `trajectory_snr = abs(delta_between) / sigma_within`. Report the observed values before applying externally supplied protocol criteria.

**Q6. 如何避免 full-cycle RMS 再次掩盖局部差异？**

Retain full-cycle, flexion, extension, peak, and phase-normalized time-local outputs simultaneously. Report the maximum local effect and phase rather than reducing every episode to one scalar.

**Q7. 当前 E2 哪些部分可以迁移到真实数据，哪些不能？**

Its anti-cancellation representation—branches, components, RMS, peaks, time-local response, and own-reference concept—can guide real analysis. The simulated E2 scalar, simulated hip/knee components, and reference normalization cannot become the real endpoint without measurement and geometry validation.

**Q8. 假腿实验可以支持什么结论？**

It can support measurement-pipeline validity, execution consistency, episode repeatability, mechanical-signal sanity, and trajectory sensitivity for that setup. It cannot support patient personalization necessity, comfort, clinical safety, benefit, or effectiveness.

**Q9. 什么 evidence 才允许进入 future subject×trajectory evaluation？**

Real measurements must pass externally reviewed Level 1 validity criteria, Level 2 repeated-episode criteria, and Level 3 predeclared trajectory-sensitivity criteria in sequence. Only then may a separate study test reproducible cross-subject ranking or branch-response differences; this framework still does not declare personalization.

**Q10. 下一次拿到第一批真实日志后具体运行哪个分析入口？**

Prepare the normalized V1 JSON with paths to the immutable static records and episode data, then run:

```bash
PYTHONPATH=. python3 -m scripts.run_real_measurement_validation_analysis \
  --input path/to/real_measurement_validation_input_v1.json \
  --output-dir path/to/new_analysis_output
```

Review `validation_summary.json` strictly in Level 1 → Level 2 → Level 3 order. Do not start subject×trajectory or BO analysis unless all three real-measurement statuses are `SUPPORTED`.

## 11. Targeted validation

```text
PYTHONPATH=. pytest -q tests/test_real_measurement_validation_analysis_v1.py
9 passed
```

The tests cover PRE/LOAD/POST analysis, missing and invalid samples, bias/drift, same-beta grouping, different-beta grouping, within-vs-between variation, time-local response, branch separation, metadata consistency, task-direction behavior, provenance, hierarchy blocking, offline-demo classification, and absence of an automatic personalization conclusion.
