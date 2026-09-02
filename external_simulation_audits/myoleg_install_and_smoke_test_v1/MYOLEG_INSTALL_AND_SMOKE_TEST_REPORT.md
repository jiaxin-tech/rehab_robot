# MYOLEG_INSTALL_AND_SMOKE_TEST_V1

## Final determination

`MYOLEG_VIRTUAL_PATIENT_BASE_FEASIBLE_WITH_LIMITATIONS`

MyoSuite/MyoLeg can be installed and run headlessly on this Apple Silicon Mac.
The actual MyoLeg model exposes the bilateral hip/knee/ankle structure, muscle
actuators, state, generalized-force fields, shank bodies/sites, and external
force APIs needed for a later virtual-patient feasibility stage. It is not yet
a supine rehabilitation model. The macOS native viewer is unavailable in the
current uv-created Python 3.10 environment, and the installed walking model's
knee range ends at 120 degrees rather than the rehab_robot formal 145 degrees.

This is an installation/API/engineering smoke audit only. It is not human,
clinical, physiological-validity, robot, or rehabilitation-effect evidence.

## Scope and isolation

- Git branch: `sync/mac-20260813`
- Git HEAD at audit start: `20f6eac MODEL_TRUST_FINITE_VALIDATION_STRESS_TEST_V1`
- Existing dirty worktree was preserved. The pre-existing BO files and the
  modified V1 regression test were not edited by this stage.
- Project environment: `/Users/fengjiaxin/VScodeprojects/rehab_robot/.venv`
- Project `.venv` Python: 3.13.3
- Project `.venv` freeze SHA before and after:
  `124535dc0dec4a75c214bb8e99b050a3fe0baa95b007797e788012a9d253a6f6`
- Project `.venv/pyvenv.cfg` SHA before and after:
  `58fff0beb08f5a552be7c53be85dae2b9328a7f50ce05130fcb04a4e06ab9c66`
- `CURRENT_REHAB_ROBOT_ENV_UNCHANGED = true`
- No `lower_limb_sim`, reference, five-parameter model, BO artifact, robot,
  hardware, SDK, control, or safety implementation was changed.
- No renderer was used in headless tests. No RL policy was trained. No 24 s
  reference, candidate landscape, PINN, BO, or virtual population was run.

## Machine and isolated environment

- macOS: 26.4.1 (build 25E253)
- architecture: arm64 / Apple Silicon
- shell: `/bin/zsh`
- unqualified `python`: unavailable
- default `python3`: 3.13.3
- default pip: 26.0.1 (Python 3.13)
- project `.venv` pip: 26.1
- conda: unavailable
- uv: 0.9.7
- isolated environment: `/Users/fengjiaxin/.virtualenvs/myosuite-v2`
- isolated Python: CPython 3.10.19
- isolated pip: not installed by `uv venv`; package operations used `uv pip`

Installed package versions:

| Package | Version |
|---|---:|
| MyoSuite | 2.12.2 |
| MuJoCo | 3.6.0 |
| Gymnasium | 1.2.3 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| JAX / JAXLIB | not installed |
| Stable-Baselines3 / MJRL | not installed |

The official current repository declares Python `>=3.10,<3.14` and documents
isolated uv installation, `python -m myosuite.tests.test_myo`, and macOS
visualization through `mjpython`:

- https://github.com/MyoHub/myosuite/blob/main/README.md
- https://github.com/MyoHub/myosuite/blob/main/pyproject.toml

## Official installation test

`MYOSUITE_OFFICIAL_TEST = PASS_AFTER_ISOLATED_DEPENDENCY_REMEDIATION`

Attempt 1, with the exact PyPI base dependencies, failed in
`myoChallengeBimanual-v0` because
`myosuite/envs/myo/myochallenge/bimanual_v0.py` imports
`scipy.spatial.transform`, while MyoSuite 2.12.2 does not declare SciPy as a
base dependency. This was an upstream packaging/dependency gap, not a MyoLeg,
MuJoCo, XML, Apple Silicon, or graphics failure.

SciPy 1.15.3 was then added only to `myosuite-v2`. Attempt 2 completed:

- exit code: 0
- unittest result: 3 test groups, `OK`
- reported environment instantiations: 392
- reference-motion playback checks: 190
- failed environments after remediation: none
- wall time: 45.60 s
- warnings: Gymnasium float32 Box casts and upstream NumPy deprecations

The complete two-attempt record is in `official_test_output.txt`.

## Registered MyoLeg environments

The installed registry contains 461 total IDs. Fifteen match `Leg`/`myoLeg`:

- `myoLegWalk-v0`, `myoLegStandRandom-v0`
- `myoLegHillyTerrainWalk-v0`, `myoLegRoughTerrainWalk-v0`,
  `myoLegStairTerrainWalk-v0`
- the corresponding five `myoFatiLeg*` variants
- the corresponding five `myoSarcLeg*` variants

`myoLegWalk-v0` exists and was used; no fallback ID was needed. The exact
entry points and episode limits are in `registered_myoleg_envs.json`.

## Headless smoke test

`MYOLEG_HEADLESS_SMOKE_TEST = PASS`

- observation shape: `(403,)`
- action shape: `(80,)`
- random environment steps: 1,000
- environment/control dt: 0.01 s
- MuJoCo integration dt: 0.001 s
- integration steps per environment step: 10
- simulated time: 10.0 s
- wall time: 0.620 s
- throughput: 1,612.9 environment steps/s
- realtime factor: 16.13x
- exceptions: 0
- NaN/Inf: none

Random muscle controls caused the walking task to return termination signals
after early falling (965 of the returned step results were terminal; one
time-limit truncation occurred). Continuing to call the engine remained finite,
but this does not demonstrate a stable walking policy or a valid post-terminal
episode. The separate zero-control throughput run remained finite through
100,000 environment steps.

## macOS visualization

- `mjpython` path:
  `/Users/fengjiaxin/.virtualenvs/myosuite-v2/bin/mjpython`
- executable exists: yes
- minimal `mjpython` MyoLeg probe: FAIL before model/viewer creation
- `MYOLEG_HEADLESS_AVAILABLE = true`
- `MYOLEG_VISUALIZATION_AVAILABLE = false`

The MuJoCo app loader could not locate `libpython3.10.dylib` for the uv-managed
CPython. No symlink, alternate system Python, upstream source modification, or
system package-manager workaround was applied. This is therefore a viewer
environment limitation, not a headless-simulation failure. Full loader output
is in `visualization_test_output.txt`.

## Actual model structure

Loaded model: `myoLegWalk-v0`

- joints: 29
- qpos size (`nq`): 35
- generalized velocity/DoF size (`nv`): 34
- equality constraints: 14
- bodies: 16
- sites: 382
- tendons: 80
- muscle/tendon actuators: 80
- all 80 actuator transmissions: `mjTRN_TENDON`
- all actuator control ranges: `[0, 1]`

The 29 joint objects include a six-DoF free `root`, bilateral primary joint
chains, knee auxiliary translation/rotation joints, and patellar joints. The
14 active equality constraints couple seven auxiliary/patellar joint objects
per side to `knee_angle_r/l`. `nv=34` is the model's nominal generalized
velocity dimension; it should not be mislabeled as 34 independent anatomical
DoFs after constraints.

Primary bilateral structure read from `MjModel`:

| Region | Actual joint/body | Type | Range |
|---|---|---|---|
| pelvis/root | `root` on body `root`; pelvis is its child | free | unbounded |
| right hip flexion | `hip_flexion_r` / `femur_r` | hinge | -30 to 120 deg |
| right hip other | `hip_adduction_r`, `hip_rotation_r` | hinges | -50..30, -40..40 deg |
| right knee | `knee_angle_r` / `tibia_r` | hinge | 0 to 120 deg |
| right ankle | `ankle_angle_r` / `talus_r` | hinge | -40 to 30 deg |
| right distal foot | `subtalar_angle_r`, `mtp_angle_r` | hinges | -20..20, -30..30 deg |
| left hip flexion | `hip_flexion_l` / `femur_l` | hinge | -30 to 120 deg |
| left hip other | `hip_adduction_l`, `hip_rotation_l` | hinges | -50..30, -40..40 deg |
| left knee | `knee_angle_l` / `tibia_l` | hinge | 0 to 120 deg |
| left ankle | `ankle_angle_l` / `talus_l` | hinge | -40 to 30 deg |
| left distal foot | `subtalar_angle_l`, `mtp_angle_l` | hinges | -20..20, -30..30 deg |

Important compatibility limitation: rehab_robot `ROM_PROTOCOL_V2` has knee
5–145 degrees, while this MyoLeg walking model's actual knee range is 0–120
degrees. This stage did not change either range. Direct full-ROM replay is not
currently established and must be addressed explicitly before integration.

## Muscles, shank sites, and passive-force APIs

At the actual reset state, the sparse MuJoCo `actuator_moment` mapping was used
to inspect mechanical coupling rather than inferring it from anatomical names:

- hip-related actuator transmissions: 50
- knee-related actuator transmissions: 26
- local hip-and-knee coupling: 14 actuators
- preliminary bilateral names: `bflh`, `grac`, `recfem`, `sart`, `semimem`,
  `semiten`, and `tfl`

This is a local mechanical moment-arm classification at the inspected state.
`BIARTICULAR_CLASSIFICATION_REQUIRES_FURTHER_MODEL_INSPECTION` remains true for
a formal across-ROM anatomical classification.

The model contains `tibia_r` and `tibia_l` bodies and 102 sites attached to
them. Particularly transparent marker candidates include `RTB1/2/3`,
`LTB1/2/3`, and the bilateral tibial-plateau sites. These prove that shank-local
points exist; they do not automatically define the correct strap attachment.

Available state/force arrays from the actual `MjData`:

| Field | Shape |
|---|---:|
| `qpos` | `(35,)` |
| `qvel` | `(34,)` |
| `actuator_force` | `(80,)` |
| `qfrc_passive` | `(34,)` |
| `qfrc_actuator` | `(34,)` |
| `qfrc_constraint` | `(34,)` |
| `qfrc_applied` | `(34,)` |
| `xfrc_applied` | `(16, 6)` |

With zero muscle control for 250 steps, the simulation remained finite and
`qfrc_passive` was nonzero (final L2 norm 0.01396); actuator force also remained
nonzero. This confirms a model passive response/API path. It does not mean zero
control equals a fully passive real human.

## Supine hip-knee rehabilitation feasibility

| Requirement | Rating | Evidence / limitation |
|---|---|---|
| Fix pelvis | REQUIRES_MODIFICATION | free `root`; pelvis descends from root and can structurally be welded/locked |
| Fix non-target leg | REQUIRES_MODIFICATION | separate bilateral joint chains are present |
| Fix ankle/foot | REQUIRES_MODIFICATION | ankle, subtalar, and MTP hinges are explicit |
| Lock hip non-sagittal DoFs | REQUIRES_MODIFICATION | adduction and rotation are separate hinges |
| Retain knee flexion only | LIKELY | primary knee hinge plus existing auxiliary equality couplings |
| Read target hip/knee q,dq,torque terms | YES | explicit qpos/dof addresses and generalized-force arrays |
| Define shank strap point | LIKELY / REQUIRES_MODIFICATION | tibia bodies and sites exist; dedicated validated strap site is not defined |
| Apply later virtual traction | YES | `xfrc_applied` and `mujoco.mj_applyFT` are available |

The structure supports constructing a derived supine, pelvis-fixed,
single-target-leg sagittal environment, but doing so requires a deliberately
new model/specification. The current walking environment is not that model.

## Performance benchmark

Zero-control headless engine/task throughput:

| Requested steps | Wall time | Steps/s | Realtime factor | NaN/exception |
|---:|---:|---:|---:|---:|
| 1,000 | 0.584 s | 1,712 | 17.12x | none |
| 10,000 | 5.784 s | 1,729 | 17.29x | none |
| 100,000 | 57.219 s | 1,748 | 17.48x | none |

Peak process RSS was approximately 291 MB. At dt 0.01 s, a 24 s trajectory
needs about 2,400 environment steps / 24,000 MuJoCo integration steps. Using
the 100,000-step rate gives the following serial engineering extrapolation:

| Trajectories | Estimated wall time |
|---:|---:|
| 1 | 1.37 s |
| 100 | 2.29 min |
| 1,000 | 22.9 min |
| 21,025 | 8.02 h |

These are throughput extrapolations, not a formal landscape runtime. They omit
episode initialization, supine-model changes, trajectory control, state/reset
logic, data logging, force-map computation, optimization, and any failure
handling. The benchmark also steps past the walking task's terminal condition;
it therefore measures computation, not valid rehabilitation episodes.

## Direct answers

### Q1 — Installation and official test

Yes, after adding the single missing SciPy dependency inside the isolated
environment. The base PyPI install alone failed the full official suite because
of an undeclared upstream dependency. Final official test: PASS.

### Q2 — Stable headless MyoLeg

Yes for numerical/API execution: 1,000 random steps and 100,000 zero-control
benchmark steps had no exception or nonfinite value. Random actions did not
produce a stable walking episode and triggered early termination.

### Q3 — Joints / DoFs / actuators

29 joint objects, `nq=35`, `nv=34`, 14 equality constraints, and 80
muscle/tendon actuators.

### Q4 — Hip, knee, ankle, pelvis structure

Bilateral three-hinge hips, primary flexion knee hinges with equality-coupled
auxiliary/patellar joints, and explicit ankle/subtalar/MTP hinges descend from
a free root/pelvis structure. Exact addresses and ranges are in
`myoleg_joint_inventory.csv`.

### Q5 — Future supine single-leg sagittal rehabilitation environment

Structurally likely, but it requires model/specification changes to fix the
root/pelvis, non-target leg, ankle and non-sagittal hip DoFs. The 120-degree
native knee limit versus formal 145 degrees must be resolved rather than
silently clipped.

### Q6 — Virtual shank strap and external force

Yes in principle. The tibia bodies have many actual sites, and MuJoCo exposes
body external wrench/application APIs. A dedicated strap site/location and
force direction still require explicit later validation and model modification.

### Q7 — PINN-relevant outputs

The APIs expose q, dq, muscle actuator forces, passive/actuator/constraint/
applied generalized forces, and body external wrenches. This is data-access
feasibility only; no PINN was trained or authorized.

### Q8 — 24 s and 21,025-candidate cost

Approximately 1.37 s per 24 s trajectory and 8.02 h for 21,025 trajectories in
the simplest serial throughput extrapolation. A real derived environment and
logging/control pipeline may be materially slower.

### Q9 — Next stage

Yes, proceed to a separately reviewed
`MYOLEG_SUPINE_HIP_KNEE_REHAB_FEASIBILITY_V1`, specifically to resolve model
locking, coordinate conventions, shank attachment definition, native ROM
compatibility, passive-state semantics, and visualization environment choice.
This report does not implement that stage.

## Git tracking recommendation

`REQUIRED_TO_TRACK`:

- this report
- `environment_versions.json`, `pip_freeze.txt`
- official/headless/visualization raw outputs
- registered environment list
- joint, actuator, body/site/tendon, and equality inventories
- model summary, runtime benchmark, checksums, and reproducible audit script

`OPTIONAL_TO_TRACK`:

- none currently generated

`DO_NOT_TRACK`:

- `/Users/fengjiaxin/.virtualenvs/myosuite-v2/`
- site-packages, uv/pip caches, downloaded Python runtime, GUI state
- any generated full landscape or learned policy (none was created)

No files were committed automatically.
