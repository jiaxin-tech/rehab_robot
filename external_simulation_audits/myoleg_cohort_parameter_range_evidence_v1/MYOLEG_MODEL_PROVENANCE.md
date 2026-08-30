# MyoLeg model provenance audit

## Provenance chain

MyoSuite's official documentation describes MyoLeg as a 10-joint, 20-DoF,
80-muscle-tendon-unit model that takes the Rajagopal full-body gait model as a
**close reference**.  That wording does not establish a one-to-one converter or
parameter identity.  Rajagopal et al. (2016) built an OpenSim generic healthy
young gait model with lower-limb architecture drawn from cadaver measurements
and MRI muscle-volume data.  The current repository's native supine model is a
derived MyoLeg MJCF that preserves all 80 muscles/tendons and the target knee
equalities while changing root pose, locking non-target coordinates and
disabling world contacts; its frozen SHA is
`20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d`.

## What can and cannot be called inherited

- Body names, meshes, tendon transmissions, muscle names and numerical MJCF
  fields come from the installed MyoLeg asset and are preserved in the frozen
  derived model unless its manifest explicitly lists a change.
- Rajagopal provides source-model context for geometry, muscle architecture and
  a generic 75 kg / 170 cm male skeleton.  The available official documentation
  does not prove that every MyoLeg body inertia or muscle curve field is an
  unchanged Rajagopal value.
- Therefore this audit calls the parameters `MyoLeg model parameters`, not
  direct measurements from Rajagopal's participants.

## OpenSim-to-MuJoCo muscle semantics

Rajagopal uses Millard-type Hill muscle-tendon units with an explicit source
model parameterization.  MuJoCo instead treats the spatial transmission as a
tendon and the muscle as an abstract force generator, assumes an inelastic
biological tendon for its shortcut mapping, infers `L0` and `LT` from length
ranges, and evaluates:

`FLV = FL(L) * FV(V) * activation + FP(L)`

`actuator_force = -F0 * FLV`

`fpmax` is the normalized passive force at `lmax`, relative to `F0`.  It is an
actuator-curve parameter.  It is **not** a directly measured human passive
stiffness, and the OpenSim passive-curve calibration evidence changes curve
strain/length parameters rather than providing a population distribution for
MuJoCo `fpmax`.

## Consequence for this stage

Anthropometric mass ranges can be anchored to model-derived adult segment
statistics.  The coupled inertia edit remains
`INERTIA_SCALING_IS_MODELING_APPROXIMATION`.  All three `fpmax` intervals remain
synthetic sensitivity ranges, despite literature support that passive and
biarticular mechanics vary between people.
