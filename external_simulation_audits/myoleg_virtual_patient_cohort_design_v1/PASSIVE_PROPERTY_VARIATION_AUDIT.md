# Passive muscle-property audit

## Actual fields

- `fpmax` = `gainprm[7]` / `biasprm[7]`: passive force at `lmax`, relative to peak rest force `F0`. The compiled model range is 0.806346–2.
- `range` = `gainprm[0:2]`: scaled muscle operating range used with the transmission range to infer `L0` and `LT`.
- `actuator_lengthrange`: physical range of the tendon transmission in metres. It is not physiological normalized muscle-fiber length.
- `lmin/lmax/vmax/fvmax`: remaining FLV shape parameters.

`fpmax` is the only V1 primary passive-property candidate. It changes passive force magnitude, does not enter the active `FL*FV` term, retains native operating geometry, and nevertheless requires external range evidence. The smoke test changes the corresponding gain and bias slots together so a future XML `fpmax` edit remains representable.

At frozen P0, `F0` and `fpmax` multiply the same passive term. The smoke results confirm exact same-group equivalence. They must therefore be mutually exclusive factors in a P0 cohort; separating force capacity from passive magnitude requires an independently designed nonzero-activation condition.

`range` and the other FLV-shape fields are Class D until muscle-specific evidence exists. They can strongly move operating lengths and may cause force growth near the edge. `actuator_lengthrange` is Class E because it follows geometry rather than constituting a free physiological fiber-length parameter.

Official semantics: https://mujoco.readthedocs.io/en/3.6.0/modeling.html#muscle-actuators and https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html#actuator-muscle
