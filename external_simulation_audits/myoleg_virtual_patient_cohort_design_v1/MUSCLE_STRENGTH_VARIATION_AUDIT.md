# Muscle strength / force-capacity audit

The frozen XML contains 80 `general` muscle actuators. Each has identical `gainprm[2]` and `biasprm[2]`, originating from the XML `force` field. In MuJoCo 3.6 this is peak active force `F0` in newtons. The observed compiled range is 245.514–5322.590 N.

The force law is `actuator_force = -F0 * (FL*FV*activation + FP)`. Consequently, scaling the actual XML `force` field changes both active capacity and the absolute passive force. An “active-only” runtime multiplier would no longer be the native XML field and is not adopted here.

Acceptable high-level factors must map to explicit actuator lists. The inventory derives hip-only, knee-only and hip+knee-spanning membership from the compiled tendon transmission moment matrix over all 401 V2 reference states, not from names. Global right-side scaling targets all 40 `_r` actuators. Group scaling remains provisional until the structural list is manually reviewed.

Official semantics: https://mujoco.readthedocs.io/en/3.6.0/modeling.html#muscle-actuators
