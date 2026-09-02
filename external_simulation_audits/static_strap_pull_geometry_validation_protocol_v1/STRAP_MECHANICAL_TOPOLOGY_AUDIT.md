# Strap Mechanical Topology Audit

## Actual intended chain and evidence class

The intended load path is: robot flange/configured tool and TCP -> physical eyelet, hook or fixture load-transfer center -> taut free strap segment -> exit/tangent region at a wide cuff -> distributed cuff/shank contact -> rigid shank surrogate for initial validation.

`PHYSICAL_HARDWARE_DEFINED` currently means only that those component roles belong to the intended apparatus. Repository evidence does **not** contain the actual eyelet offset, cuff dimensions, routing, tension, identifiers or installed coordinates. `MODEL_EQUIVALENT` comprises the planar L1/L2 kinematics, the 0.30 m knee-to-strap-equivalent point, and the start-anchored TCP displacement derived from it. It does not make the L2 point an ankle or an observed attachment.

The following remain `ASSUMED`, not measured: TCP origin equals the eyelet; the wide cuff has one fixed physical point; the bed frame is aligned with base/world; a static pull direction represents the full trajectory. No available diagram or metadata resolves these quantities.
