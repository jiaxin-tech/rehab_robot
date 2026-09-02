# Geometry Measurement Readiness

## Decision

The protocol definition is complete enough to specify what must be measured, but the physical setup is not execution-ready. There is no repository evidence of the production cuff/eyelet identities, rigid surrogate, jig, fiducials, calibrated metrology or `T_B_R` result.

## Minimum viable geometry measurement method

Use a rigid `NON_HUMAN_SHANK_SURROGATE`, the actual identified cuff/strap and robot eyelet, a repeatable jig, and a rigid fiducial frame. The minimum viable metrology is either:

- a calibrated 3-D digitizer/tracked pointer observing the eyelet, at least two taut free-span fiducials, cuff landmarks and at least three non-collinear setup/base reference points; or
- calibrated multi-view camera/photogrammetry with a validated scale/frame target and equivalent point/line uncertainty.

Advanced motion capture is not mandatory. A ruler/caliper may supplement local offsets only when its calibrated uncertainty is demonstrably adequate; by itself it normally does not establish a common 3-D base/setup transform.

Required products are: labelled raw point observations; eyelet offset `p_attach_TCP`; fitted strap exit/free-span line and residual; setup/surrogate/cuff placement IDs; `T_B_R` transform, convention, fiducials, residual and covariance; ten remove/reattach setup records with three within-setup repeats; angular uncertainty output. Robot probing is not required and remains unauthorized.

The surrogate must be rigid, dimensionally repeatable, stable in the jig, compatible with the real cuff and permit repeatable placement landmarks. It is a mechanical surrogate, not a physiological limb model.
