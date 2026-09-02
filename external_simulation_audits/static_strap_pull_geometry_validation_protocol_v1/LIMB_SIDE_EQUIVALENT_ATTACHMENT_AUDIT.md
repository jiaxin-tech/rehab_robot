# Limb-Side Equivalent Attachment Audit

| Candidate | Physical meaning | Observability/repeatability | Wide-cuff and line compatibility | Decision |
|---|---|---|---|---|
| cuff geometric center | geometric marker, not necessarily load transfer | easy to mark; placement-dependent | ignores distributed pressure and wrap | diagnostic only |
| strap exit/tangent point | boundary where contact becomes the taut free span | observable with fiducials/digitization; varies with pose, tension, wrap and placement | directly supports free-span direction if one line exists | primary operational line point |
| equivalent resultant-force application point | point/line giving the distributed contact resultant | mechanically meaningful, not directly observable from geometry alone | can represent net force but may omit a contact moment | future mechanics target |
| model L2 point | fixed 2-D knee-to-equivalent-point construction | exactly computable in model coordinates | not physical evidence | model diagnostic only |
| configuration-dependent contact/resultant | actual state-dependent distributed transfer | requires pressure/force/moment evidence | most realistic | future extension |

A wide cuff does not currently admit a defensible unique physical `p_limb_attach`. The proposed operational definition is `EQUIVALENT_LINE_POINT_AT_OBSERVED_STRAP_EXIT`: one point on the validated taut free span at its exit boundary, or equivalently a line fitted from at least two free-span fiducials. It defines direction but is not declared the true resultant application point.

Exit geometry must be measured across configuration and ten remove/reattach setups. It must not become a global constant if it varies materially. Slack, multiple spans, broad/ambiguous exit, unexpected routing contact or slip fails the gate.
