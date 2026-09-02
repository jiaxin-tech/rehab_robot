# Future Static Geometry Validation Plan

This plan is not executed and does not authorize robot or human exposure.

## Required points and frames

Use one independently verified robot-base coordinate system and record: base/world registration; flange origin; configured controller TCP; robot-side strap eye/attachment center; limb-side attachment center on a rigid rehabilitation fixture/phantom; bed-plane origin/x/z axes; and, only for testing candidates C/E, model hip and equivalent L2 point. Do not substitute TCP for the physical eyelet or L2 for an ankle/attachment without measured evidence.

## Static pose and setup design

After separate site approval, arrange representative stationary poses spanning the intended flexion/extension geometry without human loading. At each pose, measure both strap attachment points with an independently calibrated metrology method, record tool/TCP/base/bed configuration, verify strap routing and tautness, and repeat complete removal/reinstallation to quantify placement repeatability. No rehabilitation motion is part of this protocol.

Compute the line unit vector, its pose-dependent angular change, point-location repeatability and disagreement with TCP tangent, fixed bed axis, endpoint-to-hip and model-equivalent directions. Propagate point uncertainty into task-direction angular uncertainty and projected-force uncertainty.

No numeric tolerance is invented here. Freeze tolerances prospectively from metrology accuracy and acceptable propagated endpoint uncertainty before viewing trajectory outcomes. Failure, slack, multi-contact routing or inconsistent attachment geometry leaves `TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION`.
