# Strap/Pull Geometry Audit

| Geometry item | Repository meaning | Evidence class | Resolution |
|---|---|---|---|
| robot TCP | controller-configured endpoint represented by RT pose | CONFIGURED_GEOMETRY | active tool/TCP transform not frozen |
| robot-side strap attachment | physical eyelet/cuff connection on end effector | ASSUMED_GEOMETRY | no offset from TCP is measured |
| limb-side strap attachment | physical center/line of load transfer at the cuff | ASSUMED_GEOMETRY | no base-frame measurement or placement repeatability evidence |
| equivalent pull point | 2-DOF point at L2=0.30 m from knee | CONFIGURED_GEOMETRY | formal model point only; not an ankle and not automatically actual attachment |
| shank orientation | `theta_shank=q_hip-q_knee` | FROZEN_MODEL_SEMANTICS | preserved |
| bed plane | rehab x/z axes expressed in robot base | CONFIGURED_GEOMETRY | config values null and unreviewed |
| hip coordinate | origin of 2-DOF model | ASSUMED/UNAVAILABLE in robot base | start-anchored trajectory intentionally does not require it |

The path command applies the model equivalent-pull-point displacement at the captured TCP start anchor. This establishes command geometry, not the instantaneous physical strap line. Strap routing, slack/tension, attachment widths and cuff load distribution can make the real line differ from TCP tangent or endpoint-to-hip direction.

`MEASURED_GEOMETRY` currently contains only schema-capable future observations, not an approved setup record. The actual line of action remains unknown until both physical attachment points are measured in one validated frame over representative static poses.
