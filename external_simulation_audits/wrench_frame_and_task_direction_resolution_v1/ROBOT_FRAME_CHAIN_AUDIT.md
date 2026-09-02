# Robot Frame Chain Audit

## Current chain

```text
world --baseFrame()--> robot base --RT tcpPoseAbc_m / FK--> flange or configured TCP
flange --toolset.end--> tool/TCP
tool/TCP --UNKNOWN physical attachment transform--> robot-side strap attachment
limb equivalent pull point / physical cuff --UNKNOWN setup measurement--> limb-side strap attachment
rehab bed frame --rehab_frame_config--> robot base
```

| Transform/entity | Source | Evidence class | Current status |
|---|---|---|---|
| `^world T_base` | future read of xCoreSDK `baseFrame()` | CONFIGURED_GEOMETRY | API and units documented; current value not captured; physical convention unvalidated |
| `R_base_from_world` | transpose of `R_world_from_base` built from SDK XYZ Euler | CONFIGURED_GEOMETRY + math | internal math verified; physical orientation pending |
| base-to-flange/TCP pose | RT `tcpPoseAbc_m`, project treats it as base TCP | CONFIGURED_GEOMETRY | source path exists; active tool semantics require runtime validation |
| flange-to-tool/TCP | future read `toolset.end.trans/rpy` | CONFIGURED_GEOMETRY | query path exists, no frozen value |
| active HMI tool/workobject | controller/HMI state | MEASURED/CONFIGURED_GEOMETRY | explicitly unverified; available-name list is not active selection proof |
| TCP-to-robot strap eye/attachment | no repository measurement | ASSUMED_GEOMETRY | unknown |
| limb/cuff strap load-transfer point | no repository measurement | ASSUMED_GEOMETRY | unknown |
| rehab bed x/z axes in base | `config/rehab_frame_config.json` | CONFIGURED_GEOMETRY | both null, reviewed=false |
| start TCP anchor | per-session StartAnchor schema | MEASURED_GEOMETRY candidate | no active captured anchor file in repository |
| `L1=0.42 m` hip-to-knee | formal lower-limb configuration | CONFIGURED_GEOMETRY | model geometry, not a current patient measurement |
| `L2=0.30 m` knee-to-equivalent strap pull point | formal lower-limb configuration | CONFIGURED_GEOMETRY | equivalent traction point; not observed ankle and not proven physical strap attachment |
| hip origin in robot base | absent in start-anchored mode | ASSUMED/UNAVAILABLE | not measured; required for endpoint-to-hip task direction |

No active tool name, TCP offset, flange-to-tool transform, world/base pose, bed axes or physical strap attachment transform is frozen in the current unreviewed configs. This audit does not modify them.
