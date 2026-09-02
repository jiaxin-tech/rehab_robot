# Data Acquisition Readiness

## Existing reusable primitives

| Required datum | Current code evidence | Readiness |
|---|---|---|
| host monotonic timestamp | `perf_counter_ns` and state timestamps exist | DEFINED_BUT_NOT_REVIEWED for this physical protocol |
| wrench query start/end/midpoint | adapter records start/end and midpoint | DEFINED_BUT_NOT_REVIEWED |
| raw Fx/Fy/Fz and validity | adapter/state/collector fields exist | DEFINED_BUT_NOT_REVIEWED; physical frame/sign remains the validation target |
| robot TCP/joint/state | RT state and snapshot fields exist | DEFINED_BUT_NOT_REVIEWED |
| tool/TCP/config metadata | adapter/collector metadata exists | DEFINED_BUT_NOT_REVIEWED; active HMI tool/workobject remains unverified |
| pose/direction/repeat/load condition IDs | no static-validation field/state machine | MISSING |
| PRE/LOAD/POST label | protocol text only | MISSING |
| external calibrated load reading/uncertainty | no integrated formal record | MISSING |
| failure retention/checksums | generic episode logger primitives exist | DEFINED_BUT_NOT_REVIEWED for this protocol |

Existing primitives are sufficient to avoid redesigning collection. They are not an executed static-validation logger and must not be relabelled formal evidence.

## Minimum future implementation change

Add a standalone, default-off static-validation logger/runner around the existing read-only adapter and `EpisodeLogger`; do not change control behavior. Add only protocol cell metadata (`pose_id`, `direction_id`, `load_level_id`, `repeat_id`, `window_label`, calibrated-load value/uncertainty, fixture/calibration IDs), frozen run-manifest SHA, raw checksums and fail-closed invalid reasons. It must not enable, move, calibrate sensors, invoke SafetyGuard stop, or choose loads.

Before physical execution, pass fake-adapter/offline and supervised no-load dry runs proving headers, counts, PRE/LOAD/POST transitions, exception retention, flush/fsync and cleanup. Host timing supports static window averaging only and does not validate dynamic synchronization.
