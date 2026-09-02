# Sampling and Timestamp Plan

Master clock: `HOST_MONOTONIC_PERF_COUNTER_NS`.

For every PRE/LOAD/POST window record query start, query end, midpoint, publish time, query duration, sequence ID, raw requested frame, raw Fx/Fy/Fz, joint torque arrays for provenance, validity and invalid reason. Retain robot state host receive time, TCP/tool pose, operation state and state-wrench skew as context; no device/source timestamp exists.

Each window targets `100` valid host queries. Each pose x direction x approved-level cell has exactly `5` independent load applications. This supplies repeated means/SDs without allowing post-result additions. The nominal configured query target is 50 Hz, but observed query rate/source-update behavior must be reported rather than assumed; repeated identical values cannot prove unique controller updates.

Record latency distributions and missed/invalid queries. Future maximum dwell, query failure and steady-force acceptance rules require safety/calibration review and remain null. Host timing is sufficient for steady-state window averages only:

`STATIC_FRAME_VALIDATION != DYNAMIC_SYNCHRONIZATION_VALIDATION`

No result may claim transport delay, controller source latency or dynamic command-state-wrench alignment from this protocol.
