# ROKAE Option-B process isolation live validation

- Result: `native_block_isolation_failed`
- Supervisor / RT / wrench PIDs: `23572` / `17880` / `21816`
- SDK ownership: RT process exclusively owns Session A; wrench process exclusively owns Session B; supervisor owns no SDK object.
- Requested overlap duration: `180.0` s
- Main loop rate: `99.38392783639658` Hz
- Main loop P99/max: `11.021979999999996` / `48.9306` ms
- RT sequence advance: `8481`; source P99/max: `8.4082` / `15.979633333333332` ms
- Wrench sequence advance: `1317`; max age: `753.1299` ms
- Stop reason: `wrench_worker_hung`
- Wrench cleanup: `{"worker_terminated": true, "worker_exitcode": -15, "termination_latency_ms": 3.2989, "used_kill": false, "graceful_disconnect_confirmed": false, "forced": true}`
- RT cleanup: `{"worker_exited": true, "worker_exitcode": 0, "latency_ms": 109.0629, "graceful_disconnect_confirmed": true, "forced": false}`
- RT survived wrench stop: `false`

A forced process termination is not a graceful SDK disconnect.
