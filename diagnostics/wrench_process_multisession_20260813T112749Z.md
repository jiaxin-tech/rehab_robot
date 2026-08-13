# wrench_process_multisession

- Result: `pass`
- Requested overlap duration: `30.0` s
- RT/main session: parent PID `19760`
- Wrench session: child PID `7804`
- Disconnect order: `rt_first`
- Native block observed: `false`
- Main loop rate: `21.033679197354534` Hz
- Main loop P99/max: `136.10815200000002` / `344.4486` ms
- RT advance: `3733`; RT P99/max: `8.114012` / `11.308941666666668` ms
- Wrench advance: `572`; max age: `99.1348` ms
- Error codes: `[]`
- Child cleanup: `{"worker_exited": true, "worker_exitcode": 0, "latency_ms": 24.9967, "graceful_disconnect_confirmed": true, "forced": false}`
- Parent disconnect confirmed: `true`

Forced child termination, if present, is not a graceful SDK disconnect.
