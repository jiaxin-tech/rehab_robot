# wrench_process_multisession

- Result: `native_block_isolated`
- Requested overlap duration: `30.0` s
- RT/main session: parent PID `27348`
- Wrench session: child PID `9372`
- Disconnect order: `rt_first`
- Native block observed: `true`
- Main loop rate: `28.58754347556554` Hz
- Main loop P99/max: `75.78320199999997` / `79.9334` ms
- RT advance: `118`; RT P99/max: `8.106344` / `8.115166666666667` ms
- Wrench advance: `None`; max age: `None` ms
- Error codes: `[]`
- Child cleanup: `{"worker_terminated": true, "worker_exitcode": -15, "termination_latency_ms": 48.1059, "used_kill": false, "graceful_disconnect_confirmed": false, "forced": true}`
- Parent disconnect confirmed: `true`

Forced child termination, if present, is not a graceful SDK disconnect.
