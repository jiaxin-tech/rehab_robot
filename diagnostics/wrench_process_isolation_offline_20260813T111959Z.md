# Wrench process isolation offline validation

- Timestamp: `2026-08-13T11:20:18.029232+00:00`
- Overall pass: `true`
- Architecture: parent/supervisor and spawned child process; bounded latest-snapshot IPC; no SDK object crosses IPC.
- Thresholds are diagnostic-only: wrench stale 150 ms; per-case worker hung thresholds are recorded in JSON.

| Scenario | Pass | Main Hz | P99 period ms | Stale | Hung | Error | Cleanup |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | true | 100.000 | 10.106 | true | false |  | true |
| slow_40ms | true | 100.000 | 10.895 | true | false |  | true |
| stale_500ms | true | 100.000 | 10.182 | true | false |  | true |
| long_block_10s | true | 99.771 | 10.156 | true | true |  | true |
| permanent_block | true | 100.000 | 10.067 | true | true |  | true |
| error_263 | true | 100.000 | 10.067 | true | false | 263 | true |
| worker_exception | true | 97.168 | 17.574 | true | false |  | true |
| worker_crash | true | 100.000 | 10.038 | true | false |  | true |
| ipc_saturation | true | 98.484 | 15.578 | true | false |  | true |

Forced termination is intentionally reported separately from graceful SDK disconnect; offline mock cleanup does not make any claim about a native SDK session.
