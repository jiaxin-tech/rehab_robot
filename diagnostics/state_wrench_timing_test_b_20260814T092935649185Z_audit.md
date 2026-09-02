# Formal W1 state–wrench timing audit: test_b

Strictly read-only. No Servo/Move/Jog/trajectory/mode/power/reset/clear command was issued.

- Result integrity: `FAIL`
- Requested/observed duration: `900.0` / `169.610` s
- Git: `syn/win-0813` @ `4284da5cf23a6d97f22fd3987e3534a49814fc2d` (dirty=True)
- Robot/controller/SDK: `XMC12-R1300-W7G3B4` / `3.2.1` / `0.7.0`

## Timing layers

| Layer | mean | median | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|
| RT source interval ms | 8.000 | 7.999 | 8.170 | 8.266 | 58.963 |
| RT IPC age ms | 4.227 | 4.135 | 7.832 | 8.032 | 23.198 |
| RT current snapshot age ms | 5.993 | 4.266 | 7.974 | 10.100 | 724.138 |
| Supervisor loop ms | 10.028 | 9.976 | 10.764 | 10.977 | 54.297 |
| Wrench inter-arrival ms | 57.371 | 49.875 | 50.941 | 51.673 | 10001.307 |

RT IPC delivery-age counts >20/50/100/1000 ms: `{'gt_20_ms': 1, 'gt_50_ms': 0, 'gt_100_ms': 0, 'gt_1000_ms': 0}`.
RT current-snapshot-age counts >20/50/100/1000 ms: `{'gt_20_ms': 112, 'gt_50_ms': 75, 'gt_100_ms': 63, 'gt_1000_ms': 0}`.
Wrench requests/success/failure: `2701/2698/3`; errors: `{'263': 3}`.
OperationState before/after: `idle` / `None`; non-idle transitions: `0`.

## Formal gates

- `data_integrity` = `FAIL` — lossless audit telemetry and requested duration
- `operation_state_stability` = `FAIL` — operationState must remain idle
- `process_stability` = `FAIL` — no worker crash or hung event
- `cleanup` = `FAIL` — graceful worker exit and SDK disconnect
- `rt_source_timing` = `UNDEFINED` — no formal RT source interval threshold exists
- `rt_ipc_freshness` = `UNDEFINED` — max_state_age_s is unset
- `supervisor_timing` = `UNDEFINED` — max_command_lateness_s is unset
- `wrench_freshness` = `UNDEFINED` — max_wrench_age_s is unset
- `wrench_error_reliability` = `UNDEFINED` — no formal wrench failure/error-263 acceptance threshold exists

## Artifacts

- `supervisor_csv`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_b_20260814T092935649185Z_supervisor.csv`
- `rt_source_csv`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_b_20260814T092935649185Z_rt_source.csv`
- `wrench_csv`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_b_20260814T092935649185Z_wrench.csv`
- `events_json`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_b_20260814T092935649185Z_events.json`
- `summary_json`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_b_20260814T092935649185Z_summary.json`
- `audit_markdown`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_b_20260814T092935649185Z_audit.md`
- `timing_plot`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_b_20260814T092935649185Z_timing.png`

`SAFE_TO_PROCEED_MANUAL_PUSH = false` (W1 artifacts require review before W2).

`READY_FOR_FIRST_MOTION_TEST = false`
