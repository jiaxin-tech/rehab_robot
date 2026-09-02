# Formal W1 state–wrench timing audit: offline_test_b

Strictly read-only. No Servo/Move/Jog/trajectory/mode/power/reset/clear command was issued.

- Result integrity: `PASS`
- Requested/observed duration: `3.0` / `3.008` s
- Git: `syn/win-0813` @ `4284da5cf23a6d97f22fd3987e3534a49814fc2d` (dirty=True)
- Robot/controller/SDK: `None` / `None` / `None`

## Timing layers

| Layer | mean | median | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|
| RT source interval ms | 8.054 | 7.976 | 8.974 | 9.021 | 28.991 |
| RT IPC age ms | 4.560 | 4.775 | 8.005 | 8.901 | 26.166 |
| Supervisor loop ms | 10.094 | 9.976 | 10.810 | 10.979 | 34.089 |
| Wrench inter-arrival ms | 50.012 | 49.867 | 50.869 | 50.940 | 50.989 |

RT IPC descriptive counts >20/50/100/1000 ms: `{'gt_20_ms': 1, 'gt_50_ms': 0, 'gt_100_ms': 0, 'gt_1000_ms': 0}`.
Wrench requests/success/failure: `43/43/0`; errors: `{}`.
OperationState before/after: `None` / `None`; non-idle transitions: `0`.

## Formal gates

- `data_integrity` = `PASS` — lossless audit telemetry and requested duration
- `operation_state_stability` = `FAIL` — operationState must remain idle
- `process_stability` = `PASS` — no worker crash or hung event
- `cleanup` = `PASS` — graceful worker exit and SDK disconnect
- `rt_source_timing` = `UNDEFINED` — no formal RT source interval threshold exists
- `rt_ipc_freshness` = `UNDEFINED` — max_state_age_s is unset
- `supervisor_timing` = `UNDEFINED` — max_command_lateness_s is unset
- `wrench_freshness` = `UNDEFINED` — max_wrench_age_s is unset
- `wrench_error_reliability` = `UNDEFINED` — no formal wrench failure/error-263 acceptance threshold exists

## Artifacts

- `supervisor_csv`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_offline_test_b_20260814T090011Z_supervisor.csv`
- `rt_source_csv`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_offline_test_b_20260814T090011Z_rt_source.csv`
- `wrench_csv`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_offline_test_b_20260814T090011Z_wrench.csv`
- `events_json`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_offline_test_b_20260814T090011Z_events.json`
- `summary_json`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_offline_test_b_20260814T090011Z_summary.json`
- `audit_markdown`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_offline_test_b_20260814T090011Z_audit.md`
- `timing_plot`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_offline_test_b_20260814T090011Z_timing.png`

`SAFE_TO_PROCEED_MANUAL_PUSH = false` (W1 artifacts require review before W2).

`READY_FOR_FIRST_MOTION_TEST = false`
