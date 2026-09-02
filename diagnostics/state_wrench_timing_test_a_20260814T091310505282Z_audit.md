# Formal W1 state–wrench timing audit: test_a

Strictly read-only. No Servo/Move/Jog/trajectory/mode/power/reset/clear command was issued.

- Result integrity: `PASS`
- Requested/observed duration: `900.0` / `900.008` s
- Git: `syn/win-0813` @ `4284da5cf23a6d97f22fd3987e3534a49814fc2d` (dirty=True)
- Robot/controller/SDK: `XMC12-R1300-W7G3B4` / `3.2.1` / `0.7.0`

## Timing layers

| Layer | mean | median | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|
| RT source interval ms | 8.000 | 7.999 | 8.180 | 8.304 | 59.983 |
| RT IPC age ms | 4.169 | 4.163 | 7.774 | 8.044 | 36.441 |
| RT current snapshot age ms | 4.340 | 4.280 | 7.920 | 8.207 | 57.287 |
| Supervisor loop ms | 10.035 | 9.976 | 10.796 | 10.979 | 56.721 |
| Wrench inter-arrival ms | n/a | n/a | n/a | n/a | n/a |

RT IPC delivery-age counts >20/50/100/1000 ms: `{'gt_20_ms': 2, 'gt_50_ms': 0, 'gt_100_ms': 0, 'gt_1000_ms': 0}`.
RT current-snapshot-age counts >20/50/100/1000 ms: `{'gt_20_ms': 102, 'gt_50_ms': 15, 'gt_100_ms': 0, 'gt_1000_ms': 0}`.
Wrench requests/success/failure: `0/0/0`; errors: `{}`.
OperationState before/after: `idle` / `idle`; non-idle transitions: `0`.

## Formal gates

- `data_integrity` = `PASS` — lossless audit telemetry and requested duration
- `operation_state_stability` = `PASS` — operationState must remain idle
- `process_stability` = `PASS` — no worker crash or hung event
- `cleanup` = `PASS` — graceful worker exit and SDK disconnect
- `rt_source_timing` = `UNDEFINED` — no formal RT source interval threshold exists
- `rt_ipc_freshness` = `UNDEFINED` — max_state_age_s is unset
- `supervisor_timing` = `UNDEFINED` — max_command_lateness_s is unset
- `wrench_freshness` = `UNDEFINED` — wrench is intentionally off in Test A
- `wrench_error_reliability` = `UNDEFINED` — wrench is intentionally off in Test A

## Artifacts

- `supervisor_csv`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_a_20260814T091310505282Z_supervisor.csv`
- `rt_source_csv`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_a_20260814T091310505282Z_rt_source.csv`
- `wrench_csv`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_a_20260814T091310505282Z_wrench.csv`
- `events_json`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_a_20260814T091310505282Z_events.json`
- `summary_json`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_a_20260814T091310505282Z_summary.json`
- `audit_markdown`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_a_20260814T091310505282Z_audit.md`
- `timing_plot`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_test_a_20260814T091310505282Z_timing.png`

`SAFE_TO_PROCEED_MANUAL_PUSH = false` (W1 artifacts require review before W2).

`READY_FOR_FIRST_MOTION_TEST = false`
