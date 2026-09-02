# Formal W1 Test A vs Test B comparison

No A-vs-B degradation PASS threshold exists in the formal manifest; comparison gate is `UNDEFINED`.

| Case | RT source P99/max ms | RT IPC P95/P99/max ms | Current age max ms | Supervisor P99/max ms | Wrench requests/failures |
|---|---:|---:|---:|---:|---:|
| Test A | 8.304 / 59.983 | 7.774 / 8.044 / 36.441 | 57.287 | 10.979 / 56.721 | 0 / 0 |
| Test B | 8.266 / 58.963 | 7.832 / 8.032 / 23.198 | 724.138 | 10.977 / 54.297 | 2701 / 3 |

- Test A completed: `True` (900.008/900.000 s)
- Test B completed: `False` (169.610/900.000 s); fatal error: `RuntimeError:RT worker hung`
- Test B error histogram: `{'263': 3}`
- Test B error 263 count: `3`
- Comparison gate: `UNDEFINED` (`NOT_FORMALLY_DEFINED`)
- SAFE_TO_PROCEED_MANUAL_PUSH = false (requires W1 review)
- READY_FOR_FIRST_MOTION_TEST = false

## Artifacts

- `comparison_json`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_comparison_20260814T093551709145Z.json`
- `comparison_markdown`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_comparison_20260814T093551709145Z.md`
- `comparison_plot`: `C:\Users\liumai\fjx\rehab_robot\diagnostics\state_wrench_timing_comparison_20260814T093551709145Z.png`
