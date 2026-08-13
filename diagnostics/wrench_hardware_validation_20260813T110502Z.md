# ROKAE wrench hardware validation

- Test date: 2026-08-13
- Robot IP / local RT IP: `192.168.50.103` / `192.168.50.209`
- SDK / controller: `0.7.0` / `3.2.1`
- Robot: `XMC12-R1300-W7G3B4`
- Preflight state: `power=on`, `operation=idle`, `operateMode=automatic`
- Scope: strictly read-only. No motion, power, mode, reset, clear-fault, collision, safety, speed, joint-command, or TCP-command API was called.

## A. 20 Hz

The requested run was 900 s and 18,000 scheduled calls. Native blocking made the acquisition phase last 961.850 s (963.629 s including connection/preflight/cleanup), and only 8,274 calls actually started.

| Metric | Result |
|---|---:|
| Actual calls | 8,274 |
| Success / errors | 8,225 / 49 |
| Mean | 60.676 ms |
| Median / P50 | 1.211 / 1.211 ms |
| P90 | 1.720 ms |
| P95 | 1.912 ms |
| P99 | 9.392 ms |
| P99.9 | 10,013.344 ms |
| Max | 10,024.883 ms |
| >20 / >50 / >100 ms | 71 / 54 / 49 |
| >1 s | 49 |
| SDK error 263 | 49 |
| 50 ms deadline misses | 54 (0.6526%) |

Latency buckets: `<=5 ms: 8157`, `5–10 ms: 41`, `10–20 ms: 5`, `20–50 ms: 17`, `50–100 ms: 5`, `100–500 ms: 0`, `500–1000 ms: 0`, `>1 s: 49`.

The first 411.267 s comprised 8,225 consecutive successful calls: mean 1.406 ms, P95 1.881 ms, P99 2.831 ms, max 62.033 ms. It then entered a persistent failure phase: calls 8,226–8,274 were all error 263 and lasted about 10 s each. The 49-error span was 490.556 s; there was no successful next attempt and no SDK reconnect was attempted.

Decision: `WRENCH_20HZ_RELIABLE = false`.

## B. 50 Hz

Not executed. The 20 Hz prerequisite failed because the SDK postcheck and normal disconnect both returned error 263, and the connection could not be confirmed disconnected. No 50 Hz output file was fabricated.

Decision: `WRENCH_50HZ_RELIABLE = not_enough_evidence`.

## C. RT during wrench test

Before the first native long block, RT source period was mean 8.050 ms, P99 9.844 ms, max 56.429 ms; state age was mean 8.131 ms, P99 10.376 ms, max 60.175 ms. This is consistent with the earlier standalone RT result around 125 Hz, apart from a small number of host-side scheduling gaps.

Across the whole concurrent test, RT period was mean 27.284 ms, P99 9.934 ms, max 80,082.130 ms. State age was mean 114.358 ms, P99 10.767 ms, max 120,119.768 ms. The apparently normal P99 does not offset the sparse catastrophic tail.

During the 490.556 s error phase, only 105 main-loop ticks were recorded and RT sequence advanced by only 16. Failure-phase RT period was mean 27,889.465 ms, P99 77,479.605 ms, max 80,082.130 ms; RT age reached 120,119.768 ms. Sixteen frozen-state events were detected over the full run, with a maximum unchanged sequence interval of 120,115.135 ms. The `consumer_skipped_rt_sequence_count` is not treated as network packet loss: the 100 Hz observer is slower than the nominal 125 Hz producer and also stopped sampling during GIL/native stalls, so it cannot establish source-side drop count.

Standalone RT reads were reliable, but RT is not reliable in this shared-process wrench architecture.

Decision: `RT_STATE_RELIABLE = false` for concurrent wrench use.

## D. Real concurrency

The diagnostic main loop never called `getEndTorque()` directly. A wrench thread owned that call, RT used its producer/cache path, and the main loop read immutable snapshots.

Before the first long block, the requested 100 Hz loop achieved 80.608 Hz: period mean 12.406 ms, P95 31.923 ms, P99 39.925 ms, max 80.032 ms. This shortfall is host/Python scheduling behavior and is separate from wrench latency.

For the whole run, main-loop rate was 37.712 Hz: period mean 26.517 ms, P95 31.935 ms, P99 39.946 ms, max 10,023.468 ms. During the 263 phase it collapsed to 0.223 Hz: period mean 4,481.423 ms, P95 10,016.953 ms, P99 10,022.641 ms, max 10,023.468 ms.

There were 49 real native long-block events. In 46 of 49 event windows, RT sequence advance was zero or unobservable; three had no main-loop tick inside the call interval. Other windows generally had only 2–3 ticks separated by approximately 10 s, not a continuing 100 Hz loop. Thus the observed `getEndTorque()` native block prevented timely execution of both the Python main loop and RT producer thread in this process.

Wrench age reached 470,596.522 ms. All 105 failure-phase ticks reported stale/invalid, and the worker thread remained nominally alive, but same-process stale detection was not timely because Python execution itself was frozen. A process-external consumer is required for a meaningful age watchdog.

Cleanup results:

- Diagnostic worker eventually returned and the Python processes exited.
- Post-run `operationState` getter returned SDK 263, so the final robot operation state could not be re-confirmed.
- `disconnectFromRobot` was attempted and returned SDK 263; normal SDK disconnect is not confirmed.
- A privileged host check found 0 residual Python processes, 0 Python-owned UDP endpoints on `192.168.50.209`, and 0 Python-owned TCP connections to `192.168.50.103`.
- No reset, clear-fault, power, mode, or other recovery action was used.

Provisional age guidance for a future process-isolated 20 Hz sensor path: `recommended_warning_age_ms = 75` and `recommended_stale_age_ms = 150`. The basis is a 50 ms expected period, pre-block query-start P99 64.604 ms, and pre-block sampled wrench-age P99 55.085 ms. These are report-only recommendations and were not written into safety code. No 50 Hz recommendation is made because 50 Hz was not run. Thresholds alone cannot repair same-process GIL/native blocking.

## E. Architecture decision

`THREAD ISOLATION NOT SUFFICIENT`.

A process-isolated wrench worker is required before wrench data is integrated into acquisition or control-side runtime. The wrench process should exclusively own its SDK call and publish timestamped immutable snapshots over bounded IPC; the main process must enforce age/valid/stale without waiting for the producer, and a supervisor must be able to terminate/restart only that process. This recommendation is specific to the observed `getEndTorque()` behavior and is not generalized to every xCoreSDK native call. No production hardware-layer refactor was performed in this test.

`WRENCH_CAN_BE_USED_AS_SYNCHRONOUS_CONTROL_INPUT = false`.

`WRENCH_CAN_BE_USED_AS_ASYNC_TIMESTAMPED_SENSOR = true`, but only behind process isolation, external age validation, stale fail-closed behavior, and without making it motion-authoritative.

## F. Remaining blockers before motion

- P1 safety observability remains blocked: `queryEventInfo(Event.safety, ec)` previously returned SDK 259. No parameter guessing was retried.
- The current shared-process wrench-thread architecture demonstrably freezes the main loop and RT producer during native blocking.
- 20 Hz long-run reliability failed; 50 Hz has no evidence because its safety/cleanup prerequisite failed.
- Normal SDK disconnect and final operation-state read were not confirmed. An operator must verify the controller state before any later test; this run did not attempt reconnection or state recovery.
- A process-isolated implementation and read-only validation are required before wrench can be treated as an asynchronous runtime sensor.

`READY_FOR_FIRST_MOTION_TEST = false`.

Final decisions:

```text
RT_STATE_RELIABLE = false
WRENCH_20HZ_RELIABLE = false
WRENCH_50HZ_RELIABLE = not_enough_evidence
REAL_THREAD_ISOLATION_VALIDATED = false
NATIVE_LONG_BLOCK_OBSERVED = true
WRENCH_CAN_BE_USED_AS_SYNCHRONOUS_CONTROL_INPUT = false
WRENCH_CAN_BE_USED_AS_ASYNC_TIMESTAMPED_SENSOR = true
```
