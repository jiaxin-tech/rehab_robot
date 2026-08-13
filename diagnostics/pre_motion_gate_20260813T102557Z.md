# Pre-motion gate — 2026-08-13

This gate covers strict read-only diagnostics only. No power, mode, recovery,
reset, collision-configuration, register-write, or motion API was called.

## Status

```text
P1 Safety state: BLOCKED
P2 Wrench reliability: NOT YET RELIABLE
P3 Non-blocking acquisition: BLOCKED

READY_FOR_FIRST_MOTION_TEST = false
```

## P1 — Safety state

`operationState`, `powerState`, `operateMode`, `robotInfo`, and
`queryControllerLog` were confirmed callable read-only getters/queries on SDK
0.7.0 / controller 3.2.1 / XMC12-R1300-W7G3B4.

The documented collision query signature is:

```python
queryEventInfo(eventType: Event, ec: dict) -> dict
```

It was called exactly with `Event.safety` (Python type `Event`, numeric value
1) and a Python `dict`. It returned SDK error 259: parameter type/count error.
The documented result key is `EventInfoKey.Safety.Collided == "collided"`.
No guessed enum/string/integer alternative and no event watcher were tried.

`powerState` is usable for power/E-stop/safety-door observation and
`operationState` is usable for IDLE observation, but no current verified
collision/protective-stop source is available. Historical controller logs are
not a synchronized current safety latch.

Evidence: `safety_state_audit_20260813T101744Z.json` and `.md`.

## Environment mismatch and stop decision

The approved task background described an unpowered robot. The audit observed:

```text
operationState = idle
powerState     = on
operateMode    = automatic
```

The diagnostic did not create this state and did not change it. Per the task's
explicit discipline, all further real-robot tests stopped immediately. No
automatic down-power, mode change, fault clear, reset, or recovery was
attempted.

## P2 — Wrench reliability

The requested 15-minute 20 Hz and 15-minute 50 Hz tests were **not executed**
after the environment mismatch. No `wrench_longrun_20hz_*` or
`wrench_longrun_50hz_*` result was fabricated.

The earlier 10-second evidence remains applicable only as prior evidence:

- normal query latency was usually about 1.2 ms;
- one 36.90 ms deadline miss occurred;
- one 10.006 s block ended with SDK error 263;
- reliable 50 Hz deadline support was not established.

The new `scripts/characterize_wrench_longrun.py` tool is ready for a later
approved unpowered/IDLE session. It provides per-query RT association, required
latency buckets and percentiles, deadline ratios, error-263 recovery tracking,
cache-age/stale observation, a native-call watchdog, incremental fsynced CSV,
and a parent-process hard watchdog. Detecting/terminating a timed-out child is
not claimed to cancel the underlying SDK call.

## P3 — Non-blocking acquisition

Pure-software fault injection passed for 1 ms, 40 ms, 500 ms, a 10 s block,
and one synthetic SDK error 263:

- the 100 Hz main loop continued for 1000 ticks during the 10 s block;
- the independent 125 Hz state producer advanced 1250 updates;
- wrench age increased and stale state was detected;
- the error remained observable in cache history;
- the next synthetic query recovered;
- worker and state threads joined cleanly.

This proves the diagnostic thread/cache design under Python fault injection.
It does not yet prove the native xCoreSDK call releases the GIL during every
failure mode, because the required real-robot concurrent validation was stopped.
Therefore P3 is BLOCKED overall despite the offline subtest passing.

Evidence: `wrench_nonblocking_validation_20260813T102459Z.json`.

Thread isolation is sufficient for a responsive main loop when the blocking
operation releases the GIL. A thread cannot safely kill a permanently blocked
native call. Process isolation is appropriate only if forced restart becomes a
hard requirement, and only after SDK connection ownership and concurrent
controller-session behavior are validated. SDK objects must be created inside,
not passed across, a worker process.

## ROBOT_LOCAL_IP recommendation

Keep `192.168.50.209` machine-local. Do not hard-code it in public source.
Prefer a required diagnostic/runtime CLI argument such as `--local-ip`, with an
environment variable or ignored local config as a convenience fallback. Print
Robot IP, Local RT IP, SDK version, controller version, and robot model at
startup. Fail fast when Local RT IP is empty; do not guess a network adapter.

## Remaining blockers

### P0

1. A trained operator must reconcile the unexpected powered/automatic state
   with the approved test setup. Software must not change it automatically.
2. A reliable current collision/protective-stop observation is still missing;
   `queryEventInfo(Event.safety, ec)` returns SDK error 259.

### P1

1. Run the supervised 15-minute 20 Hz wrench characterization after the
   approved unpowered/IDLE state is restored.
2. Run the supervised 15-minute 50 Hz characterization; require long-tail and
   error-263 evidence, not average latency alone.
3. Run the real-robot IDLE concurrent RT/wrench/cache validation to establish
   native binding/GIL behavior and confirmed disconnect semantics.

