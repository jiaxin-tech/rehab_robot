# RT IPC freshness analysis

All timestamps below are host monotonic timestamps. The SDK exposes no controller timestamp for these RT fields, and none was synthesized.

| Case | Source P99 / max ms | IPC age P95 / P99 / max ms | IPC >50 ms | Supervisor P99 / max ms |
|---|---:|---:|---:|---:|
| RT only | 8.304 / 59.081 | 7.823 / 8.046 / 9.761 | 0 | 10.984 / 58.471 |
| RT + 20 Hz wrench | 8.289 / 61.290 | 7.863 / 8.063 / 18.864 | 0 | 10.979 / 63.371 |

RT-only source freezes >20 ms / max: 3 / 59.081 ms; publish/receive/overwrite/drop: 22684 / 17841 / 4843 / 0.
Concurrent source freezes >20 ms / max: 9 / 61.290 ms; publish/receive/overwrite/drop: 22685 / 17820 / 4865 / 0.
Wrench sequence advance / age P99 / max / heartbeat max: 3583 / 50.488 / 352.186 / 342.014 ms; natural block: false; errors: [].

RT_PROCESS_SINGLE_THREAD = true
RT_ONLY_SOURCE_RELIABLE = true
RT_ONLY_IPC_FRESH = true
RT_CONCURRENT_SOURCE_RELIABLE = true
RT_CONCURRENT_IPC_FRESH = true
WRENCH_CONCURRENCY_DEGRADES_RT_IPC = false
ROOT_CAUSE_LAYER = rt_process
PROCESS_ARCHITECTURE_STATUS = PASS
READY_FOR_FIRST_MOTION_TEST = false

Recommended next step only: preserve this diagnostic architecture and review the captured tails before designing any separately authorized first-motion test; no motion test was executed here.
