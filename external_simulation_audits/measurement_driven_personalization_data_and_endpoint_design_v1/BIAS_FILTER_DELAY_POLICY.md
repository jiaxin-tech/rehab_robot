# Bias, Filter and Delay Policy

## Bias

Raw force is always retained. Three research candidates are compared by validation design, not trajectory outcome: raw; pre-episode zero-subtracted; and model-based compensated. Pre-episode zero subtraction is the leading simple candidate only when an independently verified unloaded condition exists. A leg already attached under strap preload is not a zero-force condition. Static offset, pose-dependent tool/load/controller bias, temperature/session drift, and run-order drift must be measured across multiple relevant postures. Model-based compensation remains secondary until it predicts independent held-out static checks.

Status: `BIAS_POLICY_REQUIRES_VALIDATION`. No numeric bias or drift threshold is invented.

## Filtering

`FILTER_NOT_YET_FROZEN`. Raw samples and timestamps remain immutable. A filter may be frozen only after source-update cadence, alias content, query latency, robot motion bandwidth, and expected human-interaction bandwidth are measured. Its record must include type, cutoff, order, causal/noncausal, phase/group delay, initialization, and online/offline compatibility. A zero-phase offline filter cannot silently become a causal online filter. Filtering may not be chosen because it makes a trajectory look better.

## Delay

`MEASUREMENT_DELAY_VALIDATION_REQUIREMENT` distinguishes:

1. host query duration (`query_finished - query_started`), which current code records;
2. source update/transport latency and source age, currently unknown without device time or a controlled event;
3. RT-state/wrench alignment skew on the host master clock;
4. command-to-observation response delay, which requires a controlled validation.

No fixed delay from old simulation may be inherited as hardware delay. Future validation must report distributions, stationarity and pose/load dependence before freezing compensation or thresholds.
