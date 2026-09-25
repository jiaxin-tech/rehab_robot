# Controlled resistance algorithm comparison

Development-only controlled synthetic mechanism test; no native or patient claim.
All methods use the same reference plus frozen calibration probes and the same maximum budget. COMMON_POLICY always recommends the reference; fixed common and inactive-gate fallback may stop after calibration because the recommendation is already observed, and the actual trial cost is reported. RANDOM uses five seeds and the other methods are deterministic.

| Scenario | Method | Runs | Gate active | Common regret | Policy regret | Improvement | Trials |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COMMON_RESISTANCE | COMMON_POLICY | 2 | 0% | 0.000000 | 0.000000 | 0.000000 | 5.0 |
| COMMON_RESISTANCE | PURE_EI | 2 | 0% | 0.000000 | 0.000000 | 0.000000 | 8.0 |
| COMMON_RESISTANCE | RANDOM | 10 | 0% | 0.000000 | 0.000000 | 0.000000 | 8.0 |
| COMMON_RESISTANCE | RESIDUAL_GREEDY | 2 | 0% | 0.000000 | 0.000000 | 0.000000 | 8.0 |
| COMMON_RESISTANCE | SAST_BO | 2 | 0% | 0.000000 | 0.000000 | 0.000000 | 5.0 |
| DIVERGENT_RESISTANCE | COMMON_POLICY | 4 | 0% | 0.014250 | 0.014250 | 0.000000 | 5.0 |
| DIVERGENT_RESISTANCE | PURE_EI | 4 | 0% | 0.014250 | 0.000000 | 0.014250 | 8.0 |
| DIVERGENT_RESISTANCE | RANDOM | 20 | 0% | 0.014250 | 0.000000 | 0.014250 | 8.0 |
| DIVERGENT_RESISTANCE | RESIDUAL_GREEDY | 4 | 0% | 0.014250 | 0.000000 | 0.014250 | 8.0 |
| DIVERGENT_RESISTANCE | SAST_BO | 4 | 100% | 0.014250 | 0.000000 | 0.014250 | 8.0 |

No confirmatory profiles were loaded. Results are software mechanism evidence only.
