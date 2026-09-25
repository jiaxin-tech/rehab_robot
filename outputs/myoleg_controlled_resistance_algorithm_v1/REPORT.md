# Controlled resistance algorithm comparison

Development-only controlled synthetic mechanism test; no native or patient claim.
Random has five seeds; other methods are deterministic under the frozen protocol.

| Scenario | Method | Runs | Gate active | Common regret | Policy regret | Improvement | Trials |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COMMON_RESISTANCE | COMMON_POLICY | 2 | 0% | 0.000000 | 0.000000 | 0.000000 | 5.0 |
| COMMON_RESISTANCE | RANDOM | 10 | 0% | 0.000000 | 0.000000 | 0.000000 | 8.0 |
| COMMON_RESISTANCE | RESIDUAL_BO | 2 | 0% | 0.000000 | 0.000000 | 0.000000 | 8.0 |
| COMMON_RESISTANCE | SAST_BO | 2 | 0% | 0.000000 | 0.000000 | 0.000000 | 5.0 |
| DIVERGENT_RESISTANCE | COMMON_POLICY | 4 | 0% | 0.023750 | 0.009500 | 0.014250 | 5.0 |
| DIVERGENT_RESISTANCE | RANDOM | 20 | 0% | 0.023750 | 0.008550 | 0.015200 | 8.0 |
| DIVERGENT_RESISTANCE | RESIDUAL_BO | 4 | 0% | 0.023750 | 0.000000 | 0.023750 | 8.0 |
| DIVERGENT_RESISTANCE | SAST_BO | 4 | 100% | 0.023750 | 0.001000 | 0.022750 | 8.0 |

No confirmatory profiles were loaded. Results are software mechanism evidence only.
