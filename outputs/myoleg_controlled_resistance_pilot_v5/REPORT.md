# Controlled resistance interaction pilot

Analytical mechanism stress test; not native MyoLeg physiology or patient evidence.
The learner receives only queried values; profile parameters and oracle scoring are evaluator-only.

| Scenario / arm | Profiles | Gate active | Common regret | Policy regret | Improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| COMMON_RESISTANCE / COMMON | 2 | 0% | 0.000000 | 0.000000 | 0.000000 |
| DIVERGENT_RESISTANCE / EARLY | 1 | 100% | 0.038000 | 0.002000 | 0.036000 |
| DIVERGENT_RESISTANCE / FAST | 1 | 100% | 0.000000 | 0.000000 | 0.000000 |
| DIVERGENT_RESISTANCE / HIP | 1 | 100% | 0.019000 | 0.000000 | 0.019000 |
| DIVERGENT_RESISTANCE / LATE | 1 | 100% | 0.038000 | 0.002000 | 0.036000 |

Decision: MECHANISM_PILOT_ELIGIBLE_FOR_ALGORITHM_COMPARISON.
Null gate active rate: 0%; practical divergent common-regret fraction: 75%; practical policy-improvement fraction: 75%.
A profile can trigger the gate without benefiting from personalization (the FAST arm is an explicit example); detection and useful regret reduction are reported separately.
This pilot is a go/no-go mechanism check, not a confirmatory experiment.
No confirmatory profiles were loaded; no claim about patient physiology or clinical safety is supported.
