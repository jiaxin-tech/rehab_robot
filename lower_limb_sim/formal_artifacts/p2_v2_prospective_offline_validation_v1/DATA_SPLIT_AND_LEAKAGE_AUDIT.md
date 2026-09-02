# Data split and leakage audit

- The immutable prospective manifest SHA is `94d33675b2ae51ef80154c3bba92f31b87852267f3cffbaaacc75c3ce0aa1876`.
- DEVELOPMENT contains the 9 predeclared prior cases and contributes no primary prospective metric.
- PROSPECTIVE contains 6 newly pre-registered case IDs selected without truth outcomes.
- HELD_OUT_FINAL_TEST status is `HELD_OUT_FINAL_TEST_NOT_READ`; no held-out loader is called by this runner.
- Frozen local P95/P99 come only from the 324-row designated artifact; no prospective case contributes to calibration.
- Every prospective policy is rerun from fresh initial identification. No development execution history is reused.
- Proposal and ranking complete before one selection token is issued; only then can the virtual execution oracle reveal truth.
- Full truth landscapes and missed-improvement labels are computed only after a policy path is complete and are never fed back.
- P2 V2B is absent because bundle-scale uncertainty is not calibrated. No n-times or square-root-n assumption is used.
- Prospective outcomes did not change a threshold, K, bundle length, objective, generator, model, support gate, or equivalence tolerance.
- No hardware, control, collection, safety, or robot connection code is imported or called.
