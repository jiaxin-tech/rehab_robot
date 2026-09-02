# Data leakage audit

- Extended policy execution used the unchanged prediction, support, decision guard, exploration ranking, model fitting, and stop logic.
- Virtual truth for missed-improvement/correct-stop analysis was attached only after each EXPLOIT, EXPLORE, or STOP decision was frozen: `true`.
- Post-decision truth was fed back to policy: `false`.
- Held-out final-test data were not loaded or used.
- Truth was not used for proposal, guard calibration, frontier ranking, fitting, stopping, or threshold tuning.
- The 20-trial cap is an offline virtual diagnostic horizon, not a human trial recommendation.
