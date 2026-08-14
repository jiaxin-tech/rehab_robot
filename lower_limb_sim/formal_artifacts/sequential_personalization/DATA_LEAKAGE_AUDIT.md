# DATA_LEAKAGE_AUDIT

Status: **PASS**

The five-parameter estimator receives only the fixed observation whitelist. Subject/scenario IDs, Stage-4.5C generator parameters, true torque terms, validation rows, and held-out rows are excluded from proposal and fitting.

Held-out trajectories are evaluated once, after the search stop reason is fixed. Rejected but actually simulated trials remain adaptation data.

| scenario | subject | initial role | adaptation trials | heldout used in fit | truth calls during proposal |
|---|---:|---:|---:|---:|---:|
| matched_linear | baseline | train | 2 | 0 | unchanged |
| matched_linear | hip_stiff | train | 2 | 0 | unchanged |
| matched_linear | knee_stiff | train | 1 | 0 | unchanged |
| matched_linear | heavy_leg | train | 2 | 0 | unchanged |
| combined_mild | baseline | train | 2 | 0 | unchanged |
| combined_mild | hip_stiff | train | 2 | 0 | unchanged |
| combined_mild | knee_stiff | train | 1 | 0 | unchanged |
| combined_mild | heavy_leg | train | 2 | 0 | unchanged |
