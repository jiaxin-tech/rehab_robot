# Zero, Bias and Drift Plan

For every pose block and every independent load repetition, retain raw `PRE_LOAD_ZERO`, `LOAD`, and `POST_LOAD_ZERO` windows. Each window targets 100 valid host queries; if a future reviewed maximum dwell expires first, the cell is incomplete rather than silently shortened.

For Fx/Fy/Fz separately report raw mean, median, SD, minimum/maximum, valid count, query duration and window timestamps. Report post-minus-pre drift, within-pose baseline dispersion, between-repetition drift and between-pose zero difference. Preserve raw values before any software zero.

The validation response contrast is preregistered as `DeltaF = mean(F_load) - 0.5*(mean(F_pre)+mean(F_post))`. This local contrast isolates the applied-load response for frame/sign testing; it is not a conclusion that simple zero subtraction is an appropriate production compensation model. Results must also be shown raw and with pre-only/post-only contrasts as diagnostics. Pose dependence cannot be labelled sensor error while controller compensation/tool/load state remains unresolved.
