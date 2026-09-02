# Pose and Orientation Consistency Plan

H3 requires at least two separately approved non-degenerate tool orientations. Repeat the same registered world ±X/±Y/±Z loads without redefining axes from TCP orientation. Compare normalized response directions and cross-axis leakage across poses. A genuine world-expression output should stay world-consistent; a tool-following response will rotate with tool orientation.

P0 is the existing safe stationary pose candidate. P1/P2 contain no coordinates and this protocol authorizes no positioning. Before results, a separate safety-reviewed positioning procedure must freeze exact joint/TCP poses and prove they are away from joint limits, singularity and workspace boundaries. If no such procedure exists, execute P0 only and report `POSE_DEPENDENCE_NOT_YET_VALIDATED`; full world-frame validation is forbidden.

The previous 9/9 canonical cases remain `MATHEMATICAL_TRANSFORM_VERIFIED` only. Future physical world-axis registration plus known-load results must independently pass before any consideration of `BASE_WRENCH_ROTATION_VERIFIED=true`.
