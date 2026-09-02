# Minimum Blocking Items

Current decision: `STATIC_MEASUREMENT_VALIDATION_EXECUTION_NOT_READY`.

Only the following five consolidated items block the first formal nonhuman static physical validation:

1. **B1_SITE_SAFETY_AND_POSE** — Complete site-specific safety/config review and freeze exact P0 eligibility: robot identity, tool/TCP/payload, workspace/joint/collision limits, stationary state, E-stop/operator roles, dwell/abort/unload/cleanup. P1/P2 require separate motion approval.

2. **B2_TRACEABLE_LOAD_SYSTEM** — Provide an identified, calibrated hands-free force gauge/load cell plus direction-controlled fixture, safe mounting/secondary retention, calibration certificate and uncertainty.

3. **B3_LOAD_AND_THRESHOLD_FREEZE** — Use independent baseline/calibration evidence and reviewed robot/fixture/instrument limits to freeze load magnitudes and all PASS thresholds before formal validation results; values remain null now.

4. **B4_GEOMETRY_KIT_AND_FRAME** — Provide and identify the rigid shank surrogate, production cuff/strap, robot eyelet, repeatable jig/fiducials and calibrated minimum-viable 3-D metrology; freeze exact configurations and T_B_R registration/uncertainty.

5. **B5_STATIC_VALIDATION_LOGGER_DRY_RUN** — Implement only the standalone protocol-specific labels/manifest layer over existing read-only acquisition, then pass an offline/no-load dry run for PRE/LOAD/POST, pose/direction/repeat/load IDs, tool/config/calibration metadata, invalid rows and immutable checksums.

Next action: `RESOLVE_MINIMUM_BLOCKING_ITEMS`.

Do not open another measurement-semantics audit. Resolve these items with equipment records, site review, calibration evidence and the minimal logger dry run; then rerun this readiness audit against the same frozen protocols.
