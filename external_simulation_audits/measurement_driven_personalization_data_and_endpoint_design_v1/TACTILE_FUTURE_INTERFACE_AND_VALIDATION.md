# Tactile Future Interface and Validation

Tactile is a nullable synchronized episode stream, not a current measurement channel and not a primary endpoint. Each frame must carry host monotonic receive time, optional device time, sensor/frame ID, raw matrix and raw unit, calibration ID, calibrated pressure matrix/unit when valid, per-cell validity/missing mask, saturation mask, sensor validity and invalid reason.

Candidate secondary features after independent validation are mean/peak pressure, spatial concentration, center of pressure, active pressure area and temporal stability. They are not comfort labels.

Minimum future evidence: calibration curve and units; zero/bias and drift; repeatability; spatial consistency/orientation; saturation; effective sampling/source-update rate; latency and timestamp provenance; synchronization with robot streams; missing-cell behavior; and strap-placement repeatability. Until these pass, values remain raw/invalid and features remain null. `PRESSURE_IS_NOT_COMFORT_TRUTH`.
