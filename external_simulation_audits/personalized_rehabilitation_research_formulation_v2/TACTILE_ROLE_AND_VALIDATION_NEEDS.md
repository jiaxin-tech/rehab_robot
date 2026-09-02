# Tactile Role and Validation Needs

Future software path:

`raw pressure map -> timestamped preprocessing -> versioned episode features -> subject model / candidate endpoint diagnostics / safety diagnostics`

Potential features include mean/peak pressure, spatial concentration, pressure centroid/center-of-pressure where geometrically meaningful, temporal peak and distribution stability. Before any scientific use, the minimum validation set is:

- sensor calibration and units;
- sampling rate and dropped-sample behavior;
- acquisition latency and timestamp provenance;
- synchronization/skew with robot state and wrench;
- within-session and between-session repeatability;
- spatial mapping, orientation, contact area and sensor placement reproducibility.

Current active hardware/collection/control source contains no tactile acquisition implementation. Tactile is therefore planned, not available now. `pressure != comfort`: pressure is a measured interaction feature or possible comfort correlate. Direct feedback remains required for a comfort/preference target.
