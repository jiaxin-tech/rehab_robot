# Robot-to-Setup Frame Calibration Plan

Estimate `T_B_R` by rigid registration from at least three non-collinear reference points that are physically tied to the rehab setup and observed in a base-associated metrology frame. Prefer redundant fiducials and report registration residuals, leave-one-out error and transform covariance.

Possible non-robot methods include a calibrated 3-D digitizer/photogrammetry system that observes both rigid robot-base fiducials and setup fiducials, or a surveyed rigid jig with certified coordinates. Robot TCP probing is optional only after independent robot safety authorization; this protocol neither requires nor authorizes it.

Freeze device IDs, calibration certificates, point correspondence, transform convention, fiducial coordinates, fit algorithm and thresholds before geometry results. Bed axes are not assumed equal to base/world axes. Missing or failed registration makes `T_B_R=null` and `d_task=null`.
