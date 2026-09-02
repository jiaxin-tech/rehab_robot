# Known Load Application Plan

## Required validation equipment

- independently calibrated bidirectional force gauge or load cell with current certificate and uncertainty;
- rigid fixture or low-friction cable/pulley arrangement that does not require a person to hold the load;
- calibrated masses only for directions that can be registered to gravity without unsafe side loading;
- independent metrology/inclinometer/fixture registration for world axes and load line;
- non-human rigid end-effector attachment/phantom and secondary retention against dropped masses;
- fixture/load identifiers, calibration records and environmental metadata.

Operator hand pushing and human/subject loading are prohibited as primary calibrated evidence. The controller/TCP orientation alone cannot define a physical world load. Before execution, register the fixture axes to controller world using independent physical references and freeze the transform and its uncertainty.

Two load roles are preregistered: `L1_REVIEWED_LOW` and `L2_REVIEWED_HIGH`. Their N values are null: `FORCE_MAGNITUDE_REQUIRES_SAFETY_REVIEW`. Reviewers must choose values before physical results that are above calibrated zero/noise resolution yet below the most conservative reviewed robot, fixture and load limits with margin. The SDK output cannot be used to choose or increase them. If only one level is approved, frame/sign may be assessed but magnitude linearity is unavailable.

Apply the complete frozen matrix. A cable/pulley or rigid force-gauge arrangement may replace another method only if it produces the same preregistered world vector. If an axis cannot be safely/reliably implemented, record `AXIS_NOT_EXECUTED`; do not substitute a different direction or delete it after seeing results. The future decision can then be partial or not validated, never full.
