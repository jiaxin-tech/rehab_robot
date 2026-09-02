# BO Role Reassessment

## Current decision

`PERSONALIZED_BO_NOT_YET_JUSTIFIED_WITHOUT_SUBJECT_FEEDBACK`

BO is a trajectory-selection mechanism; it does not create a personalized objective.

- **Mechanical BO:** optimize a prespecified measured force/torque/pressure interaction metric within fixed task and feasibility constraints.
- **Model-informed BO:** use a physics/gray-box prediction as prior or mean function, then update only from the subject's executed trials.
- **Preference-based BO:** optimize a latent preference inferred from direct ratings or pairwise choices, while mechanical quantities act as constraints/features rather than comfort labels.

Without real subject feedback, BO can optimize only a simulator/common mechanical objective. That is an offline method exercise, not evidence of personalized trajectory benefit.
