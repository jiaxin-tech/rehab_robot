# Cohort Versioning and Split Plan

`NEW_VERSION_REQUIRED = true`. Preserve `MYOLEG_VIRTUAL_PATIENT_COHORT_V1`, all 32 subject identities, and its 24/8 split. A scientifically changed structural parameterization must use `MYOLEG_VIRTUAL_PATIENT_COHORT_V2`.

Before any V2 generation, freeze: factor semantics; evidence-backed bounds; joint distribution and dependence assumptions; deterministic seed; sample count; development/held-out identities; and all feasibility/integrity gates. Generate both new development and new held-out subjects from the same preregistered V2 design, while keeping held-out truth inaccessible until a separate confirmatory authorization. Do not recycle V1 development outcomes as V2 confirmation, and do not assume the V1 held-out set covers the V2 structural space.

Recommended sequence: evidence closure -> `MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_V1` -> V2 cohort design protocol -> V2 generation -> development-only truth work -> separately authorized confirmatory held-out stage. The pilot is not executed here.
