# Strap Setup Repeatability Plan

For each of `10` independent setups: remove the strap/cuff completely, reset the surrogate jig, reinstall from the same landmarks with the same wrap/closure sequence, establish the reviewed nonhuman geometry state, and digitize the complete geometry `3` times without reinstalling.

Within-setup repeats estimate instrument/point-picking noise. Between-setup variation estimates cuff placement, exit and direction repeatability. Report robot attachment covariance (including fixture reinstall where applicable), exit/line covariance, setup-to-setup displacement, free-span line-fit residual, direction angular SD/P95/max and failures by reason.

The repeat count must not be increased after results. Repeatability, displacement and angular thresholds remain null until metrology accuracy and acceptable endpoint error are independently reviewed and frozen. A failed threshold yields `STRAP_PULL_GEOMETRY_NOT_VALIDATED` or a preregistered limitation; it never triggers a guessed direction.
