# Robot-Side Attachment Definition

The primary physical point is `PHYSICAL_STRAP_EYELET_OR_HOOK_LOAD_TRANSFER_CENTER`: the center through which the taut free strap transfers load to the rigid robot fixture. TCP and flange origins are frame references, not default attachment points.

For a fixed local point `p_attach_TCP`, future reconstruction is

`p_robot_attach_B(t) = T_B_TCP(t) * homogeneous(p_attach_TCP)`.

`p_attach_TCP` remains null. It must be measured independently by calibrated caliper/CMM/3-D digitization or a rigid fixture survey. A removable fixture requires remove/reinstall repetitions. TCP may substitute only if the measured offset is consistent with zero inside a preregistered uncertainty bound. No offset is invented here.
