"""Thunder hip operating envelope; positive signed coordinate means adduction.

Signs were checked against URDF axes and wheel-center forward kinematics.
Physical joint stops remain +/-0.41 rad. These are commanded operating bounds,
not a statement that contact dynamics cannot overshoot them.
"""
PROFILE = "hip_inward012_v22"
INWARD_SIGNS = {"fr_hip_joint": -1., "fl_hip_joint": 1.,
                "rr_hip_joint": 1., "rl_hip_joint": -1.}
DEFAULT_INWARD = 0.12
INWARD_MAX = 0.12
OUTWARD_MAX = 0.369


def joint_contract(name):
    sign = INWARD_SIGNS[name]
    endpoints = (sign * -OUTWARD_MAX, sign * INWARD_MAX)
    return dict(inward_sign=sign, default_rad=sign * DEFAULT_INWARD,
                lower_rad=min(endpoints), upper_rad=max(endpoints))


def contract():
    return dict(profile=PROFILE, coordinate="inward_sign * actual_absolute_joint_angle",
                default_inward_rad=DEFAULT_INWARD, max_inward_target_rad=INWARD_MAX,
                max_outward_target_rad=OUTWARD_MAX, physical_hard_limit_rad=0.41,
                joints={name: joint_contract(name) for name in INWARD_SIGNS},
                changes_observation_default_and_action_offset=True,
                physical_state_clamping=False, hardware_release=False)
