"""Change the nominal hip pose only; retain the inherited motion limits."""
PROFILE = "hip_default012_v23"
INWARD_SIGNS = {"fr_hip_joint": -1., "fl_hip_joint": 1.,
                "rr_hip_joint": 1., "rl_hip_joint": -1.}
DEFAULT_INWARD = 0.12
SOFT_LIMIT = 0.369


def joint_contract(name):
    return dict(inward_sign=INWARD_SIGNS[name],
                default_rad=INWARD_SIGNS[name] * DEFAULT_INWARD,
                lower_rad=-SOFT_LIMIT, upper_rad=SOFT_LIMIT)


def contract():
    return dict(profile=PROFILE, coordinate="inward_sign * absolute_joint_angle",
                default_inward_rad=DEFAULT_INWARD,
                inherited_soft_limit_rad=SOFT_LIMIT, physical_hard_limit_rad=0.41,
                extra_inward_target_cap=False, motion_limits_changed=False,
                joints={name: joint_contract(name) for name in INWARD_SIGNS},
                changes_observation_default_and_action_offset=True,
                physical_state_clamping=False, hardware_release=False)
