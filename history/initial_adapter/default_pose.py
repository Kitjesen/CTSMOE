"""Explicit pose reference; never a joint limit or an actuator-gain change."""
import copy

PROFILE = "flat_reference_v1"
EXPORT_FILE = "default_pose.json"
ANGLES = {
    "fr": (-0.12, -0.93, 1.96), "fl": (0.12, 0.93, -1.96),
    "rr": (0.12, 0.93, -1.96), "rl": (-0.12, -0.93, 1.96),
}


def contract(profile):
    if profile == "asset_original":
        return {"schema_version": 1, "profile": profile}
    if profile != PROFILE:
        raise ValueError(f"Unknown default pose: {profile}")
    joints = {f"{side}_{joint}_joint": value for side, angles in ANGLES.items()
              for joint, value in zip(("hip", "thigh", "calf"), angles)}
    joints.update({f"{side}_foot_joint": 0.0 for side in ANGLES})
    return dict(schema_version=1, profile=profile, joint_position_rad=joints,
                joint_velocity_rad_s=0.0, default_is_not_limit=True,
                base_height_policy="retain_rough_terrain_reset_and_height_rewards",
                gains_and_action_scales_changed=False)


def resolve_profile(requested, checkpoint):
    saved = checkpoint.get("default_pose_contract") if checkpoint is not None else None
    inherited = saved.get("profile") if isinstance(saved, dict) else "asset_original"
    if saved is not None and saved != contract(inherited):
        raise ValueError("Checkpoint default pose contract is invalid")
    if checkpoint is not None and requested is not None and requested != inherited:
        raise ValueError("Do not silently change pose when resuming a policy")
    profile = requested or inherited
    contract(profile)
    return profile


def configure(cfg, profile):
    metadata = contract(profile)
    if profile == "asset_original":
        return metadata
    if not cfg.actions.joint_pos.use_default_offset:
        raise ValueError("Pose-relative leg actions require use_default_offset")
    # Replace regex selectors: overlapping old selectors must not win at runtime.
    state = copy.deepcopy(cfg.scene.robot.init_state)
    state.joint_pos = dict(metadata["joint_position_rad"])
    state.joint_vel = {".*": 0.0}
    cfg.scene.robot.init_state = state
    return metadata


def validate_runtime(env, profile):
    if profile == "asset_original":
        return None
    import torch

    robot = env.scene["robot"]
    expected = contract(profile)["joint_position_rad"]
    if set(robot.joint_names) != set(expected):
        raise ValueError("Unexpected robot joint names for flat reference pose")
    for name, value in expected.items():
        j = robot.joint_names.index(name)
        torch.testing.assert_close(robot.data.default_joint_pos[:, j],
                                   torch.full_like(robot.data.default_joint_pos[:, j], value))
    torch.testing.assert_close(robot.data.default_joint_vel,
                               torch.zeros_like(robot.data.default_joint_vel))
    return dict(default_positions_verified=True, default_velocities_verified=True,
                actual_reset_pose_may_be_randomized=True)
