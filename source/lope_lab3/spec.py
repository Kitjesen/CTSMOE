"""Locked Lope from-scratch numbers.

This module is the configuration contract for a 12-DoF fixed-foot quadruped.
It is not a Thunder wheel-leg run and it is not weight-compatible with Go2 or
Thunder checkpoints. Every value below was taken from Kitjesen/lope
``headless-v67`` unless the comment says it is this repository's control timing.
"""

# Assets: https://github.com/Kitjesen/lope branch cursor/headless-urdf-meshes-only-51bf
ASSET_COMMIT = "74e51939d35e31cfa6dc03951af3911fbe0fe565"
URDF_SHA256 = "56cd5b1dc81545cc324e8b741e947ba0ccc90d8343b999286ef2ce550caa3a2b"
# Training numbers: headless-v67 @ 86008cc53e119038959f93fa48b9f1b9287dfb6f
# joint_motion_config.json, smalldog.py RS02 groups, rough_cfg.py, env_cfg.py.
TRAINING_COMMIT = "86008cc53e119038959f93fa48b9f1b9287dfb6f"

TASK_ID = "Lope-Newton3-Fresh-v0"

LEGS = ("fr", "fl", "rr", "rl")
JOINT_NAMES = tuple(f"{leg}_{kind}_joint" for leg in LEGS for kind in ("hip", "thigh", "calf"))
FOOT_BODIES = tuple(f"{leg}_foot_Link" for leg in LEGS)
FOOT_JOINTS = tuple(f"{leg}_foot_joint" for leg in LEGS)

# joint_motion_config.json standing_joint_angles_rad. Left/right signs follow the URDF.
STANDING_POSE = {
    "fr_hip_joint": 0.0,
    "fr_thigh_joint": -0.6715534335840251,
    "fr_calf_joint": 1.1,
    "fl_hip_joint": 0.0,
    "fl_thigh_joint": 0.6715534335840251,
    "fl_calf_joint": -1.1,
    "rr_hip_joint": 0.0,
    "rr_thigh_joint": -0.6715534335840251,
    "rr_calf_joint": 1.1,
    "rl_hip_joint": 0.0,
    "rl_thigh_joint": 0.6715534335840251,
    "rl_calf_joint": -1.1,
}
SPAWN_HEIGHT_M = 0.2674918208774965
BASE_HEIGHT_TARGET_M = 0.2561583499877814
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
ACTION_SCALE_RAD = {"hip": 0.15, "thigh": 0.2, "calf": 0.2}
FOOT_SPHERE_RADIUS_M = 0.032

# URDF position limits from joint_motion_config.json / lope.urdf. Do not widen them.
JOINT_POSITION_LIMITS_RAD = {
    "fr_hip_joint": (-1.05, 1.05),
    "fr_thigh_joint": (-3.140796326794897, 1.9192036732051037),
    "fr_calf_joint": (0.4, 2.1),
    "fl_hip_joint": (-1.05, 1.05),
    "fl_thigh_joint": (-1.9192036732051037, 3.140796326794897),
    "fl_calf_joint": (-2.1, -0.4),
    "rr_hip_joint": (-1.05, 1.05),
    "rr_thigh_joint": (-3.140796326794897, 1.9192036732051037),
    "rr_calf_joint": (0.4, 2.1),
    "rl_hip_joint": (-1.05, 1.05),
    "rl_thigh_joint": (-1.9192036732051037, 3.140796326794897),
    "rl_calf_joint": (-2.1, -0.4),
}

# smalldog.py DCMotorCfg. The 2026-11-12 recovery run and the URDF `effort="17"`
# field flatten every joint to 17 N·m. That model is rejected. Stall torque stays
# 17/34 N·m; the clipped joint effort is 7/14 N·m.
RS02_HIP_THIGH = {
    "effort_limit": 7.0,
    "saturation_effort": 17.0,
    "velocity_limit": 42.9350995991,
    "stiffness": 40.0,
    "damping": 0.8,
    "armature": 0.0042,
}
RS02_CALF = {
    "effort_limit": 14.0,
    "saturation_effort": 34.0,
    "velocity_limit": 21.4675497995,
    "stiffness": 60.0,
    "damping": 1.5,
    "armature": 0.0168,
}
REJECTED_FLAT_EFFORT_NM = 17.0

MERGE_FIXED_JOINTS = False
SELF_COLLISION = True

# SmallDogEnvCfg inherits Go2's 5 ms physics and decimation 4, and this repo's
# Thunder fresh task uses the same 50 Hz policy. Episode length is Lope's 15 s,
# not Thunder's 25 s.
PHYSICS_DT_S = 0.005
DECIMATION = 4
POLICY_HZ = 50.0
EPISODE_LENGTH_S = 15.0

# rough_cfg.py, not the later R6 warm start (1.5 m/s, terrain level 10).
NUM_ENVS = 4096
CUMULATIVE_UPDATES = 3000
ROLLOUT_STEPS = 24

COMMAND_RANGES = {
    "lin_vel_x": (-0.15, 0.35),
    "lin_vel_y": (-0.1, 0.1),
    "ang_vel_z": (-0.3, 0.3),
}
RESAMPLE_TIME_S = (15.0, 15.0)
REL_STANDING_ENVS = 0.1
FORWARD_SPEED = {
    "max_speed": 1.0,
    "increment": 0.1,
    "batch_size": 128,
    "success_ratio": 0.7,
    "x_lower": -0.15,
}

# rough_cfg.py terrain. Stairs stay at 10 cm. slope_range is radians in Isaac Lab 3,
# so 0.22 rad is about 12.6 degrees. R6's 1.5 m/s ceiling is not used.
TERRAIN = {
    "size": (6.0, 6.0),
    "border_width": 10.0,
    "num_rows": 8,
    "num_cols": 20,
    "max_init_terrain_level": 0,
    "horizontal_scale": 0.1,
    "vertical_scale": 0.005,
    "slope_threshold": 0.75,
    "flat_proportion": 0.30,
    "rough_proportion": 0.15,
    "rough_noise_range": (0.005, 0.025),
    "rough_noise_step": 0.005,
    "rough_downsampled_scale": 0.2,
    "slope_proportion": 0.10,
    "slope_range_rad": (0.0, 0.22),
    "slope_platform_width": 1.5,
    "slope_border_width": 0.25,
    "stair_proportion": 0.10,
    "stair_step_height_range": (0.01, 0.10),
    "stair_step_width": 0.25,
    "stair_platform_width": 1.5,
    "stair_border_width": 0.5,
    "obstacle_proportion": 0.10,
    "obstacle_width_range": (0.3, 0.8),
    "obstacle_height_range": (0.01, 0.10),
    "obstacle_count": 20,
    "wave_proportion": 0.05,
    "wave_amplitude_range": (0.01, 0.08),
    "wave_count": 3,
}

# Robot_lab Go2/Lope critic, 12 revolute joints, no wheel block.
# lin3 + ang3 + grav3 + cmd3 + q12 + dq12 + act12 + acc12 + torque12 + foot4 + height187.
FRAME_DIM = 45
HISTORY_LENGTH = 10
POLICY_DIM = FRAME_DIM * HISTORY_LENGTH
ACTION_DIM = 12
HEIGHT_RAYS = 17 * 11
CRITIC_DIM = 76 + HEIGHT_RAYS
EXPERTS = 8
LATENT_DIM = 32

# Sagittal swap FR<->FL and RR<->RL. Signs apply after the permutation.
# Thigh/calf must flip: the standing pose is an odd left/right pair because the
# URDF negates left thigh/calf axes. Hips also flip: each front pair shares -X
# and each rear pair shares +X, so reflection reverses roll rate.
JOINT_PERM = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
JOINT_SIGN = (-1,) * 12
FOOT_PERM = (1, 0, 3, 2)

# Lope env_cfg.py reward overrides on top of the robot_lab Go2 weights.
# Height and vertical-velocity curricula are disabled there (set to None).
TRACK_LIN_STD = 0.2
TRACK_YAW_STD = 0.35
BASE_HEIGHT_WEIGHT = -1.0
LIN_VEL_Z_WEIGHT = -2.0
FEET_AIR_TIME_WEIGHT = 0.25
FEET_AIR_TIME_THRESHOLD_S = 0.12
FLAT_ORIENTATION_WEIGHT = -2.0
BAD_ORIENTATION_LIMIT_RAD = 0.85
UNDESIRED_CONTACT_THRESHOLD_N = 5.0
HIP_STAND_STILL_SCALE = 1.0


def soft_position_window(lower, upper, factor=SOFT_JOINT_POS_LIMIT_FACTOR):
    """Match SmallDogEnvCfg: clip about the URDF-window center, not about zero."""
    center = (lower + upper) / 2.0
    half = (upper - lower) * factor / 2.0
    return (center - half, center + half)


def action_clips(limits):
    """Soft action windows for the 12 revolute joints, in canonical order."""
    return {name: soft_position_window(*limits[name]) for name in JOINT_NAMES}


def advance_forward_ceiling(ceiling, passed, count, curriculum=None):
    """One gate step from rough_cfg.ForwardSpeedCurriculum.

    Returns the new positive-x ceiling and whether this batch advanced it.
    Lateral and yaw ranges are not modified. The ceiling never exceeds 1.0 m/s.
    """
    curriculum = FORWARD_SPEED if curriculum is None else curriculum
    if count < curriculum["batch_size"]:
        return ceiling, False
    ratio = passed / count
    advance = ratio >= curriculum["success_ratio"] and ceiling < curriculum["max_speed"]
    if advance:
        ceiling = min(curriculum["max_speed"], round(ceiling + curriculum["increment"], 6))
    return ceiling, advance
