"""Locked Lope from-scratch numbers.

This module is the configuration contract for a 12-DoF fixed-foot quadruped.
It is not a Thunder wheel-leg run and it is not weight-compatible with Go2 or
Thunder checkpoints. Robot, command, and reward numbers come from Kitjesen/lope
``headless-v67`` unless a comment names a different source. Stair height and
slope gradient are the user ceilings below, not the old 1–10 cm / 0.22 rough cap.
"""
import math

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
# Thigh pitch and calf stay at the solved magnitudes. Hips adduct 0.08 rad
# (about 4.6°). Front hip axes are -X and rear hip axes are +X, so these signs
# move all four feet toward the midline. The root is raised so the original
# 11.333470889715092 mm sole gap remains. Stance width is 0.25688538375184233 m.
STANDING_POSE = {
    "fr_hip_joint": -0.08,
    "fr_thigh_joint": -0.6715534335840251,
    "fr_calf_joint": 1.1,
    "fl_hip_joint": 0.08,
    "fl_thigh_joint": 0.6715534335840251,
    "fl_calf_joint": -1.1,
    "rr_hip_joint": 0.08,
    "rr_thigh_joint": -0.6715534335840251,
    "rr_calf_joint": 1.1,
    "rl_hip_joint": -0.08,
    "rl_thigh_joint": 0.6715534335840251,
    "rl_calf_joint": -1.1,
}
SPAWN_HEIGHT_M = 0.2740510271690917
BASE_HEIGHT_TARGET_M = 0.2627175562793766
SOLE_GAP_M = 0.011333470889715092
STANCE_WIDTH_M = 0.25688538375184233
HIP_ADDUCTION_RAD = 0.08
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

# Official Go2MoECTSRunnerCfg / RobotEnvCfg timing. 5 ms × 4 is unchanged.
PHYSICS_DT_S = 0.005
DECIMATION = 4
POLICY_HZ = 50.0
EPISODE_LENGTH_S = 25.0
NUM_ENVS = 16384
CUMULATIVE_UPDATES = 100000
ROLLOUT_STEPS = 24
CLIP_REWARDS = 10.0
SAVE_INTERVAL = 500

# Official mixed generator is 9 m tiles, 10 rows, 20 columns, border 25 m,
# and max_init_terrain_level 5. Stair height and slope gradient stay the
# approved ceilings: 5–25 cm and tan(35°), not official (0.05, 0.257) / (0.1, 0.568)
# and not Thunder tan(40°). pyramid_sloped_terrain uses height = slope * run.
# Official tread is 0.31 m. Lope keeps 0.25 m, a modest reduction: scaling the
# tread by the height ratio would be about 0.21 m, near the foot length.
SLOPE_MAX_DEG = 35.0
SLOPE_RANGE = (0.0, math.tan(math.radians(SLOPE_MAX_DEG)))
TERRAIN = {
    "size": (9.0, 9.0),
    "border_width": 25.0,
    "num_rows": 10,
    "num_cols": 20,
    "max_init_terrain_level": 5,
    "horizontal_scale": 0.1,
    "vertical_scale": 0.005,
    "slope_threshold": 0.75,
    "flat_proportion": 0.30,
    "rough_proportion": 0.15,
    "rough_noise_range": (0.005, 0.025),
    "rough_noise_step": 0.005,
    "rough_downsampled_scale": 0.2,
    "slope_proportion": 0.10,
    "slope_range": SLOPE_RANGE,
    "slope_max_deg": SLOPE_MAX_DEG,
    "slope_platform_width": 1.5,
    "slope_border_width": 0.25,
    "stair_proportion": 0.10,
    "stair_step_height_range": (0.05, 0.25),
    "stair_step_width": 0.25,
    "official_stair_step_width": 0.31,
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
# lin3 + ang3 + grav3 + cmd3 + q12 + dq12 + act12 + acc12 + torque12
# + official foot-force history 16 (max, then 3 steps, 4 feet) + height 187.
FRAME_DIM = 45
HISTORY_LENGTH = 10
POLICY_DIM = FRAME_DIM * HISTORY_LENGTH
ACTION_DIM = 12
HEIGHT_RAYS = 17 * 11
FOOT_FORCE_DIM = 16
CRITIC_PROPRIO_DIM = 72 + FOOT_FORCE_DIM
CRITIC_DIM = CRITIC_PROPRIO_DIM + HEIGHT_RAYS
EXPERTS = 8
LATENT_DIM = 32

# Sagittal swap FR<->FL and RR<->RL. Signs apply after the permutation.
# Thigh/calf must flip: the standing pose is an odd left/right pair because the
# URDF negates left thigh/calf axes. Hips also flip: each front pair shares -X
# and each rear pair shares +X, so reflection reverses roll rate.
JOINT_PERM = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
JOINT_SIGN = (-1,) * 12
FOOT_PERM = (1, 0, 3, 2)

# Official mixed Newton reward numbers. Height target stays Lope's, never 0.38.
TRACK_LIN_STD = 0.5
TRACK_YAW_STD = 0.5
BASE_HEIGHT_WEIGHT = -1.0
BASE_HEIGHT_FINAL_WEIGHT = -10.0
BASE_HEIGHT_FINAL_UPDATE = 5000
LIN_VEL_Z_WEIGHT = -2.0
LIN_VEL_Z_FINAL_WEIGHT = 0.0
LIN_VEL_Z_FINAL_UPDATE = 1500
JOINT_ACC_WEIGHT = -3.0e-7
JOINT_POWER_WEIGHT = -1.5e-5
JOINT_TORQUE_WEIGHT = -6.0e-5
UNDESIRED_CONTACT_THRESHOLD_N = 110.0
HIP_STAND_STILL_SCALE = 10.0
BAD_ORIENTATION_LIMIT_RAD = 2.0
# Official command/linear-velocity observation scales. x is 0.5 because the
# curriculum x range is wider than y; this is not a second speed scale.
OBS_LIN_VEL_SCALE = (0.5, 1.0, 1.0)
OBS_COMMAND_SCALE = (0.5, 1.0, 1.0)

# Go2's 0.38 m height is not a Lope reward target and is not a speed scale.
GO2_BASE_HEIGHT_M = 0.38

# RLGymVelCommandCfg.ranges defaults, before any command_range_curriculum stage.
# These official values are used unchanged. Do not multiply them by height.
OFFICIAL_INITIAL_RANGES = {
    "lin_vel_x": (-0.5, 0.5),
    "lin_vel_y": (-0.5, 0.5),
    "ang_vel_yaw": (-1.0, 1.0),
}
# RLGymVelCommandCfg.command_range_curriculum, used unchanged.
OFFICIAL_COMMAND_RANGE_CURRICULUM = (
    {"iter": 5000, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-2.0, 2.0)},
    {"iter": 15000, "lin_vel_x": (-2.0, 2.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-3.0, 3.0)},
    {"iter": 25000, "lin_vel_x": (-3.0, 3.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-3.0, 3.0)},
    {"iter": 35000, "lin_vel_x": (-4.0, 4.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-3.0, 3.0)},
    {"iter": 45000, "lin_vel_x": (-4.5, 4.2), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-3.0, 3.0)},
)
# CommandsCfg override of limit_vel_curriculum. These are direction codes, not speeds.
OFFICIAL_LIMIT_VEL_CURRICULUM = (
    {"iter": 0, "lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]},
    {"iter": 25000, "lin_vel_x": [-1, 1], "lin_vel_y": [0], "ang_vel_yaw": [-1, 0, 1]},
)
# RLGymVelCommandCfg.terrain_max_command_ranges, without the wheels terrain.
OFFICIAL_TERRAIN_MAX_COMMANDS = {
    "flat": {"lin_vel_x": (-4.5, 4.2), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-3.0, 3.0)},
    "wave": {"lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-1.5, 1.5)},
    "slope_up": {"lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-1.5, 1.5)},
    "slope_down": {"lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-1.5, 1.5)},
    "rough_slope": {"lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-1.5, 1.5)},
    "stairs_up": {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-1.5, 1.5)},
    "stairs_down": {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-1.5, 1.5)},
    "obstacles": {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-1.5, 1.5)},
    "small_squares": {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-1.5, 1.5)},
}
OFFICIAL_CLIP_RULES = (
    {
        "command": "lin_vel_x",
        "limit": 2.0,
        "clip": {"lin_vel_y": (-0.5, 0.5), "ang_vel_yaw": (-1.0, 1.0)},
    },
    {
        "command": "ang_vel_yaw",
        "limit": 2.0,
        "clip": {"lin_vel_x": (-2.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
    },
)
OFFICIAL_PUSH_LINEAR = {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}
OFFICIAL_PUSH_ANGULAR = {"roll": (-0.6, 0.6), "pitch": (-0.6, 0.6), "yaw": (-0.6, 0.6)}
OFFICIAL_RESET_LINEAR_VELOCITY = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)}
OFFICIAL_RESET_ANGULAR_VELOCITY = {"roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}
GO2_HEIGHT_SCAN_X = (-0.2, 0.2)
GO2_HEIGHT_SCAN_Y = (-0.15, 0.15)
GO2_OBSTACLE_HEIGHT = (0.05, 0.275)
GO2_WAVE_AMPLITUDE = (0.1, 0.28)


def soft_position_window(lower, upper, factor=SOFT_JOINT_POS_LIMIT_FACTOR):
    """Match SmallDogEnvCfg: clip about the URDF-window center, not about zero."""
    center = (lower + upper) / 2.0
    half = (upper - lower) * factor / 2.0
    return (center - half, center + half)


def action_clips(limits):
    """Soft action windows for the 12 revolute joints, in canonical order."""
    return {name: soft_position_window(*limits[name]) for name in JOINT_NAMES}


# Command speeds are the official numbers. The names below are aliases so the
# task reads Lope fields without a second copy that could drift.
LOPE_INITIAL_RANGES = OFFICIAL_INITIAL_RANGES
LOPE_COMMAND_RANGE_CURRICULUM = OFFICIAL_COMMAND_RANGE_CURRICULUM
LOPE_TERRAIN_MAX_COMMANDS = OFFICIAL_TERRAIN_MAX_COMMANDS
LOPE_CLIP_RULES = OFFICIAL_CLIP_RULES
LOPE_PUSH_LINEAR = OFFICIAL_PUSH_LINEAR
LOPE_RESET_LINEAR_VELOCITY = OFFICIAL_RESET_LINEAR_VELOCITY
ZERO_COMMAND_CURRICULUM = {
    "start_iter": 0,
    "end_iter": 1500,
    "start_value": 0.0,
    "end_value": 0.1,
}
RESAMPLING_TIME_S = 5.0
# Body-sized height crop. This is geometry, not a command-speed scale.
HEIGHT_SCAN_X = (-0.135, 0.135)
HEIGHT_SCAN_Y = (-0.101, 0.101)
TERRAIN["obstacle_width_range"] = (1.0, 2.0)
TERRAIN["obstacle_height_range"] = GO2_OBSTACLE_HEIGHT
TERRAIN["wave_amplitude_range"] = GO2_WAVE_AMPLITUDE
TERRAIN["stairs_up_proportion"] = 0.25
TERRAIN["stairs_down_proportion"] = 0.10
TERRAIN["flat_proportion"] = 0.20

# Same-task fall recovery. Locomotion stays the main objective.
FALLEN_RESET_FRACTION = 0.20
FALLEN_RESET_FRACTION_FINAL = 0.35
FALLEN_FRACTION_END_UPDATE = 20000
FALLEN_Z_DROP_M = (0.06, 0.12)
FALLEN_TIMEOUT_S = 6.0
# Upright projected gravity z is about -1. Above this value the robot is still down.
FALLEN_UPRIGHT_GRAVITY_Z = -0.7
RECOVERY_UPRIGHT_WEIGHT = 2.0
RECOVERY_HEIGHT_WEIGHT = -2.0
STAND_UP_BONUS_WEIGHT = 5.0
RECOVERY_SCORE_SATURATION = 0.9
