"""Pure-Python task, terrain, command, reward, and observation constants for Lope locomotion."""
import math

TASK_ID = "lope-locomotion-mixed"
EXPERIMENT_NAME = "lope_locomotion_mixed"

FEATURE_NAMES = (
    "joint_acc_12",
    "joint_applied_torque_12",
    "foot_contact_force_norm_16",
)

# Official Go2MoECTSRunnerCfg / RobotEnvCfg timing. 5 ms x 4 is unchanged.
PHYSICS_DT_S = 0.005
DECIMATION = 4
POLICY_HZ = 50.0
EPISODE_LENGTH_S = 25.0
NUM_ENVS = 16384
CUMULATIVE_UPDATES = 100000
ROLLOUT_STEPS = 24
CLIP_REWARDS = 10.0
SAVE_INTERVAL = 500

# Mixed terrain generator: 9 m tiles, 10 rows, 20 columns, border 25 m, max_init_terrain_level=1.
# Stair step height stays 5-25 cm (tread 0.25 m); slope gradient stays (0, tan(35 deg)).
SLOPE_MAX_DEG = 35.0
SLOPE_RANGE = (0.0, math.tan(math.radians(SLOPE_MAX_DEG)))
GO2_HEIGHT_SCAN_X = (-0.2, 0.2)
GO2_HEIGHT_SCAN_Y = (-0.15, 0.15)
GO2_OBSTACLE_HEIGHT = (0.05, 0.275)
GO2_WAVE_AMPLITUDE = (0.1, 0.28)

TERRAIN = {
    "size": (9.0, 9.0),
    "border_width": 25.0,
    "num_rows": 10,
    "num_cols": 20,
    "max_init_terrain_level": 1,
    "horizontal_scale": 0.1,
    "vertical_scale": 0.005,
    "slope_threshold": 0.75,
    "flat_proportion": 0.20,
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
    "stairs_up_proportion": 0.25,
    "stairs_down_proportion": 0.10,
    "stair_step_height_range": (0.05, 0.25),
    "stair_step_width": 0.25,
    "official_stair_step_width": 0.31,
    "stair_platform_width": 1.5,
    "stair_border_width": 0.5,
    "obstacle_proportion": 0.10,
    "obstacle_width_range": (1.0, 2.0),
    "obstacle_height_range": GO2_OBSTACLE_HEIGHT,
    "obstacle_count": 20,
    "wave_proportion": 0.05,
    "wave_amplitude_range": GO2_WAVE_AMPLITUDE,
    "wave_count": 3,
}

# Robot_lab Go2/Lope critic, 12 revolute joints, no wheel block.
# lin3 + ang3 + grav3 + cmd3 + q12 + dq12 + act12 + acc12 + torque12
# + official foot-force history 16 (max, then 3 steps, 4 feet) + height 187.
FRAME_DIM = 45
HISTORY_LENGTH = 10
POLICY_DIM = FRAME_DIM * HISTORY_LENGTH
HEIGHT_RAYS = 17 * 11
FOOT_FORCE_DIM = 16
CRITIC_PROPRIO_DIM = 72 + FOOT_FORCE_DIM
CRITIC_DIM = CRITIC_PROPRIO_DIM + HEIGHT_RAYS
EXPERTS = 8
LATENT_DIM = 32

# Mixed Newton reward and pure locomotion termination constants.
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
# Penalize shin/thigh collisions against 90-deg stair risers (~23% of 13.3 kg body weight).
UNDESIRED_CONTACT_THRESHOLD_N = 30.0
HIP_STAND_STILL_SCALE = 10.0
BAD_ORIENTATION_LIMIT_RAD = 2.0
ILLEGAL_CONTACT_THRESHOLD_N = 1.0

OBS_LIN_VEL_SCALE = (0.5, 1.0, 1.0)
OBS_COMMAND_SCALE = (0.5, 1.0, 1.0)
GO2_BASE_HEIGHT_M = 0.38

OFFICIAL_INITIAL_RANGES = {
    "lin_vel_x": (-0.5, 0.5),
    "lin_vel_y": (-0.5, 0.5),
    "ang_vel_yaw": (-1.0, 1.0),
}
OFFICIAL_COMMAND_RANGE_CURRICULUM = (
    {"iter": 5000, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-2.0, 2.0)},
    {"iter": 15000, "lin_vel_x": (-2.0, 2.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-3.0, 3.0)},
    {"iter": 25000, "lin_vel_x": (-3.0, 3.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-3.0, 3.0)},
    {"iter": 35000, "lin_vel_x": (-4.0, 4.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-3.0, 3.0)},
    {"iter": 45000, "lin_vel_x": (-4.5, 4.2), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-3.0, 3.0)},
)
OFFICIAL_LIMIT_VEL_CURRICULUM = (
    {"iter": 0, "lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]},
    {"iter": 25000, "lin_vel_x": [-1, 1], "lin_vel_y": [0], "ang_vel_yaw": [-1, 0, 1]},
)
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
