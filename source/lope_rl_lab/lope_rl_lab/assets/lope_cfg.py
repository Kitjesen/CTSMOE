"""Lope 12-DoF hardware constants, URDF import, and 14S RS02 articulation config."""
import math
import os
from pathlib import Path

# Assets: https://github.com/Kitjesen/lope branch cursor/headless-urdf-meshes-only-51bf
ASSET_COMMIT = "74e51939d35e31cfa6dc03951af3911fbe0fe565"
# Collision geometry cleaned over ASSET_COMMIT for Newton margin=0.01 self-collision.
URDF_SHA256 = "939b56e92233a3866efdc9bcf65e21298cda35547e02b100c9c3086aaa5b40d8"
# Training numbers: headless-v67 @ 86008cc53e119038959f93fa48b9f1b9287dfb6f
TRAINING_COMMIT = "86008cc53e119038959f93fa48b9f1b9287dfb6f"

LEGS = ("fr", "fl", "rr", "rl")
JOINT_NAMES = tuple(f"{leg}_{kind}_joint" for leg in LEGS for kind in ("hip", "thigh", "calf"))
FOOT_BODIES = tuple(f"{leg}_foot_Link" for leg in LEGS)
FOOT_JOINTS = tuple(f"{leg}_foot_joint" for leg in LEGS)
ACTION_DIM = 12

# joint_motion_config.json standing_joint_angles_rad. Left/right signs follow the URDF.
# Thigh pitch and calf stay at the solved magnitudes. Hips adduct 0.08 rad
# (about 4.6 deg). Front hip axes are -X and rear hip axes are +X, so these signs
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
STANDING_HEIGHT_M = 0.2627175562793766
# Raised locomotion height target (matching Go2's +5 cm target-over-default philosophy):
# 0.295 m base_link origin -> 0.355 m hip height (~80% straight-leg limit, ~100 deg knee)
# while preserving STANDING_POSE and SPAWN_HEIGHT_M for hardware zero-pose and hot-resume.
BASE_HEIGHT_TARGET_M = 0.295
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

# RobStride RS02 @ 14S nominal (~51.8 V, 48 V loaded floor), plus RK3588 USB-CAN delay.
# The flat 17 N m effort cap from URDF / 2026-11-12 recovery run remains rejected.
RS02_BATTERY_CELLS = 14
RS02_HIP_THIGH = {
    "peak_torque_motoring": 14.5,
    "peak_torque_braking": 16.5,
    "continuous_torque_nm": 7.0,
    "corner_velocity": 24.0,
    "velocity_limit": 42.9350995991,
    "stiffness": 40.0,
    "damping": 0.8,
    "armature": 0.0042,
    "friction": 0.01,
    "min_delay": 0,
    "max_delay": 3,
}
RS02_CALF = {
    "peak_torque_motoring": 28.0,
    "peak_torque_braking": 32.0,
    "continuous_torque_nm": 14.0,
    "corner_velocity": 12.0,
    "velocity_limit": 21.4675497995,
    "stiffness": 60.0,
    "damping": 1.5,
    "armature": 0.0168,
    "friction": 0.02,
    "min_delay": 0,
    "max_delay": 3,
}
REJECTED_FLAT_EFFORT_NM = 17.0

MERGE_FIXED_JOINTS = False
# Internal battery and phantom thigh/hip collisions are cleaned in lope.urdf for Newton margin=0.01.
SELF_COLLISION = True
SOLVER_NJMAX = 512
BODY_COM_OFFSET_X = (-0.05, 0.05)
BODY_COM_OFFSET_Y = (-0.03, 0.03)
BODY_COM_OFFSET_Z = (-0.02, 0.03)

# Sagittal swap FR<->FL and RR<->RL. Signs apply after the permutation.
# Thigh/calf must flip: the standing pose is an odd left/right pair because the
# URDF negates left thigh/calf axes. Hips also flip: each front pair shares -X
# and each rear pair shares +X, so reflection reverses roll rate.
JOINT_PERM = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
JOINT_SIGN = (-1,) * 12
FOOT_PERM = (1, 0, 3, 2)


def soft_position_window(lower, upper, factor=SOFT_JOINT_POS_LIMIT_FACTOR):
    """Clip about the URDF-window center, not about zero."""
    center = (lower + upper) / 2.0
    half = (upper - lower) * factor / 2.0
    return (center - half, center + half)


def action_clips(limits):
    """Soft action windows for the 12 revolute joints, in canonical order."""
    return {name: soft_position_window(*limits[name]) for name in JOINT_NAMES}


def urdf_path():
    """Return the overridden LOPE_URDF or the packaged cleaned Lope URDF."""
    from lope_rl_lab.assets.contract import packaged_urdf

    override = os.environ.get("LOPE_URDF")
    return Path(override) if override else packaged_urdf()


def _rs02_actuator(joint_names_expr, model):
    """Build a 14S RobStride RS02 4-quadrant FOC actuator with control delay."""
    from lope_rl_lab.assets.rs02_actuator import RS02ActuatorCfg

    return RS02ActuatorCfg(
        joint_names_expr=joint_names_expr,
        peak_torque_motoring=model["peak_torque_motoring"],
        peak_torque_braking=model["peak_torque_braking"],
        corner_velocity=model["corner_velocity"],
        no_load_velocity=model["velocity_limit"],
        joint_velocity_limit=model["velocity_limit"],
        stiffness=model["stiffness"],
        damping=model["damping"],
        armature=model["armature"],
        friction=model["friction"],
        min_delay=model["min_delay"],
        max_delay=model["max_delay"],
    )


def robot_cfg():
    """12 revolute joints, fixed feet, self-collision enabled on cleaned URDF, fixed joints kept."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.converters import UrdfConverterCfg
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg

    from lope_rl_lab.assets.contract import validate_urdf

    if not math.isfinite(SPAWN_HEIGHT_M) or SPAWN_HEIGHT_M <= 0:
        raise ValueError("Spawn height must be finite and positive")
    asset = validate_urdf(urdf_path())
    cache = os.environ.get("LOPE_USD_CACHE")
    if not cache:
        raise ValueError("Set LOPE_USD_CACHE before importing the Lope URDF")
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=asset["urdf"],
            usd_dir=str(Path(cache).resolve()),
            force_usd_conversion=True,
            fix_base=False,
            merge_fixed_joints=MERGE_FIXED_JOINTS,
            collision_from_visuals=False,
            collision_type="Convex Hull",
            self_collision=SELF_COLLISION,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            activate_contact_sensors=True,
            articulation_props=[NewtonArticulationCfg(self_collision_enabled=SELF_COLLISION)],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, SPAWN_HEIGHT_M),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=dict(STANDING_POSE),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=SOFT_JOINT_POS_LIMIT_FACTOR,
        actuators={
            "hip_thigh_rs02": _rs02_actuator([".*_(hip|thigh)_joint"], RS02_HIP_THIGH),
            "calf_rs02_2to1": _rs02_actuator([".*_calf_joint"], RS02_CALF),
        },
    )
