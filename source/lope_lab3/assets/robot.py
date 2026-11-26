"""Lope URDF import for Newton with 14S RS02 4-quadrant delayed PD actuators."""
import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

from lope_lab3.assets.actuator import RS02ActuatorCfg
from lope_lab3.assets.contract import packaged_urdf, validate_urdf
from lope_lab3.spec import (
    MERGE_FIXED_JOINTS,
    RS02_CALF,
    RS02_HIP_THIGH,
    SELF_COLLISION,
    SOFT_JOINT_POS_LIMIT_FACTOR,
    SPAWN_HEIGHT_M,
    STANDING_POSE,
)


def urdf_path():
    override = os.environ.get("LOPE_URDF")
    return Path(override) if override else packaged_urdf()


def _rs02_actuator(joint_names_expr, model):
    """Build a 14S RobStride RS02 4-quadrant FOC actuator with control delay."""
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
