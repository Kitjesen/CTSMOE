"""Lope URDF import for Newton. RS02 limits override the URDF's flat 17 N·m effort."""
import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

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


def _dc_motor(joint_names_expr, model):
    """Map the historical RS02 fields onto Isaac Lab 3 actuator names.

    ``effort_limit`` is both the explicit clip and the solver effort. Stall torque
    stays on ``saturation_effort`` so the DC curve is not the rejected flat cap.
    """
    return DCMotorCfg(
        joint_names_expr=joint_names_expr,
        actuator_effort_limit=model["effort_limit"],
        joint_effort_limit=model["effort_limit"],
        saturation_effort=model["saturation_effort"],
        actuator_velocity_limit=model["velocity_limit"],
        joint_velocity_limit=model["velocity_limit"],
        stiffness=model["stiffness"],
        damping=model["damping"],
        armature=model["armature"],
        friction=0.0,
    )


def robot_cfg():
    """12 revolute joints, fixed feet, self-collision, fixed joints kept."""
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
            "hip_thigh_rs02": _dc_motor([".*_(hip|thigh)_joint"], RS02_HIP_THIGH),
            "calf_rs02_2to1": _dc_motor([".*_calf_joint"], RS02_CALF),
        },
    )
