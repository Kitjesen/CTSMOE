# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""DOSO Thunder v4 articulation configuration."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


_THUNDER_DATA_DIR = Path(__file__).resolve().parent / "data" / "Robots" / "DOSO" / "thunder"

DOSO_THUNDER_V4_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=str(_THUNDER_DATA_DIR / "urdf" / "thunder_v4.urdf"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            "fr_hip_joint": -0.20,
            "fr_thigh_joint": -0.93,
            "fr_calf_joint": 1.96,
            "fl_hip_joint": 0.20,
            "fl_thigh_joint": 0.93,
            "fl_calf_joint": -1.96,
            "rr_hip_joint": 0.20,
            "rr_thigh_joint": 0.93,
            "rr_calf_joint": -1.96,
            "rl_hip_joint": -0.20,
            "rl_thigh_joint": -0.93,
            "rl_calf_joint": 1.96,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=120.0,
            damping=8.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=120.0,
            damping=8.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=120.0,
            damping=8.0,
            friction=0.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=17.0,
            velocity_limit_sim=44.0,
            stiffness=0.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)
"""Configuration of the DOSO Thunder v4 wheeled-legged robot."""
