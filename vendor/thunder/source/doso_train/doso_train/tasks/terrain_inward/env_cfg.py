# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DOSO Thunder v4 terrain-inward velocity task configurations."""

import math

from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from doso_train.assets import DOSO_THUNDER_V4_CFG

from . import mdp


BASE_LINK_NAME = "base_link"
FOOT_LINK_PATTERN = ".*_foot_Link"
LEG_JOINT_NAMES = [
    "fr_hip_joint",
    "fr_thigh_joint",
    "fr_calf_joint",
    "fl_hip_joint",
    "fl_thigh_joint",
    "fl_calf_joint",
    "rr_hip_joint",
    "rr_thigh_joint",
    "rr_calf_joint",
    "rl_hip_joint",
    "rl_thigh_joint",
    "rl_calf_joint",
]
WHEEL_JOINT_NAMES = ["fr_foot_joint", "fl_foot_joint", "rr_foot_joint", "rl_foot_joint"]
JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES

POLICY_OBS_DIM = 53
FLAT_CRITIC_OBS_DIM = 56
HEIGHT_SCAN_DIM = 187
ROUGH_CRITIC_OBS_DIM = FLAT_CRITIC_OBS_DIM + HEIGHT_SCAN_DIM


@configclass
class DosoThunderTerrainInwardCommandsCfg:
    """Velocity command distribution from the RobotLab source task."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.15,
        rel_pure_yaw_envs=0.15,
        rel_pure_lateral_envs=0.10,
        pure_command_min_magnitude=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class DosoThunderTerrainInwardActionsCfg:
    """Twelve leg position actions followed by four wheel velocity actions."""

    joint_pos = env_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    joint_vel = env_mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=5.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class DosoThunderTerrainInwardObservationsCfg:
    """Asymmetric 53-dimensional policy and 56/243-dimensional critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=env_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=env_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=env_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=env_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=env_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                )
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=env_mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=env_mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0
        )
        base_ang_vel = ObsTerm(
            func=env_mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0
        )
        projected_gravity = ObsTerm(
            func=env_mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0
        )
        velocity_commands = ObsTerm(
            func=env_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=env_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
                )
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=env_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                )
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(func=env_mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        height_scan = ObsTerm(
            func=env_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class DosoThunderTerrainInwardEventCfg:
    """Domain randomization copied from the source task's active terms."""

    randomize_rigid_body_material = EventTerm(
        func=env_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 1.2),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    randomize_base_payload = EventTerm(
        func=mdp.randomize_base_payload,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "payload_mass_range": (0.0, 12.0),
            "payload_zero_probability": 0.35,
            "payload_position_range": {
                "x": (-0.15, 0.15),
                "y": (-0.10, 0.10),
                "z": (0.05, 0.20),
            },
            "base_mass_scale_range": (0.95, 1.05),
            "base_com_bias_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
            },
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=env_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[f"^(?!.*{BASE_LINK_NAME}).*"]
            ),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_apply_external_force_torque = EventTerm(
        func=env_mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "force_range": (-30.0, 30.0),
            "torque_range": (-10.0, 10.0),
        },
    )
    randomize_reset_joints = EventTerm(
        func=env_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    randomize_reset_hips = EventTerm(
        func=env_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=env_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_reset_base = EventTerm(
        func=env_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    randomize_push_robot = EventTerm(
        func=env_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class DosoThunderTerrainInwardRewardsCfg:
    """Only rewards with non-zero weight in the source task."""

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=env_mdp.flat_orientation_l2, weight=0.0)
    joint_torques_l2 = RewTerm(
        func=env_mdp.joint_torques_l2,
        weight=-3.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_acc_l2 = RewTerm(
        func=env_mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_acc_wheel_l2 = RewTerm(
        func=env_mdp.joint_acc_l2,
        weight=-2.5e-10,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    joint_pos_limits = RewTerm(
        func=env_mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    wheel_torques_l2 = RewTerm(
        func=env_mdp.joint_torques_l2,
        weight=-1.5e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    wheel_power = RewTerm(
        func=mdp.joint_power,
        weight=-3.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    standing_lin_vel_xy_l2 = RewTerm(
        func=mdp.standing_lin_vel_xy_l2,
        weight=-2.0,
        params={"command_name": "base_velocity"},
    )
    standing_ang_vel_z_l2 = RewTerm(
        func=mdp.standing_ang_vel_z_l2,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )
    standing_wheel_vel_l2 = RewTerm(
        func=mdp.standing_wheel_vel_l2,
        weight=-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
        },
    )
    pure_yaw_lin_vel_xy_l2 = RewTerm(
        func=mdp.pure_yaw_lin_vel_xy_l2,
        weight=-2.0,
        params={"command_name": "base_velocity"},
    )
    pure_yaw_wheel_mirror_l2 = RewTerm(
        func=mdp.pure_yaw_wheel_mirror_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    hip_abs_limit_l2 = RewTerm(
        func=mdp.hip_abs_limit_l2,
        weight=0.0,
        params={
            "limit": 0.28,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
        },
    )
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["fr_(hip|thigh|calf).*", "rl_(hip|thigh|calf).*"],
                ["fl_(hip|thigh|calf).*", "rr_(hip|thigh|calf).*"],
            ],
            "command_name": "base_velocity",
            "standing_scale": 1.0,
            "pure_yaw_scale": 1.0,
            "pure_lateral_scale": 1.0,
            "mixed_scale": 1.0,
        },
    )
    action_rate_l2 = RewTerm(func=env_mdp.action_rate_l2, weight=-0.01)
    wheel_contact_slip_l2 = RewTerm(
        func=mdp.wheel_contact_slip_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "wheel_radius": 0.093,
            "contact_force_threshold": 1.0,
            "velocity_deadzone": 0.02,
            "max_slip_speed": 1.0,
            "collision_center_y": mdp.WHEEL_COLLISION_CENTER_Y,
            "asset_cfg": SceneEntityCfg("robot", body_names=[FOOT_LINK_PATTERN]),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[FOOT_LINK_PATTERN]
            ),
        },
    )
    mode_swing_clearance_l2 = RewTerm(
        func=mdp.mode_swing_clearance_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "wheel_radius": 0.093,
            "lateral_clearance": 0.04,
            "yaw_clearance": 0.03,
            "contact_force_threshold": 1.0,
            "collision_center_y": mdp.WHEEL_COLLISION_CENTER_Y,
            "asset_cfg": SceneEntityCfg("robot", body_names=[FOOT_LINK_PATTERN]),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[FOOT_LINK_PATTERN]
            ),
        },
    )
    minimum_wheel_contacts_l2 = RewTerm(
        func=mdp.minimum_wheel_contacts_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "minimum_contacts": 2,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[FOOT_LINK_PATTERN]
            ),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["^(?!.*_foot_Link).*"]
            ),
            "threshold": 1.0,
        },
    )
    contact_forces = RewTerm(
        func=env_mdp.contact_forces,
        weight=-1.0e-2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[FOOT_LINK_PATTERN]
            ),
            "threshold": 450.0,
        },
    )
    touchdown_velocity_l2 = RewTerm(
        func=mdp.touchdown_velocity_l2,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[FOOT_LINK_PATTERN]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=[FOOT_LINK_PATTERN]),
            "velocity_threshold": 0.15,
        },
    )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.875,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    upward = RewTerm(func=mdp.upward, weight=3.0)


@configclass
class DosoThunderTerrainInwardTerminationsCfg:
    """Only episodic timeouts used by the source task."""

    time_out = DoneTerm(func=env_mdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(
        func=velocity_mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class DosoThunderTerrainInwardCurriculumCfg:
    """Rough terrain-level curriculum; disabled entirely by the flat task."""

    terrain_levels = CurrTerm(func=velocity_mdp.terrain_levels_vel)


@configclass
class DosoThunderTerrainInwardBaseEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Unregistered common DOSO delta on the official Isaac Lab velocity task."""

    observations: DosoThunderTerrainInwardObservationsCfg = (
        DosoThunderTerrainInwardObservationsCfg()
    )
    actions: DosoThunderTerrainInwardActionsCfg = DosoThunderTerrainInwardActionsCfg()
    commands: DosoThunderTerrainInwardCommandsCfg = (
        DosoThunderTerrainInwardCommandsCfg()
    )
    rewards: DosoThunderTerrainInwardRewardsCfg = DosoThunderTerrainInwardRewardsCfg()
    terminations: DosoThunderTerrainInwardTerminationsCfg = (
        DosoThunderTerrainInwardTerminationsCfg()
    )
    events: DosoThunderTerrainInwardEventCfg = DosoThunderTerrainInwardEventCfg()
    curriculum: DosoThunderTerrainInwardCurriculumCfg = (
        DosoThunderTerrainInwardCurriculumCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = DOSO_THUNDER_V4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{BASE_LINK_NAME}"
        self.scene.terrain.physics_material.restitution = 1.0
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class DosoThunderTerrainInwardRoughEnvCfg(DosoThunderTerrainInwardBaseEnvCfg):
    """Rough-terrain DOSO Thunder v4 velocity task."""


@configclass
class DosoThunderTerrainInwardFlatEnvCfg(DosoThunderTerrainInwardBaseEnvCfg):
    """Flat-terrain DOSO Thunder v4 velocity task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
