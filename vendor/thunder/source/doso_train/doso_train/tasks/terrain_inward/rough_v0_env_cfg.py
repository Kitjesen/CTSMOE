# Copyright (c) 2026 Shanghai Qiongpei Technology Co., Ltd.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Frozen rough-from-zero V0 configuration for DOSO Thunder v4."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import (
    BASE_LINK_NAME,
    FOOT_LINK_PATTERN,
    LEG_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    DosoThunderTerrainInwardBaseEnvCfg,
    DosoThunderTerrainInwardCommandsCfg,
    DosoThunderTerrainInwardEventCfg,
    DosoThunderTerrainInwardObservationsCfg,
    DosoThunderTerrainInwardRewardsCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


LEG_STIFFNESS_DIM = 12
JOINT_DAMPING_DIM = 16
FOOT_MATERIAL_DIM = 3
ROUGH_V0_CRITIC_OBS_DIM = (
    243 + LEG_STIFFNESS_DIM + JOINT_DAMPING_DIM + FOOT_MATERIAL_DIM
)


def actuator_gain_ratios(
    env: ManagerBasedRLEnv,
    actuator_names: Sequence[str],
    gain: Literal["stiffness", "damping"],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return randomized actuator gains divided by their nominal values."""
    asset: Articulation = env.scene[asset_cfg.name]
    default_gain = getattr(asset.data, f"default_joint_{gain}")
    ratios = []
    for name in actuator_names:
        actuator = asset.actuators[name]
        joint_ids = actuator.joint_indices
        nominal = default_gain[:, joint_ids]
        if torch.any(nominal <= 0.0):
            raise ValueError(f"Actuator '{name}' has non-positive nominal {gain}.")
        ratios.append(getattr(actuator, gain) / nominal)
    return torch.cat(ratios, dim=1)


def _mean_selected_materials(
    materials: torch.Tensor,
    num_shapes_per_body: Sequence[int],
    body_ids: Sequence[int],
) -> torch.Tensor:
    """Average static friction, dynamic friction and restitution over selected bodies."""
    shape_ids: list[int] = []
    for body_id in body_ids:
        start = sum(num_shapes_per_body[:body_id])
        shape_ids.extend(range(start, start + num_shapes_per_body[body_id]))
    if not shape_ids:
        raise ValueError("No material shapes resolved for privileged observation.")
    return materials[:, shape_ids].mean(dim=1)


class randomize_foot_material_with_privileged(env_mdp.randomize_rigid_body_material):
    """Randomize foot materials once and retain their exact values for the critic."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ) -> None:
        super().__call__(
            env,
            env_ids,
            static_friction_range,
            dynamic_friction_range,
            restitution_range,
            num_buckets,
            asset_cfg,
            make_consistent,
        )
        if self.num_shapes_per_body is None:
            raise ValueError(
                "Privileged foot material randomization requires explicit body selection."
            )
        if env_ids is None:
            env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        materials = self.asset.root_physx_view.get_material_properties()[env_ids_cpu]
        selected = _mean_selected_materials(
            materials,
            self.num_shapes_per_body,
            self.asset_cfg.body_ids,
        ).to(env.device)
        foot_material = _foot_material_buffer(env)
        foot_material[env_ids_cpu.to(env.device)] = selected


def _foot_material_buffer(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "_doso_foot_material_properties"):
        env._doso_foot_material_properties = torch.zeros(
            env.scene.num_envs,
            FOOT_MATERIAL_DIM,
            device=env.device,
        )
    return env._doso_foot_material_properties


def foot_material_properties(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return [static friction, dynamic friction, restitution] for each environment."""
    return _foot_material_buffer(env)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flag foot contacts dominated by horizontal force, usually a vertical obstacle hit."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    forces_z = torch.abs(forces[..., 2])
    forces_xy = torch.linalg.vector_norm(forces[..., :2], dim=-1)
    stumble = torch.any(forces_xy > 4.0 * forces_z, dim=1).float()
    return (
        stumble
        * torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7)
        / 0.7
    )


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward new foot contacts only while a standing command is active."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    reward = torch.sum(first_contact, dim=-1).float()
    reward *= (
        torch.linalg.vector_norm(env.command_manager.get_command(command_name), dim=1)
        < 0.1
    )
    return (
        reward
        * torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7)
        / 0.7
    )


@configclass
class DosoThunderRoughV0CommandsCfg(DosoThunderTerrainInwardCommandsCfg):
    """Command mixture used by the verified flat model_13500 run."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.10,
        rel_pure_yaw_envs=0.10,
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
            heading=(-torch.pi, torch.pi),
        ),
    )


@configclass
class DosoThunderRoughV0ObservationsCfg(DosoThunderTerrainInwardObservationsCfg):
    """53-D actor and 274-D privileged critic."""

    @configclass
    class CriticCfg(DosoThunderTerrainInwardObservationsCfg.CriticCfg):
        leg_stiffness_ratio = ObsTerm(
            func=actuator_gain_ratios,
            params={"actuator_names": ("hip", "thigh", "calf"), "gain": "stiffness"},
            clip=(0.0, 3.0),
        )
        joint_damping_ratio = ObsTerm(
            func=actuator_gain_ratios,
            params={
                "actuator_names": ("hip", "thigh", "calf", "wheel"),
                "gain": "damping",
            },
            clip=(0.0, 3.0),
        )
        foot_material = ObsTerm(func=foot_material_properties, clip=(0.0, 2.0))

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    critic: CriticCfg = CriticCfg()


@configclass
class DosoThunderRoughV0EventCfg(DosoThunderTerrainInwardEventCfg):
    """model_13500 domain randomization plus exact foot-material capture."""

    randomize_rigid_body_material = EventTerm(
        func=env_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    randomize_foot_material_with_privileged = EventTerm(
        func=randomize_foot_material_with_privileged,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[FOOT_LINK_PATTERN]),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    randomize_base_payload = None
    randomize_rigid_body_mass_base = EventTerm(
        func=env_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_com_positions = EventTerm(
        func=env_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    randomize_reset_hips = None
    randomize_actuator_gains = EventTerm(
        func=env_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class DosoThunderRoughV0RewardsCfg(DosoThunderTerrainInwardRewardsCfg):
    """model_13500 rewards plus wheel effort terms; stumble remains an explicit zero-weight candidate."""

    flat_orientation_l2 = None
    standing_lin_vel_xy_l2 = None
    standing_ang_vel_z_l2 = None
    standing_wheel_vel_l2 = None
    pure_yaw_lin_vel_xy_l2 = None
    hip_abs_limit_l2 = None
    wheel_contact_slip_l2 = None
    mode_swing_clearance_l2 = None
    minimum_wheel_contacts_l2 = None

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
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
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["fr_(hip|thigh|calf).*", "rl_(hip|thigh|calf).*"],
                ["fl_(hip|thigh|calf).*", "rr_(hip|thigh|calf).*"],
            ],
        },
    )
    action_rate_l2 = RewTerm(func=env_mdp.action_rate_l2, weight=-0.01)
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
        weight=-0.01,
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
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.875,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_contact_without_cmd = RewTerm(
        func=feet_contact_without_cmd,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[FOOT_LINK_PATTERN]
            ),
            "command_name": "base_velocity",
        },
    )
    feet_stumble = RewTerm(
        func=feet_stumble,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[FOOT_LINK_PATTERN]
            )
        },
    )
    upward = RewTerm(func=mdp.upward, weight=3.0)


@configclass
class DosoThunderTerrainInwardRoughV0EnvCfg(DosoThunderTerrainInwardBaseEnvCfg):
    """Rough terrain from random actor/critic initialization; no checkpoint inheritance."""

    observations: DosoThunderRoughV0ObservationsCfg = (
        DosoThunderRoughV0ObservationsCfg()
    )
    commands: DosoThunderRoughV0CommandsCfg = DosoThunderRoughV0CommandsCfg()
    events: DosoThunderRoughV0EventCfg = DosoThunderRoughV0EventCfg()
    rewards: DosoThunderRoughV0RewardsCfg = DosoThunderRoughV0RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 5
        for actuator_name in ("hip", "thigh", "calf"):
            actuator = self.scene.robot.actuators[actuator_name]
            actuator.stiffness = 90.0
            actuator.damping = 6.93
