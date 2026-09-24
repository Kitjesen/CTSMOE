# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""DOSO-specific MDP terms missing from Isaac Lab 2.3.2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


COMMAND_DEADZONE = 0.2
PIT_FORWARD_SPEED_RANGE = (0.3, 0.6)
WHEEL_COLLISION_CENTER_Y = {
    "fr_foot_Link": -0.01631,
    "fl_foot_Link": 0.01631,
    "rr_foot_Link": -0.01631,
    "rl_foot_Link": 0.01631,
}


def _upright_scale(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded planar velocity while suppressing reward when tilted."""
    asset: RigidObject = env.scene[asset_cfg.name]
    error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2]
            - asset.data.root_lin_vel_b[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-error / std**2) * _upright_scale(env)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded yaw velocity while suppressing reward when tilted."""
    asset: RigidObject = env.scene[asset_cfg.name]
    error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_ang_vel_b[:, 2]
    )
    return torch.exp(-error / std**2) * _upright_scale(env)


def lin_vel_z_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize vertical velocity while suppressing the term when tilted."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2]) * _upright_scale(env)


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize roll/pitch velocity while suppressing the term when tilted."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1
    ) * _upright_scale(env)


def joint_power(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return absolute mechanical power summed over selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(
            asset.data.joint_vel[:, asset_cfg.joint_ids]
            * asset.data.applied_torque[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def _command_mode_mask(
    env: ManagerBasedRLEnv, command_name: str, attribute: str
) -> torch.Tensor:
    """Return an exact mode mask maintained by the command term."""
    command_term = env.command_manager.get_term(command_name)
    try:
        return getattr(command_term, attribute)
    except AttributeError as exc:
        raise AttributeError(
            f"Command term '{command_name}' does not expose '{attribute}'."
        ) from exc


def standing_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar drift only in explicitly sampled standing environments."""
    asset: RigidObject = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    return penalty * _command_mode_mask(env, command_name, "is_standing_env")


def standing_ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw drift only in explicitly sampled standing environments."""
    asset: RigidObject = env.scene[asset_cfg.name]
    penalty = torch.square(asset.data.root_ang_vel_b[:, 2])
    return penalty * _command_mode_mask(env, command_name, "is_standing_env")


def standing_wheel_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize wheel motion only in explicitly sampled standing environments."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(
        torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
    )
    return penalty * _command_mode_mask(env, command_name, "is_standing_env")


def pure_yaw_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize translation only in explicitly sampled pure-yaw environments."""
    asset: RigidObject = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    return penalty * _command_mode_mask(env, command_name, "is_pure_yaw_env")


def pure_yaw_wheel_mirror_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Match raw wheel speeds during pure yaw; mirrored URDF axes set direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    penalty = torch.mean(
        torch.square(wheel_vel - wheel_vel.mean(dim=1, keepdim=True)), dim=1
    )
    return penalty * _command_mode_mask(env, command_name, "is_pure_yaw_env")


def _lateral_or_yaw_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return torch.logical_or(
        _command_mode_mask(env, command_name, "is_pure_lateral_env"),
        _command_mode_mask(env, command_name, "is_pure_yaw_env"),
    )


def _wheel_collision_centers_b(
    body_names: list[str], collision_center_y: dict[str, float], device: str
) -> torch.Tensor:
    if set(body_names) != set(collision_center_y):
        raise ValueError(
            "Wheel collision offsets do not match resolved bodies: "
            f"{body_names} != {sorted(collision_center_y)}"
        )
    centers = torch.zeros(1, len(body_names), 3, device=device)
    centers[0, :, 1] = torch.tensor(
        [collision_center_y[name] for name in body_names], device=device
    )
    return centers


def _wheel_contact_offset_w(
    body_quat_w: torch.Tensor,
    collision_center_b: torch.Tensor,
    wheel_radius: float,
) -> torch.Tensor:
    """Return the lowest wheel contact offset from each link frame."""
    collision_center_b = collision_center_b.expand(*body_quat_w.shape[:-1], 3)
    wheel_axis_b = torch.zeros_like(collision_center_b)
    wheel_axis_b[..., 1] = 1.0
    wheel_axis_w = quat_apply(body_quat_w, wheel_axis_b)
    ground_normal_w = torch.zeros_like(wheel_axis_w)
    ground_normal_w[..., 2] = 1.0
    radial_up_w = (
        ground_normal_w
        - torch.sum(ground_normal_w * wheel_axis_w, dim=-1, keepdim=True) * wheel_axis_w
    )
    radial_up_w /= torch.linalg.vector_norm(
        radial_up_w, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    collision_center_w = quat_apply(body_quat_w, collision_center_b)
    return collision_center_w - wheel_radius * radial_up_w


def _wheel_contact_point_slip_l2(
    contact: torch.Tensor,
    body_link_lin_vel_w: torch.Tensor,
    body_link_ang_vel_w: torch.Tensor,
    body_link_quat_w: torch.Tensor,
    collision_center_b: torch.Tensor,
    wheel_radius: float,
    velocity_deadzone: float,
    max_slip_speed: float,
) -> torch.Tensor:
    """Measure planar wheel-ground slip without penalizing ideal rolling."""
    contact_offset_w = _wheel_contact_offset_w(
        body_link_quat_w, collision_center_b, wheel_radius
    )
    contact_velocity_w = body_link_lin_vel_w + torch.cross(
        body_link_ang_vel_w, contact_offset_w, dim=-1
    )
    slip_speed = torch.linalg.vector_norm(contact_velocity_w[..., :2], dim=-1)
    slip_speed = torch.clamp(
        slip_speed - velocity_deadzone, min=0.0, max=max_slip_speed
    )
    return torch.sum(contact.float() * slip_speed.square(), dim=1)


class wheel_contact_slip_l2(ManagerTermBase):
    """Penalize wheel contact-point slip during pure lateral and yaw commands."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset_names = [asset.body_names[index] for index in asset_cfg.body_ids]
        sensor_names = [sensor.body_names[index] for index in sensor_cfg.body_ids]
        if asset_names != sensor_names:
            raise ValueError(
                "Wheel body order differs between robot and contact sensor: "
                f"{asset_names} != {sensor_names}"
            )
        collision_center_y = cfg.params["collision_center_y"]
        self._collision_center_b = _wheel_collision_centers_b(
            asset_names, collision_center_y, env.device
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        wheel_radius: float,
        contact_force_threshold: float,
        velocity_deadzone: float,
        max_slip_speed: float,
        collision_center_y: dict[str, float],
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contact = (
            torch.linalg.vector_norm(
                sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1
            )
            > contact_force_threshold
        )
        _ = collision_center_y
        penalty = _wheel_contact_point_slip_l2(
            contact,
            asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids],
            asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids],
            asset.data.body_link_quat_w[:, asset_cfg.body_ids],
            self._collision_center_b,
            wheel_radius,
            velocity_deadzone,
            max_slip_speed,
        )
        return penalty * _lateral_or_yaw_mask(env, command_name)


class mode_swing_clearance_l2(ManagerTermBase):
    """Penalize insufficient wheel clearance only while a wheel is in swing."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset_names = [asset.body_names[index] for index in asset_cfg.body_ids]
        sensor_names = [sensor.body_names[index] for index in sensor_cfg.body_ids]
        if asset_names != sensor_names:
            raise ValueError(
                "Unexpected wheel body order for clearance: "
                f"{asset_names} != {sensor_names}"
            )
        collision_center_y = cfg.params["collision_center_y"]
        self._collision_center_b = _wheel_collision_centers_b(
            asset_names, collision_center_y, env.device
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        wheel_radius: float,
        lateral_clearance: float,
        yaw_clearance: float,
        contact_force_threshold: float,
        collision_center_y: dict[str, float],
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contact = (
            torch.linalg.vector_norm(
                sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1
            )
            > contact_force_threshold
        )
        lateral = _command_mode_mask(env, command_name, "is_pure_lateral_env")
        yaw = _command_mode_mask(env, command_name, "is_pure_yaw_env")
        target = lateral.float() * lateral_clearance + yaw.float() * yaw_clearance
        _ = collision_center_y
        contact_offset_w = _wheel_contact_offset_w(
            asset.data.body_link_quat_w[:, asset_cfg.body_ids],
            self._collision_center_b,
            wheel_radius,
        )
        clearance = (
            asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
            + contact_offset_w[..., 2]
            - env.scene.env_origins[:, None, 2]
        )
        shortfall = torch.clamp(target[:, None] - clearance, min=0.0)
        return torch.sum((~contact).float() * shortfall.square(), dim=1) * (
            lateral | yaw
        )


def minimum_wheel_contacts_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimum_contacts: int,
    contact_force_threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Keep at least two wheel contacts while learning lateral/yaw stepping."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_count = torch.sum(
        torch.linalg.vector_norm(
            sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1
        )
        > contact_force_threshold,
        dim=1,
    )
    missing = torch.clamp(minimum_contacts - contact_count, min=0).float()
    return missing.square() * _lateral_or_yaw_mask(env, command_name)


def _parallel_axis_matrix(offset: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(3, dtype=offset.dtype, device=offset.device)
    return torch.sum(torch.square(offset), dim=-1)[..., None, None] * identity - (
        offset[..., :, None] * offset[..., None, :]
    )


def _compose_point_mass_payload(
    base_mass: torch.Tensor,
    base_com: torch.Tensor,
    base_inertia: torch.Tensor,
    payload_mass: torch.Tensor,
    payload_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine a rigid body and point payload about their resulting center of mass."""
    total_mass = base_mass + payload_mass
    if torch.any(total_mass <= 0.0):
        raise ValueError("Combined rigid-body mass must be positive.")
    total_com = (
        base_mass[..., None] * base_com + payload_mass[..., None] * payload_position
    ) / total_mass[..., None]
    total_inertia = (
        base_inertia
        + base_mass[..., None, None] * _parallel_axis_matrix(base_com - total_com)
        + payload_mass[..., None, None]
        * _parallel_axis_matrix(payload_position - total_com)
    )
    total_inertia = 0.5 * (total_inertia + total_inertia.transpose(-1, -2))
    zero_payload = payload_mass == 0.0
    total_com = torch.where(zero_payload[..., None], base_com, total_com)
    total_inertia = torch.where(
        zero_payload[..., None, None], base_inertia, total_inertia
    )
    if torch.any(torch.linalg.eigvalsh(total_inertia) <= 0.0):
        raise ValueError("Combined rigid-body inertia must be positive definite.")
    return total_mass, total_com, total_inertia


def randomize_base_payload(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    payload_mass_range: tuple[float, float],
    payload_zero_probability: float,
    payload_position_range: dict[str, tuple[float, float]],
    base_mass_scale_range: tuple[float, float],
    base_com_bias_range: dict[str, tuple[float, float]],
) -> None:
    """Randomize base manufacturing properties and attach a point-mass payload."""
    if not 0.0 <= payload_zero_probability <= 1.0:
        raise ValueError("payload_zero_probability must be within [0, 1].")
    if payload_mass_range[0] < 0.0 or payload_mass_range[1] < payload_mass_range[0]:
        raise ValueError("payload_mass_range must be non-negative and ordered.")
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()
    coms = asset.root_physx_view.get_coms().clone()
    inertias = asset.root_physx_view.get_inertias()
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=masses.device)
    else:
        env_ids = env_ids.to(device=masses.device, dtype=torch.long)
    body_id = asset_cfg.body_ids[0]

    base_mass = asset.data.default_mass[env_ids, body_id].to(masses.device)
    base_com = coms[env_ids, body_id, :3]
    base_inertia = (
        asset.data.default_inertia[env_ids, body_id]
        .to(inertias.device)
        .reshape(-1, 3, 3)
    )

    mass_scale = torch.empty_like(base_mass).uniform_(*base_mass_scale_range)
    randomized_mass = base_mass * mass_scale
    randomized_inertia = base_inertia * mass_scale[:, None, None]
    randomized_com = base_com.clone()
    payload_position = torch.empty_like(randomized_com)
    for axis_index, axis in enumerate(("x", "y", "z")):
        randomized_com[:, axis_index] += torch.empty_like(base_mass).uniform_(
            *base_com_bias_range[axis]
        )
        payload_position[:, axis_index].uniform_(*payload_position_range[axis])

    payload_mass = torch.empty_like(base_mass).uniform_(*payload_mass_range)
    payload_mass[torch.rand_like(payload_mass) < payload_zero_probability] = 0.0
    total_mass, total_com, total_inertia = _compose_point_mass_payload(
        randomized_mass,
        randomized_com,
        randomized_inertia,
        payload_mass,
        payload_position,
    )
    masses[env_ids, body_id] = total_mass
    coms[env_ids, body_id, :3] = total_com
    inertias[env_ids, body_id] = total_inertia.reshape(-1, 9)
    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_coms(coms, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)


def _touchdown_velocity_excess_l2(
    first_contact: torch.Tensor,
    previous_foot_vel_z: torch.Tensor,
    velocity_threshold: float,
) -> torch.Tensor:
    """Return squared excess downward speed only for newly contacting feet."""
    excess_downward_speed = torch.clamp(
        -previous_foot_vel_z - velocity_threshold,
        min=0.0,
    )
    return torch.sum(first_contact.float() * torch.square(excess_downward_speed), dim=1)


class touchdown_velocity_l2(ManagerTermBase):
    """Penalize the pre-contact downward foot speed at each first contact."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset: Articulation = env.scene[self._asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[self._sensor_cfg.name]
        asset_names = [asset.body_names[index] for index in self._asset_cfg.body_ids]
        sensor_names = [
            contact_sensor.body_names[index] for index in self._sensor_cfg.body_ids
        ]
        if asset_names != sensor_names:
            raise ValueError(
                "Touchdown body order differs between robot and contact sensor: "
                f"{asset_names} != {sensor_names}"
            )

        self._previous_foot_vel_z = torch.zeros(
            env.num_envs,
            len(self._asset_cfg.body_ids),
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        if env_ids is None:
            self._previous_foot_vel_z.zero_()
        else:
            self._previous_foot_vel_z[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        velocity_threshold: float,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[
            :, sensor_cfg.body_ids
        ]
        penalty = _touchdown_velocity_excess_l2(
            first_contact,
            self._previous_foot_vel_z,
            velocity_threshold,
        )
        current_foot_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
        self._previous_foot_vel_z.copy_(current_foot_vel_z)
        return penalty


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from the default pose when the command is small."""
    penalty = env_mdp.joint_deviation_l1(env, asset_cfg)
    penalty *= (
        torch.norm(env.command_manager.get_command(command_name), dim=1)
        < command_threshold
    )
    return penalty * _upright_scale(env)


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint deviation more strongly when neither command nor body moves."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_norm = torch.linalg.norm(
        env.command_manager.get_command(command_name), dim=1
    )
    body_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    deviation = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1,
    )
    penalty = torch.where(
        torch.logical_or(
            command_norm > command_threshold, body_speed > velocity_threshold
        ),
        deviation,
        stand_still_scale * deviation,
    )
    return penalty * _upright_scale(env)


def hip_abs_limit_l2(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize either hip extreme before it reaches the URDF soft limit."""
    if limit <= 0.0:
        raise ValueError("hip limit must be positive")
    asset: Articulation = env.scene[asset_cfg.name]
    excess = torch.clamp(
        asset.data.joint_pos[:, asset_cfg.joint_ids].abs() - limit, min=0.0
    )
    return torch.sum(excess.square(), dim=1)


def command_mode_scale(
    env: ManagerBasedRLEnv,
    command_name: str,
    standing_scale: float,
    pure_yaw_scale: float,
    pure_lateral_scale: float,
    mixed_scale: float,
) -> torch.Tensor:
    """Return an exact per-environment scale from the sampled command mode."""
    command = env.command_manager.get_term(command_name)
    scale = torch.full(
        (env.num_envs,), mixed_scale, dtype=torch.float, device=env.device
    )
    scale = torch.where(command.is_standing_env, standing_scale, scale)
    scale = torch.where(command.is_pure_yaw_env, pure_yaw_scale, scale)
    return torch.where(command.is_pure_lateral_env, pure_lateral_scale, scale)


def joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
    command_name: str = "base_velocity",
    standing_scale: float = 1.0,
    pure_yaw_scale: float = 1.0,
    pure_lateral_scale: float = 1.0,
    mixed_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize disagreement between diagonal mirrored leg joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    if (
        not hasattr(env, "joint_mirror_joints_cache")
        or env.joint_mirror_joints_cache is None
    ):
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair]
            for joint_pair in mirror_joints
        ]
    penalty = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_mirror_joints_cache:
        penalty += torch.sum(
            torch.square(
                asset.data.joint_pos[:, joint_pair[0][0]]
                - asset.data.joint_pos[:, joint_pair[1][0]]
            ),
            dim=-1,
        )
    if mirror_joints:
        penalty *= 1 / len(mirror_joints)
    return (
        penalty
        * command_mode_scale(
            env,
            command_name,
            standing_scale,
            pure_yaw_scale,
            pure_lateral_scale,
            mixed_scale,
        )
        * _upright_scale(env)
    )


def upward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the source task's upright projected-gravity convention."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(1 - asset.data.projected_gravity_b[:, 2])


def undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Count non-foot contact violations while suppressing the term when tilted."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
        > threshold
    )
    return torch.sum(is_contact, dim=1).float() * _upright_scale(env)


def _get_terrain_column_range(
    terrain_cfg, terrain_name: str, device
) -> tuple[int, int] | None:
    if terrain_cfg.sub_terrains is None or terrain_name not in terrain_cfg.sub_terrains:
        return None
    names = list(terrain_cfg.sub_terrains.keys())
    proportions = torch.tensor(
        [sub_cfg.proportion for sub_cfg in terrain_cfg.sub_terrains.values()],
        device=device,
    )
    proportions /= proportions.sum()
    cumulative = torch.cumsum(proportions, dim=0)
    terrain_index = names.index(terrain_name)
    column_start = round(
        (0.0 if terrain_index == 0 else cumulative[terrain_index - 1].item())
        * terrain_cfg.num_cols
    )
    column_end = round(cumulative[terrain_index].item() * terrain_cfg.num_cols)
    return column_start, column_end


def is_robot_on_terrain(
    env: ManagerBasedEnv, terrain_name: str, asset_name: str = "robot"
) -> torch.Tensor:
    """Return a mask for robots currently occupying the named terrain columns."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    column_range = _get_terrain_column_range(
        terrain.cfg.terrain_generator, terrain_name, env.device
    )
    if column_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    column_start, column_end = column_range
    robot_positions = env.scene[asset_name].data.root_pos_w[:, :2]
    terrain_origins = terrain.terrain_origins[:, :, :2]
    num_rows, num_cols, _ = terrain_origins.shape
    closest_index = torch.argmin(
        torch.cdist(robot_positions, terrain_origins.reshape(num_rows * num_cols, 2)),
        dim=1,
    )
    column_index = closest_index % num_cols
    return (column_index >= column_start) & (column_index < column_end)


class UniformThresholdVelocityCommand(env_mdp.UniformVelocityCommand):
    """Uniform SE(2) command with specialized yaw/lateral modes and pit restrictions."""

    cfg: UniformThresholdVelocityCommandCfg

    def __init__(self, cfg: UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        mode_probability = (
            cfg.rel_standing_envs + cfg.rel_pure_yaw_envs + cfg.rel_pure_lateral_envs
        )
        if mode_probability > 1.0:
            raise ValueError(
                f"Velocity command mode probabilities sum to {mode_probability}, expected <= 1.0."
            )
        if cfg.rel_pure_yaw_envs > 0.0 and not (
            cfg.ranges.ang_vel_z[0] <= -cfg.pure_command_min_magnitude
            and cfg.ranges.ang_vel_z[1] >= cfg.pure_command_min_magnitude
        ):
            raise ValueError(
                "The yaw range must cover both signs of pure_command_min_magnitude."
            )
        if cfg.rel_pure_lateral_envs > 0.0 and not (
            cfg.ranges.lin_vel_y[0] <= -cfg.pure_command_min_magnitude
            and cfg.ranges.lin_vel_y[1] >= cfg.pure_command_min_magnitude
        ):
            raise ValueError(
                "The lateral range must cover both signs of pure_command_min_magnitude."
            )
        self.is_pure_yaw_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_pure_lateral_env = torch.zeros_like(self.is_pure_yaw_env)
        self.was_on_pit = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        rel_standing_envs = self.cfg.rel_standing_envs
        self.cfg.rel_standing_envs = 0.0
        try:
            super()._resample_command(env_ids)
        finally:
            self.cfg.rel_standing_envs = rel_standing_envs
        nonzero = torch.norm(self.vel_command_b[env_ids, :2], dim=1) > COMMAND_DEADZONE
        self.vel_command_b[env_ids, :2] *= nonzero.unsqueeze(1)

        mode = torch.rand(len(env_ids), device=self.device)
        standing_end = self.cfg.rel_standing_envs
        pure_yaw_end = standing_end + self.cfg.rel_pure_yaw_envs
        pure_lateral_end = pure_yaw_end + self.cfg.rel_pure_lateral_envs
        standing = mode < standing_end
        pure_yaw = (mode >= standing_end) & (mode < pure_yaw_end)
        pure_lateral = (mode >= pure_yaw_end) & (mode < pure_lateral_end)
        self.is_standing_env[env_ids] = standing
        self.is_pure_yaw_env[env_ids] = pure_yaw
        self.is_pure_lateral_env[env_ids] = pure_lateral

        specialized_ids = env_ids[standing | pure_yaw | pure_lateral]
        self.is_heading_env[specialized_ids] = False

        standing_ids = env_ids[standing]
        self.vel_command_b[standing_ids] = 0.0

        pure_yaw_ids = env_ids[pure_yaw]
        if len(pure_yaw_ids) > 0:
            yaw = torch.empty(len(pure_yaw_ids), device=self.device).uniform_(
                *self.cfg.ranges.ang_vel_z
            )
            yaw = torch.where(
                yaw >= 0.0,
                yaw.clamp(min=self.cfg.pure_command_min_magnitude),
                yaw.clamp(max=-self.cfg.pure_command_min_magnitude),
            )
            self.vel_command_b[pure_yaw_ids, :2] = 0.0
            self.vel_command_b[pure_yaw_ids, 2] = yaw

        pure_lateral_ids = env_ids[pure_lateral]
        if len(pure_lateral_ids) > 0:
            lateral = torch.empty(len(pure_lateral_ids), device=self.device).uniform_(
                *self.cfg.ranges.lin_vel_y
            )
            lateral = torch.where(
                lateral >= 0.0,
                lateral.clamp(min=self.cfg.pure_command_min_magnitude),
                lateral.clamp(max=-self.cfg.pure_command_min_magnitude),
            )
            self.vel_command_b[pure_lateral_ids, 0] = 0.0
            self.vel_command_b[pure_lateral_ids, 1] = lateral
            self.vel_command_b[pure_lateral_ids, 2] = 0.0

    def _update_command(self):
        super()._update_command()
        on_pits = is_robot_on_terrain(self._env, "pits")
        left_pits = self.was_on_pit & ~on_pits
        if left_pits.any():
            self._resample_command(torch.where(left_pits)[0])
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            self.is_standing_env[pit_env_ids] = False
            self.is_pure_yaw_env[pit_env_ids] = False
            self.is_pure_lateral_env[pit_env_ids] = False
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]),
                min=PIT_FORWARD_SPEED_RANGE[0],
                max=PIT_FORWARD_SPEED_RANGE[1],
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0
            self.vel_command_b[pit_env_ids, 2] = 0.0
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(env_mdp.UniformVelocityCommandCfg):
    """Configuration for :class:`UniformThresholdVelocityCommand`."""

    class_type: type = UniformThresholdVelocityCommand
    rel_pure_yaw_envs: float = 0.0
    rel_pure_lateral_envs: float = 0.0
    pure_command_min_magnitude: float = COMMAND_DEADZONE
