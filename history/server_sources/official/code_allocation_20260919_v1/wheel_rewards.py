"""Thunder wheel adaptation of the upstream foot-regulation penalty.

The upstream penalty and ground-height estimate are retained. Only its foot
position and velocity are evaluated at the wheel bottom instead of the wheel
center, so ideal rolling is not mistaken for foot dragging. The bottom point
uses gravity as the surface normal; it is an approximation on sloped terrain.
The reference point is the wheel link origin; the collider's approximately
16 mm axial offset is omitted, so this is not an exact collision-contact query.
"""

from __future__ import annotations

import torch


def wheel_contact_velocity(
    center_velocity_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
    wheel_radius: float,
    down_w: torch.Tensor,
) -> torch.Tensor:
    """Velocity at a wheel bottom, with ``down_w`` a unit world vector."""
    if wheel_radius <= 0:
        raise ValueError("wheel_radius must be positive")
    bottom_offset_w = (wheel_radius * down_w).expand_as(angular_velocity_w)
    return center_velocity_w + torch.linalg.cross(
        angular_velocity_w, bottom_offset_w, dim=-1
    )


def wheel_regulation_penalty(
    center_positions_w: torch.Tensor,
    center_velocities_w: torch.Tensor,
    angular_velocities_w: torch.Tensor,
    base_positions_w: torch.Tensor,
    base_heights: torch.Tensor,
    gravity_w: torch.Tensor,
    base_height_target: float,
    wheel_radius: float,
) -> torch.Tensor:
    """Return the original XY-speed-squared/height penalty at wheel bottoms."""
    if base_height_target <= 0:
        raise ValueError("base_height_target must be positive")
    down_w = gravity_w / torch.linalg.vector_norm(gravity_w)
    contact_velocities_w = wheel_contact_velocity(
        center_velocities_w, angular_velocities_w, wheel_radius, down_w
    )
    bottom_positions_w = center_positions_w + wheel_radius * down_w
    bottom_to_base_height = torch.sum(
        (bottom_positions_w - base_positions_w.unsqueeze(1)) * down_w, dim=-1
    )
    bottom_heights = torch.clamp(
        base_heights.unsqueeze(1) - bottom_to_base_height, min=0.0
    )
    return (
        contact_velocities_w[..., :2].square().sum(dim=-1)
        * torch.exp(-bottom_heights / (0.025 * base_height_target))
    ).sum(dim=-1)


def wheel_feet_regulation(
    env,
    base_height_target: float,
    wheel_radius: float,
    asset_cfg,
    sensor_cfg=None,
) -> torch.Tensor:
    """Reward-manager entry point; simulator imports stay out of CPU helpers."""
    from robot_lab.tasks.go2.mdp.rewards import _get_base_height

    asset = env.scene[asset_cfg.name]
    wheel_ids = asset_cfg.body_ids
    # Link-origin velocities are required: COM linear velocities can differ
    # because Thunder's wheel inertial origins have an axial offset.
    positions_w = asset.data.body_link_pos_w[:, wheel_ids]
    return wheel_regulation_penalty(
        center_positions_w=positions_w,
        center_velocities_w=asset.data.body_link_lin_vel_w[:, wheel_ids],
        angular_velocities_w=asset.data.body_ang_vel_w[:, wheel_ids],
        base_positions_w=asset.data.root_pos_w,
        base_heights=_get_base_height(env, base_height_target, asset_cfg, sensor_cfg),
        gravity_w=torch.as_tensor(
            env.sim.cfg.gravity, device=positions_w.device, dtype=positions_w.dtype
        ),
        base_height_target=base_height_target,
        wheel_radius=wheel_radius,
    )
