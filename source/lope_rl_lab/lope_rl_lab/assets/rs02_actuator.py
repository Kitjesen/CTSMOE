"""14S RobStride RS02 4-quadrant FOC delayed PD actuator model for Lope."""

from __future__ import annotations

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.actuators.actuator_base import resolve_joint_parameter
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions


class RS02Actuator(DelayedPDActuator):
    """Apply RS02 4-quadrant motoring/braking limits and communication delay."""

    cfg: RS02ActuatorCfg

    def __init__(self, cfg: RS02ActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._peak_motoring = self._resolve_motor_parameter(cfg.peak_torque_motoring, 1.0e9)
        self._peak_braking = self._resolve_motor_parameter(cfg.peak_torque_braking, cfg.peak_torque_motoring)
        self._corner_vel = self._resolve_motor_parameter(cfg.corner_velocity, 1.0e9)
        self._no_load_vel = self._resolve_motor_parameter(cfg.no_load_velocity, 1.0e9)
        if not bool(torch.all((0.0 < self._corner_vel) & (self._corner_vel < self._no_load_vel))):
            raise ValueError("RS02Actuator requires 0 < corner_velocity < no_load_velocity.")
        if not bool(torch.all((0.0 < self._peak_motoring) & (self._peak_motoring <= self._peak_braking))):
            raise ValueError("RS02Actuator requires 0 < peak_torque_motoring <= peak_torque_braking.")

    def _resolve_motor_parameter(self, value, default) -> torch.Tensor:
        """Resolve actuator curve parameters in joint order."""
        return resolve_joint_parameter(value, default, self.joint_names, self._num_envs, self._device)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Compute delayed PD joint efforts bounded by the 4-quadrant RS02 envelope."""
        control_action = super().compute(control_action, joint_pos, joint_vel)
        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """Limit torque across motoring (back-EMF roll-off) and braking quadrants."""
        motoring = (joint_vel * effort) > 0.0
        abs_vel = joint_vel.abs()
        slope = -self._peak_motoring / (self._no_load_vel - self._corner_vel)
        rolled_off = (slope * (abs_vel - self._corner_vel) + self._peak_motoring).clamp(min=0.0)
        motoring_limit = torch.where(abs_vel <= self._corner_vel, self._peak_motoring, rolled_off)
        max_effort = torch.where(motoring, motoring_limit, self._peak_braking)
        return torch.clip(effort, -max_effort, max_effort)


@configclass
class RS02ActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for the 14S RobStride RS02 4-quadrant FOC actuator."""

    class_type: type = RS02Actuator

    joint_effort_limit: float = 1.0e9
    """Leave solver joint clipping open so the 4-quadrant RS02 curve governs torque."""

    actuator_effort_limit: float = 1.0e9
    """Leave base PD effort clipping open so :meth:`RS02Actuator._clip_effort` applies Y1/Y2."""

    peak_torque_motoring: float | dict[str, float] = 14.5
    """Peak motoring torque Y1 (quadrants I & III, effort * vel > 0) [N m]."""

    peak_torque_braking: float | dict[str, float] = 16.5
    """Peak braking/generating torque Y2 (quadrants II & IV, effort * vel <= 0) [N m]."""

    corner_velocity: float | dict[str, float] = 24.0
    """Joint speed X1 where motoring back-EMF roll-off begins [rad/s]."""

    no_load_velocity: float | dict[str, float] = 42.9350995991
    """Joint no-load speed X2 where motoring torque reaches zero [rad/s]."""
