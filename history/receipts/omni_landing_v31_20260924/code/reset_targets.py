"""Keep articulation targets continuous across Thunder environment resets.

The installer wraps one environment instance.  The original ``_reset_idx``
finishes first, then the robot's supported target setters copy the resulting
joint pose and clear velocity/effort targets.  This places the correction
before an explicit reset's subsequent ``scene.write_data_to_sim()`` without
writing simulator state or reaching into actuator delay buffers.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import torch

RESET_TARGETS_VERSION = "thunder_cts_v13_reset_targets_v1"
_INSTALLATION_ATTR = "_thunder_reset_targets_installation"


def synchronize_reset_targets(robot: Any, env_ids: Any) -> None:
    """Synchronize targets for the reset environments to their actual pose.

    ``env_ids=None`` selects every environment and is useful for callers that
    represent an explicit all-environment reset that way.
    """
    joint_pos = robot.data.joint_pos
    if env_ids is None:
        ids = torch.arange(joint_pos.shape[0], device=joint_pos.device, dtype=torch.long)
    else:
        ids = torch.as_tensor(env_ids, device=joint_pos.device, dtype=torch.long)
    if ids.ndim != 1:
        raise ValueError("Reset ids must be a one-dimensional sequence")
    if ids.numel() == 0:
        return

    position = joint_pos[ids].clone()
    zero = torch.zeros_like(position)
    robot.set_joint_position_target(position, env_ids=ids)
    robot.set_joint_velocity_target(zero, env_ids=ids)
    robot.set_joint_effort_target(zero, env_ids=ids)


def install_reset_target_sync(env: Any, robot_name: str = "robot") -> dict[str, str]:
    """Install reset-target synchronization on one environment instance.

    Returns a stable receipt suitable for runtime manifests.  Reinstalling on
    the same environment is idempotent and returns the existing receipt.
    """
    installed = getattr(env, _INSTALLATION_ATTR, None)
    if installed is not None:
        return dict(installed)

    robot = env.scene[robot_name]
    original_reset_idx = env._reset_idx

    @wraps(original_reset_idx)
    def reset_and_sync_targets(env_ids: Any) -> Any:
        result = original_reset_idx(env_ids)
        synchronize_reset_targets(robot, env_ids)
        return result

    receipt = {"version": RESET_TARGETS_VERSION, "robot": robot_name}
    env._reset_idx = reset_and_sync_targets
    setattr(env, _INSTALLATION_ATTR, receipt)
    return dict(receipt)
