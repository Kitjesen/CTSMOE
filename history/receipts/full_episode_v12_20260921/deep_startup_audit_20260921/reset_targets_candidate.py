"""Candidate integration seam for Isaac explicit-actuator reset target continuity.

Call after the environment's original _reset_idx and before write_data_to_sim.
GPU evidence in results_r2 demonstrates why: explicit reset previously filled
the delay buffer with zero absolute positions, although the calf reset pose was
+/-1.96 rad. This helper uses supported articulation target setters; it neither
changes simulator state nor edits delay-buffer internals. It is not installed
in the production training environment by importing this module.
"""
import torch


def synchronize_reset_targets(robot, env_ids):
    """Hold selected reset poses until their first policy target is processed."""
    ids = torch.as_tensor(env_ids, device=robot.data.joint_pos.device, dtype=torch.long)
    if ids.ndim != 1:
        raise ValueError("Reset ids must be a one-dimensional sequence")
    if ids.numel() == 0:
        return
    position = robot.data.joint_pos[ids].clone()
    zero = torch.zeros_like(position)
    robot.set_joint_position_target(position, env_ids=ids)
    robot.set_joint_velocity_target(zero, env_ids=ids)
    robot.set_joint_effort_target(zero, env_ids=ids)
