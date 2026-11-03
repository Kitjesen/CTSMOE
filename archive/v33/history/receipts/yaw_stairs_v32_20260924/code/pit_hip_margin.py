"""Charge for time spent near the inward hip hard stop on the pit task."""

import torch

from hip_contract import INWARD_SIGNS


START_INWARD_RAD = 0.34
HARD_MARGIN_RAD = 0.07
WEIGHT = -2.0


def margin_cost(actual_hip_radians, inward_signs):
    inward = actual_hip_radians * inward_signs
    excess = ((inward - START_INWARD_RAD) / HARD_MARGIN_RAD).clamp_min(0.)
    return excess.square().sum(-1)


def reward(env):
    command = env.command_manager.get_term("base_velocity")
    if not hasattr(command, "pit"):
        raise TypeError("Hip-margin cost requires the reviewed mixed pit task")
    robot = env.scene["robot"]
    if not hasattr(env, "_pit_hip_margin_ids"):
        env._pit_hip_margin_ids = [robot.joint_names.index(name) for name in INWARD_SIGNS]
        env._pit_hip_margin_signs = torch.tensor(
            list(INWARD_SIGNS.values()), device=env.device)
    cost = margin_cost(robot.data.joint_pos[:, env._pit_hip_margin_ids],
                       env._pit_hip_margin_signs)
    return torch.where(command.pit.active, cost, torch.zeros_like(cost))
