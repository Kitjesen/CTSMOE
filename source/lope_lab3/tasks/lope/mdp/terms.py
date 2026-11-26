"""Lope locomotion helpers that the pinned Go2 module does not provide."""
import torch

from lope_lab3.tasks.lope.mdp.privileged import tensor


def feet_air_time(env, command_name, threshold, sensor_cfg):
    """Lope's robot_lab step reward, reading Lab 3 tensors explicitly.

    The historical function indexed ProxyArray-less buffers. This copy uses
    ``.torch`` and the same threshold, command gate, and upright scale.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    first = sensor.compute_first_contact(env.step_dt)
    first = tensor(first)[:, sensor_cfg.body_ids]
    last_air_time = tensor(sensor.data.last_air_time)[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first, dim=1)
    command = tensor(env.command_manager.get_command(command_name))
    reward = reward * (torch.linalg.vector_norm(command, dim=1) > 0.1)
    gravity = tensor(env.scene["robot"].data.projected_gravity_b)
    return reward * torch.clamp(-gravity[:, 2], 0, 0.7) / 0.7


def terrain_out_of_bounds(env, distance_buffer=3.0):
    generator = env.scene.terrain.cfg.terrain_generator
    position = tensor(env.scene["robot"].data.root_pos_w)[:, :2]
    extent = torch.tensor(
        [generator.num_rows * generator.size[0], generator.num_cols * generator.size[1]],
        device=env.device,
    ) / 2 + generator.border_width - distance_buffer
    return (position.abs() > extent).any(1)


def terrain_tile_exit(env):
    """End the episode at the assigned tile edge so the grade stays on one terrain."""
    size = torch.tensor(env.scene.terrain.cfg.terrain_generator.size, device=env.device)
    offset = tensor(env.scene["robot"].data.root_pos_w)[:, :2] - env.scene.env_origins[:, :2]
    return (offset.abs() > size / 2).any(1)
