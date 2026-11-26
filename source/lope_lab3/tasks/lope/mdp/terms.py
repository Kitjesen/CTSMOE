"""Lope reward and termination helpers that the pinned Go2 module does not provide."""
import math

import torch

from lope_lab3.recovery import FALLEN_MODES, fallen_orientation
from lope_lab3.spec import FALLEN_RESET_FRACTION, FALLEN_Z_DROP_M, ROLLOUT_STEPS
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


def _gravity_z(env):
    return tensor(env.scene["robot"].data.projected_gravity_b)[:, 2]


def _upright_score(gravity_z):
    return ((1.0 - gravity_z) * 0.5).clamp(0.0, 1.0)


def recovery_upright(env, saturation):
    """Continuous upright deficit cost; smoothly reaches 0 once standing."""
    score = _upright_score(_gravity_z(env))
    return (saturation - score).clamp(min=0.0)


def stand_up_bonus(env, saturation):
    """One-time bonus when an environment spawned fallen first reaches standing."""
    score = _upright_score(_gravity_z(env))
    standing = score >= saturation
    recovering_mask = getattr(env, "_lope_recovering_from_reset", None)
    if recovering_mask is None or tuple(recovering_mask.shape) != tuple(standing.shape):
        return torch.zeros(env.num_envs, device=standing.device)
    bonus = (recovering_mask & standing).float()
    recovering_mask[standing] = False
    return bonus


def undesired_contacts_locomotion(env, threshold, sensor_cfg):
    """Apply official undesired leg-contact penalty only after fallen-spawn recovery."""
    from go2_rl_lab.tasks.mixed.mdp.rewards import undesired_contacts

    cost = undesired_contacts(env, threshold, sensor_cfg)
    recovering_mask = getattr(env, "_lope_recovering_from_reset", None)
    if recovering_mask is None or recovering_mask.shape[0] != env.num_envs:
        return cost
    return torch.where(recovering_mask, torch.zeros_like(cost), cost)


def bad_orientation_locomotion(env, limit_angle):
    """Terminate severe inversion during locomotion without killing fallen-spawn recovery."""
    angle = torch.acos((-_gravity_z(env)).clamp(-1.0, 1.0))
    bad = angle > limit_angle
    recovering_mask = getattr(env, "_lope_recovering_from_reset", None)
    if recovering_mask is None or recovering_mask.shape[0] != env.num_envs:
        return bad
    return bad & ~recovering_mask


def fallen_timeout(env, max_time, upright_gravity_z, walk_max_time=None):
    """End the episode if the base stays down past the allowed recovery window."""
    down = _gravity_z(env) > upright_gravity_z
    timer = getattr(env, "_lope_fallen_timer", None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, device=env.device)
    timer = torch.where(down, timer + env.step_dt, torch.zeros_like(timer))
    env._lope_fallen_timer = timer
    if walk_max_time is None:
        return timer >= max_time
    recovering_mask = getattr(env, "_lope_recovering_from_reset", None)
    if recovering_mask is None or recovering_mask.shape[0] != env.num_envs:
        return timer >= walk_max_time
    limit = torch.where(
        recovering_mask,
        torch.full_like(timer, float(max_time)),
        torch.full_like(timer, float(walk_max_time)),
    )
    return timer >= limit


def fallen_reset_fraction(env, env_ids, initial, final, start_it, end_it):
    """Hold fallen resets at initial until start_it, then ramp linearly to final."""
    del env_ids
    if end_it <= start_it:
        raise ValueError(f"end_it must be greater than start_it, got {start_it=} and {end_it=}.")
    iteration = int(env.common_step_counter) // ROLLOUT_STEPS
    if iteration < start_it:
        progress = 0.0
    else:
        progress = min(1.0, (iteration - start_it) / float(end_it - start_it))
    fraction = initial + (final - initial) * progress
    env._lope_fallen_fraction = fraction
    return fraction


def reset_root_with_recovery(env, env_ids, pose_range, velocity_range, asset_cfg=None):
    """Normal standing reset, then a fallen orientation on a fraction of those resets."""
    from go2_rl_lab.tasks.mixed.mdp.events import reset_root_state_uniform
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_from_euler_xyz

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)
    if env_ids is None or isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids or slice(None)]
    else:
        ids = torch.as_tensor(env_ids, device=env.device)
        if ids.dtype == torch.bool:
            ids = ids.nonzero().flatten()
        else:
            ids = ids.long()
    timer = getattr(env, "_lope_fallen_timer", None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, device=env.device)
        env._lope_fallen_timer = timer
    timer[ids] = 0.0
    fraction = float(getattr(env, "_lope_fallen_fraction", FALLEN_RESET_FRACTION))
    chosen = ids[torch.rand(len(ids), device=env.device) < fraction]
    recovering_mask = getattr(env, "_lope_recovering_from_reset", None)
    if recovering_mask is None or recovering_mask.shape[0] != env.num_envs:
        recovering_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._lope_recovering_from_reset = recovering_mask
    recovering_mask[ids] = False
    recovering_mask[chosen] = True
    if len(chosen) == 0:
        return
    robot = env.scene[asset_cfg.name]
    pose = tensor(robot.data.root_pose_w)[chosen].clone()
    drop = torch.empty(len(chosen), device=env.device).uniform_(*FALLEN_Z_DROP_M)
    pose[:, 2] = (pose[:, 2] - drop).clamp_min(0.04)
    mode_index = torch.randint(0, len(FALLEN_MODES), (len(chosen),), device=env.device)
    jitter = torch.empty(len(chosen), 2, device=env.device).uniform_(-1.0, 1.0)
    yaw = torch.empty(len(chosen), device=env.device).uniform_(-math.pi, math.pi)
    rolls, pitches, yaws = [], [], []
    for index in range(len(chosen)):
        roll, pitch, yaw_value = fallen_orientation(
            FALLEN_MODES[int(mode_index[index])],
            float(jitter[index, 0]),
            float(jitter[index, 1]),
            float(yaw[index]),
        )
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw_value)
    pose[:, 3:7] = quat_from_euler_xyz(
        torch.tensor(rolls, device=env.device, dtype=pose.dtype),
        torch.tensor(pitches, device=env.device, dtype=pose.dtype),
        torch.tensor(yaws, device=env.device, dtype=pose.dtype),
    )
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=chosen)


def terrain_tile_exit(env):
    """End the episode at the assigned tile edge so the grade stays on one terrain."""
    size = torch.tensor(env.scene.terrain.cfg.terrain_generator.size, device=env.device)
    offset = tensor(env.scene["robot"].data.root_pos_w)[:, :2] - env.scene.env_origins[:, :2]
    return (offset.abs() > size / 2).any(1)
