"""Keep flat commands and the robot inside the real contiguous flat atlas band."""
from __future__ import annotations

import torch


MARGIN_M = 1.2
GUARD_MARGIN_M = 1.0
RECOVERY_SPEED_MPS = 0.2


def paths(position, yaw, goals, duration=5.0):
    """Integrate a constant body-frame twist in world coordinates."""
    times = torch.linspace(0.0, duration, 51, device=goals.device)
    angle = yaw[:, None] + goals[:, 2:3] * times
    w = goals[:, 2:3]
    denom = torch.where(w.abs() > 1e-5, w, torch.ones_like(w))
    sine = (angle.sin() - yaw[:, None].sin()) / denom
    cosine = (yaw[:, None].cos() - angle.cos()) / denom
    sine = torch.where(w.abs() > 1e-5, sine, yaw[:, None].cos() * times)
    cosine = torch.where(w.abs() > 1e-5, cosine, yaw[:, None].sin() * times)
    dx = goals[:, 0:1] * sine - goals[:, 1:2] * cosine
    dy = goals[:, 0:1] * cosine + goals[:, 1:2] * sine
    return position[:, None, :] + torch.stack((dx, dy), -1)


def _allowed_bounds(position, lower, upper):
    """Allow a robot already in the planning margin to stand or move inward."""
    return (
        torch.minimum(lower.expand_as(position), position),
        torch.maximum(upper.expand_as(position), position),
    )


def _valid_paths(trajectory, allowed_lower, allowed_upper):
    return (
        (trajectory >= allowed_lower[:, None, :] - 1e-6)
        & (trajectory <= allowed_upper[:, None, :] + 1e-6)
    ).all(dim=(1, 2))


def reflect_goals(position, yaw, goals, lower, upper):
    """Try all translation and yaw reflections without changing magnitudes."""
    output = goals.clone()
    allowed_lower, allowed_upper = _allowed_bounds(position, lower, upper)
    chosen = torch.zeros(len(goals), device=goals.device, dtype=torch.bool)
    reflected = torch.zeros_like(chosen)
    for sx, sy, sw in (
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0),
        (-1.0, -1.0, -1.0),
    ):
        candidate = goals.clone()
        candidate[:, 0] *= sx
        candidate[:, 1] *= sy
        candidate[:, 2] *= sw
        valid = _valid_paths(paths(position, yaw, candidate), allowed_lower, allowed_upper)
        select = valid & ~chosen
        output[select] = candidate[select]
        reflected[select] = sx != 1.0 or sy != 1.0 or sw != 1.0
        chosen |= valid
    return output, reflected, chosen


def contain_goals(position, yaw, goals, lower, upper):
    """Strict reflection helper retained for focused tests and diagnostics."""
    output, reflected, chosen = reflect_goals(position, yaw, goals, lower, upper)
    if not bool(chosen.all()):
        raise RuntimeError(
            f"{int((~chosen).sum())} flat commands have no feasible reflected 5s path"
        )
    return output, reflected


def cardinal_recovery(position, yaw, lower, upper):
    """Choose a legal minimum-speed 5 s cardinal path with best end clearance."""
    count = len(position)
    base = torch.tensor(
        (
            (RECOVERY_SPEED_MPS, 0.0, 0.0),
            (-RECOVERY_SPEED_MPS, 0.0, 0.0),
            (0.0, RECOVERY_SPEED_MPS, 0.0),
            (0.0, -RECOVERY_SPEED_MPS, 0.0),
        ),
        device=position.device,
        dtype=position.dtype,
    )
    candidates = base[None].expand(count, -1, -1)
    flat_candidates = candidates.reshape(-1, 3)
    repeated_position = position[:, None].expand(-1, 4, -1).reshape(-1, 2)
    repeated_yaw = yaw[:, None].expand(-1, 4).reshape(-1)
    trajectory = paths(repeated_position, repeated_yaw, flat_candidates).reshape(count, 4, 51, 2)
    allowed_lower, allowed_upper = _allowed_bounds(position, lower, upper)
    valid = (
        (trajectory >= allowed_lower[:, None, None, :] - 1e-6)
        & (trajectory <= allowed_upper[:, None, None, :] + 1e-6)
    ).all(dim=(2, 3))
    endpoint = trajectory[:, :, -1]
    end_clearance = torch.minimum(endpoint - lower, upper - endpoint).amin(-1)
    score = torch.where(valid, end_clearance, torch.full_like(end_clearance, -torch.inf))
    best_score, best = score.max(-1)
    if not bool(torch.isfinite(best_score).all()):
        raise RuntimeError("No legal minimum-speed cardinal recovery path")
    goals = candidates[torch.arange(count, device=position.device), best]
    return goals, best


def _ensure_state(command):
    count = command.num_envs
    device = command.device
    scalar_defaults = {
        "flat_boundary_reflections": 0,
        "flat_boundary_resamples": 0,
        "flat_boundary_cardinal_recoveries": 0,
        "flat_boundary_guard_timeouts": 0,
        "flat_boundary_crossings": 0,
    }
    for name, value in scalar_defaults.items():
        if not hasattr(command, name):
            setattr(command, name, value)
    if not hasattr(command, "flat_boundary_guard_latched"):
        command.flat_boundary_guard_latched = torch.zeros(count, device=device, dtype=torch.bool)
    if not hasattr(command, "flat_boundary_timeout_mask"):
        command.flat_boundary_timeout_mask = torch.zeros(count, device=device, dtype=torch.bool)


def bounds(command):
    _ensure_state(command)
    if not hasattr(command, "flat_band_bounds"):
        from robot_lab.tasks.go2.mdp.utils import _get_terrain_column_range

        terrain = command._env.scene.terrain
        cfg = terrain.cfg.terrain_generator
        first, last = _get_terrain_column_range(cfg, "flat", command.device)
        origins = terrain.terrain_origins[:, first:last, :2]
        if last - first != 3:
            raise ValueError("Expected three contiguous flat columns in the pinned atlas")
        half = torch.tensor(cfg.size, device=command.device) / 2
        physical_lower = origins.amin(dim=(0, 1)) - half
        physical_upper = origins.amax(dim=(0, 1)) + half
        command.flat_band_bounds = (physical_lower, physical_upper)
    return command.flat_band_bounds


def _pose(command, ids):
    data = command.robot.data
    q = data.root_quat_w[ids]
    yaw = torch.atan2(
        2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1 - 2 * (q[:, 2].square() + q[:, 3].square()),
    )
    return data.root_pos_w[ids, :2], yaw


def constrain(command, ids, goals):
    """Return reflected goals and a resolved mask; callers may resample failures."""
    lower, upper = bounds(command)
    position, yaw = _pose(command, ids)
    result, changed, resolved = reflect_goals(
        position, yaw, goals, lower + MARGIN_M, upper - MARGIN_M
    )
    command.flat_boundary_reflections += int((changed & resolved).sum())
    return result, resolved


def recover(command, ids):
    lower, upper = bounds(command)
    position, yaw = _pose(command, ids)
    goals, kinds = cardinal_recovery(position, yaw, lower + MARGIN_M, upper - MARGIN_M)
    command.flat_boundary_cardinal_recoveries += len(ids)
    return goals, kinds


def clear_guard(command, ids):
    _ensure_state(command)
    command.flat_boundary_guard_latched[ids] = False
    command.flat_boundary_timeout_mask[ids] = False


def _flat_mask(command):
    return command.terrain_idxs == command.terrain_type2idx["flat"]


def _physical_outside(command):
    lower, upper = bounds(command)
    flat = _flat_mask(command)
    result = torch.zeros(command.num_envs, device=command.device, dtype=torch.bool)
    position = command.robot.data.root_pos_w[flat, :2]
    result[flat] = ((position < lower) | (position > upper)).any(-1)
    return result


def flat_boundary_timeout(env):
    """End only an affected flat episode before it can enter a neighboring tile."""
    command = env.command_manager.get_term("base_velocity")
    lower, upper = bounds(command)
    physical_outside = _physical_outside(command)
    if bool(physical_outside.any()):
        command.flat_boundary_crossings += int(physical_outside.sum())
        raise RuntimeError("Flat robot crossed the true physical flat-band boundary")
    flat = _flat_mask(command)
    guard = torch.zeros(command.num_envs, device=command.device, dtype=torch.bool)
    position = command.robot.data.root_pos_w[flat, :2]
    guard[flat] = (
        (position < lower + GUARD_MARGIN_M) | (position > upper - GUARD_MARGIN_M)
    ).any(-1)
    new_guard = guard & ~command.flat_boundary_guard_latched
    command.flat_boundary_guard_timeouts += int(new_guard.sum())
    command.flat_boundary_guard_latched |= guard
    command.flat_boundary_timeout_mask.copy_(guard)
    return guard


def boundary_safe_curriculum(env, env_ids, original):
    """Run normal bookkeeping, then undo guard-only row changes when type stayed fixed."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    command = env.command_manager.get_term("base_velocity")
    # Isaac's first reset computes curricula before command.reset/terminations.
    # Initialize the zero guard state here so that first reset is a normal case.
    _ensure_state(command)
    # A simultaneous physical termination remains a real curriculum outcome; only
    # a boundary timeout by itself preserves the row.
    physical = env.termination_manager.terminated[ids]
    guarded = ids[command.flat_boundary_timeout_mask[ids] & ~physical]
    terrain = env.scene.terrain
    saved_levels = terrain.terrain_levels[guarded].clone()
    saved_types = terrain.terrain_types[guarded].clone()
    result = original(env, ids)
    if len(guarded):
        unchanged = terrain.terrain_types[guarded] == saved_types
        restore = guarded[unchanged]
        terrain.terrain_levels[restore] = saved_levels[unchanged]
        terrain.env_origins[restore] = terrain.terrain_origins[
            terrain.terrain_levels[restore], terrain.terrain_types[restore]
        ]
        env.scene.env_origins[restore] = terrain.env_origins[restore]
        # The pinned original is mixed_curriculum, whose metric is the non-pit
        # mean level. Recompute after restoration so logs match live state.
        return terrain.terrain_levels[~command.pit.active].float().mean()
    return result


def check(command):
    """Backstop only true physical crossings; the guard itself is an env timeout."""
    outside = _physical_outside(command)
    if bool(outside.any()):
        command.flat_boundary_crossings += int(outside.sum())
        raise RuntimeError("Flat robot crossed the true physical flat-band boundary")


def configure(cfg):
    from isaaclab.managers import TerminationTermCfg

    if getattr(cfg.terminations, "flat_boundary_timeout", None) is not None:
        raise ValueError("flat boundary timeout already configured")
    cfg.terminations.flat_boundary_timeout = TerminationTermCfg(
        func=flat_boundary_timeout, time_out=True
    )
    term = cfg.curriculum.terrain_levels
    original = term.func
    params = dict(term.params or {})
    if params:
        raise ValueError("V19 flat boundary expects an unparameterized terrain curriculum")
    term.func = boundary_safe_curriculum
    term.params = {"original": original}


def snapshot(command):
    lower, upper = bounds(command)
    return {
        "physical_lower_xy": lower.cpu().tolist(),
        "physical_upper_xy": upper.cpu().tolist(),
        "planning_margin_m": MARGIN_M,
        "guard_root_margin_m": GUARD_MARGIN_M,
        "reflections": command.flat_boundary_reflections,
        "resamples": command.flat_boundary_resamples,
        "cardinal_recoveries": command.flat_boundary_cardinal_recoveries,
        "guard_timeouts": command.flat_boundary_guard_timeouts,
        "crossings": command.flat_boundary_crossings,
        "prediction": "5s body twist integrated using measured yaw",
    }
