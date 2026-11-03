"""Keep sampled body-frame twists inside the real contiguous flat atlas band."""
import torch

MARGIN_M = 1.2


def paths(position, yaw, goals, duration=5.):
    """Constant body twist integrated in world coordinates (not body velocity as world)."""
    times = torch.linspace(0., duration, 51, device=goals.device)
    angle = yaw[:, None] + goals[:, 2:3]*times
    w = goals[:, 2:3]
    denom = torch.where(w.abs() > 1e-5, w, torch.ones_like(w))
    sine = (angle.sin()-yaw[:, None].sin())/denom
    cosine = (yaw[:, None].cos()-angle.cos())/denom
    sine = torch.where(w.abs() > 1e-5, sine, yaw[:, None].cos()*times)
    cosine = torch.where(w.abs() > 1e-5, cosine, yaw[:, None].sin()*times)
    dx = goals[:, 0:1]*sine - goals[:, 1:2]*cosine
    dy = goals[:, 0:1]*cosine + goals[:, 1:2]*sine
    return position[:, None, :] + torch.stack((dx, dy), -1)


def contain_goals(position, yaw, goals, lower, upper):
    """Prefer original command; reflect signed translation components when needed.

    The 5s duration and magnitudes remain unchanged. Reject infeasible geometry
    instead of silently returning a shorter episode or forced-stop command.
    """
    output = goals.clone()
    # Tracking error can place a robot inside the 1.2m planning margin while it
    # remains on flat ground. Permit recovery/standing paths that never move
    # farther outward; the separate 0.5m runtime boundary remains authoritative.
    allowed_lower = torch.minimum(lower.expand_as(position), position)
    allowed_upper = torch.maximum(upper.expand_as(position), position)
    chosen = torch.zeros(len(goals), device=goals.device, dtype=torch.bool)
    reflected = torch.zeros_like(chosen)
    for sx, sy in ((1., 1.), (-1., 1.), (1., -1.), (-1., -1.)):
        candidate = goals.clone()
        candidate[:, 0] *= sx
        candidate[:, 1] *= sy
        trajectory = paths(position, yaw, candidate)
        valid = ((trajectory >= allowed_lower[:, None, :]-1e-6)
                 & (trajectory <= allowed_upper[:, None, :]+1e-6)).all(dim=(1, 2))
        select = valid & ~chosen
        output[select] = candidate[select]
        reflected[select] = sx != 1. or sy != 1.
        chosen |= valid
    if not bool(chosen.all()):
        raise RuntimeError(f'{int((~chosen).sum())} flat commands have no feasible 5s path inside buffered flat band')
    return output, reflected


def bounds(command):
    if not hasattr(command, 'flat_band_bounds'):
        from robot_lab.tasks.go2.mdp.utils import _get_terrain_column_range
        terrain = command._env.scene.terrain
        cfg = terrain.cfg.terrain_generator
        first, last = _get_terrain_column_range(cfg, 'flat', command.device)
        origins = terrain.terrain_origins[:, first:last, :2]
        if last-first != 3:
            raise ValueError('Expected three contiguous flat columns in the pinned atlas')
        half = torch.tensor(cfg.size, device=command.device)/2
        physical_lower = origins.amin(dim=(0, 1))-half
        physical_upper = origins.amax(dim=(0, 1))+half
        command.flat_band_bounds = (physical_lower, physical_upper)
        command.flat_boundary_reflections = 0
        command.flat_boundary_crossings = 0
    return command.flat_band_bounds


def constrain(command, ids, goals):
    lower, upper = bounds(command)
    data = command.robot.data
    q = data.root_quat_w[ids]
    yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2].square()+q[:, 3].square()))
    result, changed = contain_goals(data.root_pos_w[ids, :2], yaw, goals,
                                    lower+MARGIN_M, upper-MARGIN_M)
    command.flat_boundary_reflections += int(changed.sum())
    return result


def check(command):
    lower, upper = bounds(command)
    flat = command.terrain_idxs == command.terrain_type2idx['flat']
    positions = command.robot.data.root_pos_w[flat, :2]
    outside = ((positions < lower+.5) | (positions > upper-.5)).any(-1)
    command.flat_boundary_crossings += int(outside.sum())
    if bool(outside.any()):
        raise RuntimeError('Flat robot left its assigned physical flat band; do not apply mislabeled rewards')


def snapshot(command):
    lower, upper = bounds(command)
    return dict(physical_lower_xy=lower.cpu().tolist(), physical_upper_xy=upper.cpu().tolist(),
                planning_margin_m=MARGIN_M, reflections=command.flat_boundary_reflections,
                crossings=command.flat_boundary_crossings, prediction='5s body twist integrated using measured yaw')
