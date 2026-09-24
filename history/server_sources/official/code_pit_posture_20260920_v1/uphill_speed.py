"""Advance only the upstream uphill vx range, retaining its sampling branches."""


def apply_uphill_range(command):
    requested = getattr(command._env.cfg, "uphill_speed_limit", None)
    if requested is None:
        return
    if requested not in (1.2, 1.5):
        raise ValueError("Uphill speed limit must be 1.2 or 1.5 m/s")
    cap = command.cfg.terrain_max_command_ranges["slope_up"]["lin_vel_x"]
    mask = command.terrain_idxs == command.terrain_type2idx["slope_up"]
    # Later official stages may expand farther; never regress their range.
    command.env_command_ranges["lin_vel_x"][mask, 0] = max(cap[0], min(command.command_ranges["lin_vel_x"][0], -requested))
    command.env_command_ranges["lin_vel_x"][mask, 1] = min(cap[1], max(command.command_ranges["lin_vel_x"][1], requested))


def snapshot(command):
    requested = getattr(command._env.cfg, "uphill_speed_limit", None)
    if requested is None:
        return None
    mask = command.terrain_idxs == command.terrain_type2idx["slope_up"]
    sampled = command.commands[mask]
    ranges = command.env_command_ranges["lin_vel_x"][mask]
    if len(sampled) and bool(((sampled[:,0] < ranges[:,0]-1e-5) | (sampled[:,0] > ranges[:,1]+1e-5)).any()):
        raise RuntimeError("Uphill vx command outside effective range")
    return dict(requested_limit_mps=requested, count=int(mask.sum()),
                effective_x_ranges=ranges.unique(dim=0).cpu().tolist(),
                sampled_min=sampled.amin(0).cpu().tolist() if len(sampled) else None,
                sampled_max=sampled.amax(0).cpu().tolist() if len(sampled) else None)
