"""Flat-only forward/reverse command expansion over the upstream sampler."""


def speed_limit(course, step, steps_per_iteration):
    progress = max(0., min(1., (step - course["start_step"]) / (2000 * steps_per_iteration)))
    return 1.5 + 0.5 * progress


def apply_flat_range(command):
    course = getattr(command._env.cfg, "flat_speed_course", None)
    if course is None:
        return
    limit = speed_limit(course, command._env.common_step_counter, command.cfg.num_steps_per_iter)
    mask = command.terrain_idxs == command.terrain_type2idx["flat"]
    command.env_command_ranges["lin_vel_x"][mask, 0] = -limit
    command.env_command_ranges["lin_vel_x"][mask, 1] = limit


def snapshot(command):
    course = getattr(command._env.cfg, "flat_speed_course", None)
    if course is None:
        return None
    mask = command.terrain_idxs == command.terrain_type2idx["flat"]
    ids = mask.nonzero().flatten()
    limit = speed_limit(course, command._env.common_step_counter, command.cfg.num_steps_per_iter)
    sampled = command.commands[ids]
    if len(ids) and bool((sampled[:, 0].abs() > limit + 1e-5).any()):
        raise RuntimeError("Flat forward command exceeded its scheduled range")
    return dict(**course, target_limit_mps=2., ramp_updates=2000, scheduled_limit_mps=limit,
                count=len(ids), effective_x_ranges=command.env_command_ranges["lin_vel_x"][ids].unique(dim=0).cpu().tolist(),
                sampled_min=sampled.amin(dim=0).cpu().tolist() if len(ids) else None,
                sampled_max=sampled.amax(dim=0).cpu().tolist() if len(ids) else None)
