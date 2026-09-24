"""Explicit one-time low-ledge curriculum migration when first entering V8."""
import torch


def migrate(env):
    command = env.command_manager.get_term('base_velocity')
    terrain = env.scene.terrain
    ids = command.pit.active.nonzero().flatten()
    before = terrain.terrain_levels[ids].clone()
    # Distribute 15/20/25/30/35cm cases; retain independent per-speed promotion.
    levels = torch.arange(len(ids), device=env.device).remainder(5)
    terrain.terrain_levels[ids] = levels
    command.pit.individual.initialize_ids(ids, levels)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.scene.env_origins[:] = terrain.env_origins
    compute = env.curriculum_manager.compute
    env.curriculum_manager.compute = lambda env_ids: None
    try:
        env.reset()
    finally:
        env.curriculum_manager.compute = compute
    env._phase_step = -1
    return dict(pit_count=len(ids), previous_mean_level=float(before.float().mean()),
                initial_heights_m=[.15,.20,.25,.30,.35], max_height_m=1.2,
                promotion='existing independent course: two successes promote; two failures demote',
                other_terrain_levels='unchanged; stair riser heights retained, landing geometry changed')
