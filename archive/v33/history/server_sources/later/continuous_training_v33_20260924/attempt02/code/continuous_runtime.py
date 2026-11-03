"""Runtime adapter for the six V33 continuous 9 m route families.

Install this last, after the existing mixed, variety, omni-route, and reset
wrappers.  Columns 0..19 retain their established geometry and behavior;
columns 20..25 use the continuous routes and columns 26..39 remain pit.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from functools import lru_cache

import numpy as np
import torch


PROFILE = "continuous_routes_v33"
FAMILIES = (
    "continuous_curbs",
    "continuous_blocks",
    "continuous_ramp_stairs",
    "continuous_split_height",
    "continuous_cross_bumps",
    "continuous_split_route",
)
COLUMNS = {20 + index: family for index, family in enumerate(FAMILIES)}
REQUIRED_METADATA = {"spawn", "final_obstacle_x", "finish_x", "exit_bound_x"}
TRACKING_LIMITS = (.30, .20, .20)
COMMAND_HOLD_S = 5.0


def continuous_mask(env):
    columns = env.scene.terrain.terrain_types
    return (columns >= 20) & (columns <= 25)


def _validated_geometry(family, level):
    from route_geometry import mesh_for

    result = mesh_for(family, int(level))
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("route_geometry.mesh_for must return exactly (mesh, metadata)")
    mesh, metadata = result
    if not isinstance(metadata, dict) or not REQUIRED_METADATA <= metadata.keys():
        missing = REQUIRED_METADATA - (metadata.keys() if isinstance(metadata, dict) else set())
        raise ValueError(f"Continuous route metadata is incomplete: {sorted(missing)}")
    spawn = np.asarray(metadata["spawn"], dtype=float)
    if spawn.shape != (3,) or not np.allclose(spawn, (-3.5, 0.0, 0.0), atol=1e-9, rtol=0):
        raise ValueError("Continuous route spawn must be exactly (-3.5, 0, 0)")
    values = [float(metadata[key]) for key in ("final_obstacle_x", "finish_x", "exit_bound_x")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Continuous route metadata must be finite")
    final_obstacle_x, finish_x, exit_bound_x = values
    if not final_obstacle_x <= finish_x < exit_bound_x <= 4.5:
        raise ValueError("Continuous route metadata must satisfy obstacle <= finish < exit <= 4.5")
    return mesh, dict(metadata)


@lru_cache(maxsize=60)
def metadata_for(family, level):
    _, metadata = _validated_geometry(family, level)
    return metadata


def geometry_source_sha256():
    import route_geometry

    source = inspect.getsourcefile(route_geometry)
    if source is None:
        raise ValueError("route_geometry must have an inspectable source file")
    with open(source, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def _continuous_route_terrain(family, difficulty, cfg):
    """Shared terrain-function implementation with family bound by the caller."""
    level = min(int(float(difficulty) * 10), 9)
    mesh, metadata = _validated_geometry(family, level)
    return [mesh], np.asarray(metadata["spawn"], dtype=float)


def continuous_curbs_terrain(difficulty, cfg):
    return _continuous_route_terrain("continuous_curbs", difficulty, cfg)


def continuous_blocks_terrain(difficulty, cfg):
    return _continuous_route_terrain("continuous_blocks", difficulty, cfg)


def continuous_ramp_stairs_terrain(difficulty, cfg):
    return _continuous_route_terrain("continuous_ramp_stairs", difficulty, cfg)


def continuous_split_height_terrain(difficulty, cfg):
    return _continuous_route_terrain("continuous_split_height", difficulty, cfg)


def continuous_cross_bumps_terrain(difficulty, cfg):
    return _continuous_route_terrain("continuous_cross_bumps", difficulty, cfg)


def continuous_split_route_terrain(difficulty, cfg):
    return _continuous_route_terrain("continuous_split_route", difficulty, cfg)


TERRAIN_FUNCTIONS = {
    "continuous_curbs": continuous_curbs_terrain,
    "continuous_blocks": continuous_blocks_terrain,
    "continuous_ramp_stairs": continuous_ramp_stairs_terrain,
    "continuous_split_height": continuous_split_height_terrain,
    "continuous_cross_bumps": continuous_cross_bumps_terrain,
    "continuous_split_route": continuous_split_route_terrain,
}


class ContinuousTerrainMixin:
    """Top-most generator hook: replace only V33 columns 20..25."""

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        if col in COLUMNS:
            level = min(int(row), 9)
            family = COLUMNS[col]
            mesh, metadata = _validated_geometry(family, level)
            origin = np.asarray(metadata["spawn"], dtype=float)
            if not hasattr(self, "continuous_route_receipts"):
                self.continuous_route_receipts = []
            self.continuous_route_receipts.append({
                "row": int(row), "column": int(col), "family": family,
                "level": level, "metadata": metadata,
            })
        result = super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        self.cfg.continuous_route_receipts = getattr(self, "continuous_route_receipts", [])
        return result


def tracking_is_good(error, limits=TRACKING_LIMITS):
    bounds = torch.as_tensor(limits, device=error.device, dtype=error.dtype)
    return (error <= bounds).all(-1)


def evaluate_route_logic(body_x_min, body_x_max, root_x, body_y_abs_max, final_obstacle_x,
                         finish_x, exit_bound_x, tracking_good):
    """Direct logical contract shared by control, curriculum, and unit tests."""
    inside_lane = body_y_abs_max < 4.45
    # Metadata's 4.4 m root guard leaves 10 cm nominal margin.  Apply that
    # margin to the leading body envelope so no body point crosses the 4.5 m
    # cell edge before the timeout is observed.
    forward_guard = torch.clamp(exit_bound_x + .05, max=4.45)
    boundary = ((body_x_max >= forward_guard) | (body_y_abs_max >= 4.45)
                | (body_x_min <= -4.45))
    success = ((body_x_min >= final_obstacle_x) & (root_x >= finish_x)
               & inside_lane & tracking_good & ~boundary)
    return success, boundary


def _route_pose(command, ids):
    env = command._env
    robot = command.robot
    if not hasattr(command, "pit") or not hasattr(command.pit, "geometry"):
        raise RuntimeError("Continuous routes require the established full-body geometry witness")
    hull = command.pit.geometry.world_points(
        robot.data.joint_pos[ids], robot.data.root_state_w[ids, :7])
    center = env.scene.env_origins[ids, :2].clone()
    center[:, 0] += 3.5
    delta = hull[..., :2] - center[:, None, :]
    root_x = robot.data.root_pos_w[ids, 0] - center[:, 0]
    return (delta[..., 0].amin(-1), delta[..., 0].amax(-1), root_x,
            delta[..., 1].abs().amax(-1))


class ContinuousCommandMixin:
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, device = self.num_envs, self.device
        self.continuous_speed = torch.zeros(n, device=device)
        self.continuous_tracking_sum = torch.zeros((n, 3), device=device)
        self.continuous_tracking_count = torch.zeros(n, device=device)
        self.continuous_success = torch.zeros(n, dtype=torch.bool, device=device)
        self.continuous_boundary = torch.zeros(n, dtype=torch.bool, device=device)
        self.continuous_final_obstacle_x = torch.zeros(n, device=device)
        self.continuous_finish_x = torch.zeros(n, device=device)
        self.continuous_exit_bound_x = torch.zeros(n, device=device)
        # family, level, [episodes, success, boundary, physical, promoted]
        self.continuous_totals = torch.zeros((6, 10, 5), dtype=torch.long, device=device)

    def _resample(self, env_ids):
        super()._resample(env_ids)
        if not hasattr(self, "continuous_speed"):
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ids = ids[continuous_mask(self._env)[ids]]
        if not len(ids):
            return
        self.continuous_speed[ids] = .4 + .5 * torch.rand(len(ids), device=self.device)
        # Commands resample every five seconds inside a 25 second episode.
        # Episode evidence must survive those resamples and clear only at reset.
        fresh = ids[self._env.episode_length_buf[ids] == 0]
        self.continuous_tracking_sum[fresh] = 0
        self.continuous_tracking_count[fresh] = 0
        self.continuous_success[fresh] = False
        self.continuous_boundary[fresh] = False
        terrain = self._env.scene.terrain
        for column, family in COLUMNS.items():
            for level in range(10):
                selected = ids[(terrain.terrain_types[ids] == column)
                               & (terrain.terrain_levels[ids] == level)]
                if not len(selected):
                    continue
                metadata = metadata_for(family, level)
                self.continuous_final_obstacle_x[selected] = float(metadata["final_obstacle_x"])
                self.continuous_finish_x[selected] = float(metadata["finish_x"])
                self.continuous_exit_bound_x[selected] = float(metadata["exit_bound_x"])
        self.time_left[ids] = COMMAND_HOLD_S
        self._set_continuous_commands(ids)

    def _set_continuous_commands(self, ids):
        self.commands[ids] = 0
        self.commands[ids, 0] = self.continuous_speed[ids]

    def _update_command(self):
        super()._update_command()
        if hasattr(self, "continuous_speed"):
            self._set_continuous_commands(continuous_mask(self._env).nonzero().flatten())

    def _update_metrics(self):
        super()._update_metrics()
        if not hasattr(self, "continuous_speed"):
            return
        ids = (continuous_mask(self._env) & (self._env.episode_length_buf > 0)).nonzero().flatten()
        if not len(ids):
            return
        actual = torch.cat((self.robot.data.root_lin_vel_b[ids, :2],
                            self.robot.data.root_ang_vel_b[ids, 2:3]), -1)
        self.continuous_tracking_sum[ids] += (actual - self.commands[ids]).abs()
        self.continuous_tracking_count[ids] += 1
        mean_error = self.continuous_tracking_sum[ids] / self.continuous_tracking_count[ids, None]
        body_x_min, body_x_max, root_x, body_y = _route_pose(self, ids)
        success, boundary = evaluate_route_logic(
            body_x_min, body_x_max, root_x, body_y, self.continuous_final_obstacle_x[ids],
            self.continuous_finish_x[ids], self.continuous_exit_bound_x[ids],
            tracking_is_good(mean_error))
        self.continuous_success[ids] |= success
        self.continuous_boundary[ids] |= boundary

    def snapshot(self):
        result = super().snapshot()
        result[PROFILE] = snapshot(self)
        return result


def reset_root(env, env_ids, pose_range, velocity_range, asset_cfg, continuous_original, **original_params):
    continuous_original(env, env_ids, pose_range, velocity_range, asset_cfg, **original_params)
    from robot_lab.tasks.go2.mdp.events import reset_root_state_uniform

    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[continuous_mask(env)[ids]]
    if len(ids):
        yaw = math.radians(5.0)
        reset_root_state_uniform(
            env, ids, dict(pose_range, x=(0., 0.), y=(0., 0.), yaw=(-yaw, yaw)),
            velocity_range, asset_cfg)


def route_done(env):
    command = env.command_manager.get_term("base_velocity")
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    selected = continuous_mask(env)
    result[selected] = command.continuous_success[selected] | command.continuous_boundary[selected]
    return result


def curriculum(env, env_ids, continuous_original, **original_params):
    """Delegate old columns and grade V33 rows without importer update_env_origins."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    selected_mask = continuous_mask(env)[ids]
    old_ids, route_ids = ids[~selected_mask], ids[selected_mask]
    old_result = None
    if len(old_ids):
        old_result = continuous_original(env, old_ids, **original_params)
    # Isaac computes curriculum once before the first physical episode.  That
    # call has no outcome to grade and may precede command-state initialization.
    route_ids = route_ids[env.episode_length_buf[route_ids] > 0]
    if not len(route_ids):
        return old_result

    command = env.command_manager.get_term("base_velocity")
    allocation = getattr(env, "terrain_allocation", None)
    if allocation is not None:
        allocation.record(route_ids)
    physical = env.termination_manager.terminated[route_ids]
    success = command.continuous_success[route_ids]
    boundary = command.continuous_boundary[route_ids]
    tracking = (command.continuous_tracking_sum[route_ids]
                / command.continuous_tracking_count[route_ids, None].clamp_min(1))
    promote = success & tracking_is_good(tracking) & ~boundary & ~physical
    downgrade = boundary | physical | ~promote
    terrain = env.scene.terrain
    old_levels = terrain.terrain_levels[route_ids]
    terrain.terrain_levels[route_ids] = (old_levels + promote.long() - downgrade.long()).clamp(0, 9)
    terrain.env_origins[route_ids] = terrain.terrain_origins[
        terrain.terrain_levels[route_ids], terrain.terrain_types[route_ids]]
    env.scene.env_origins[route_ids] = terrain.env_origins[route_ids]
    family = terrain.terrain_types[route_ids] - 20
    values = torch.stack((torch.ones_like(old_levels), success.long(), boundary.long(),
                          physical.long(), promote.long()), -1)
    flat_keys = family * 10 + old_levels.clamp(0, 9)
    command.continuous_totals.view(60, 5).index_add_(0, flat_keys, values)
    if allocation is not None:
        allocation.after_curriculum(route_ids)
    # Keep the inherited curriculum metric semantically old-terrain-only;
    # continuous rows have their own snapshot/statistics and must not shift it.
    old = terrain.terrain_types < 20
    return terrain.terrain_levels[old].float().mean()


def checkpoint_payload(command):
    return {
        "continuous_routes_profile": PROFILE,
        "continuous_geometry_sha256": geometry_source_sha256(),
        "continuous_totals": command.continuous_totals.detach().clone(),
    }


def restore(saved, command):
    profile = saved.get("continuous_routes_profile")
    if profile is None:
        return {"from": "legacy", "to": PROFILE, "continuous_totals": "initialized empty"}
    if profile != PROFILE:
        raise ValueError("Unknown continuous route checkpoint profile")
    expected_hash = geometry_source_sha256()
    initial_migration = saved.get("continuous_routes_initial_migration") is True
    saved_hash = saved.get("continuous_geometry_sha256")
    if saved_hash is None and not initial_migration:
        raise ValueError("Continuous route geometry hash is missing")
    if saved_hash is not None and saved_hash != expected_hash:
        raise ValueError("Continuous route geometry source changed across resume")
    totals = saved.get("continuous_totals")
    if totals is None and initial_migration:
        command.continuous_totals.zero_()
        return {"from": "initial migration", "to": PROFILE,
                "continuous_totals": "initialized empty"}
    if totals is None or tuple(totals.shape) != (6, 10, 5):
        raise ValueError("Continuous route cumulative statistics are missing or malformed")
    command.continuous_totals.copy_(totals.to(command.device))
    return {"from": PROFILE, "to": PROFILE, "continuous_totals": "restored exactly"}


def restore_checkpoint(saved, command):
    """Compatibility name for focused callers; train.py uses ``restore``."""
    return restore(saved, command)


def snapshot(env_or_command):
    command = (env_or_command.command_manager.get_term("base_velocity")
               if hasattr(env_or_command, "command_manager") else env_or_command)
    return {
        "profile": PROFILE,
        "families": list(FAMILIES),
        "columns": list(COLUMNS),
        "geometry_sha256": geometry_source_sha256(),
        "command": {"lin_vel_x": [.4, .9], "lin_vel_y": [0., 0.],
                    "ang_vel_yaw": [0., 0.], "hold_s": COMMAND_HOLD_S},
        "statistics": command.continuous_totals.cpu().tolist(),
        "statistic_axes": ["family", "level", "episodes_success_boundary_physical_promoted"],
        "success": "full body beyond final obstacle, finish progress, and command tracking",
    }


def configure(cfg):
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.terrains import SubTerrainBaseCfg

    terrain = cfg.scene.terrain.terrain_generator
    if terrain.num_cols != 40 or terrain.num_rows != 22 or tuple(terrain.size) != (9., 9.):
        raise ValueError("V33 requires the established 40x22 atlas of 9 m cells")
    if list(terrain.sub_terrains)[-1] != "pit" or terrain.sub_terrains["pit"].proportion != .5:
        raise ValueError("V33 requires the established final 0.5 pit allocation")
    legacy = [(name, id(sub), sub.proportion, sub.function)
              for name, sub in list(terrain.sub_terrains.items())[:-1]]
    pit = terrain.sub_terrains.pop("pit")
    pit.proportion = .35
    for family in FAMILIES:
        sub = SubTerrainBaseCfg(function=TERRAIN_FUNCTIONS[family], proportion=.025)
        terrain.sub_terrains[family] = sub
    terrain.sub_terrains["pit"] = pit
    after = [(name, id(sub), sub.proportion, sub.function)
             for name, sub in list(terrain.sub_terrains.items())[:len(legacy)]]
    if legacy != after:
        raise RuntimeError("V33 changed an established column 0..19 terrain configuration")
    if not math.isclose(sum(sub.proportion for sub in terrain.sub_terrains.values()), 1.0,
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("V33 terrain proportions do not cover exactly forty columns")

    parent_generator = terrain.class_type
    generator_type = type("ContinuousTerrainGenerator", (ContinuousTerrainMixin, parent_generator),
                          {"__module__": __name__})
    globals()[generator_type.__name__] = generator_type
    terrain.class_type = generator_type

    parent_command = cfg.commands.base_velocity.class_type
    command_type = type("ContinuousRouteCommand", (ContinuousCommandMixin, parent_command),
                        {"__module__": __name__})
    globals()[command_type.__name__] = command_type
    cfg.commands.base_velocity.class_type = command_type
    for family in FAMILIES:
        cfg.commands.base_velocity.terrain_max_command_ranges[family] = {
            "lin_vel_x": (.4, .9), "lin_vel_y": (0., 0.), "ang_vel_yaw": (0., 0.)}

    reset = cfg.events.reset_base
    reset.params = dict(reset.params or {})
    reset.params["continuous_original"] = reset.func
    reset.func = reset_root
    course = cfg.curriculum.terrain_levels
    course.params = dict(course.params or {})
    course.params["continuous_original"] = course.func
    course.func = curriculum
    if getattr(cfg.terminations, "continuous_route_done", None) is not None:
        raise ValueError("continuous route termination already configured")
    cfg.terminations.continuous_route_done = TerminationTermCfg(func=route_done, time_out=True)
    cfg.continuous_routes_profile = PROFILE
    terrain.continuous_routes_profile = PROFILE
    cfg.continuous_geometry_sha256 = geometry_source_sha256()
    return {"profile": PROFILE, "columns": COLUMNS, "geometry_sha256": cfg.continuous_geometry_sha256}
