"""Opt-in Isaac adapter for the 9x9 slope-to-stair route.

No change is made to actor observations or low-level actuator limits.  This
adapter uses the already-present filtered wheel/ground sensors and the same
collision-hull sampler as the pit task.  It remains a candidate until a live
zero-update Isaac preflight validates the sensor timing and contact geometry.
"""

from __future__ import annotations

import math

import torch

from composite_course_state import CompositeCourseState
from composite_route import COMPACT_9X9, GEOMETRY_LEVELS

# Thunder's foot-link collision cylinders and feet_regulation use this radius.
# A wheel centre only witnesses a support surface when the *bottom* of that
# cylinder matches the route height and its footprint is clear of an edge.
WHEEL_RADIUS_M = 0.093
SURFACE_INSET_M = WHEEL_RADIUS_M + 0.01
HEIGHT_TOLERANCE_M = 0.03
MIN_VERTICAL_SUPPORT_N = 5.0


def terrain_wheel_support(wheel_tile, forces_w, levels):
    """Conservative support witness for [N,4,3] tile-local wheel centres.

    forces_w must be the world-frame force exerted by the filtered terrain
    mesh on each foot link, not a net force over all collision partners. A
    large horizontal riser reaction is not upward support.
    """
    if (wheel_tile.ndim != 3 or wheel_tile.shape[1:] != (4, 3)
            or forces_w.shape != wheel_tile.shape or levels.shape != (len(wheel_tile),)):
        raise ValueError("Expected [N,4,3] wheel centres/terrain forces and [N] levels")
    x, y, wheel_bottom = wheel_tile.unbind(-1)
    wheel_bottom = wheel_bottom - WHEEL_RADIUS_M
    slope_top = ((COMPACT_9X9.slope_end_x_m - COMPACT_9X9.approach_end_x_m)
                 * torch.tan((10.0 + 10.0 * levels / 9.0)[:, None] * (math.pi / 180.0)))
    step_height = (0.08 + 0.04 * levels / 9.0)[:, None]
    expected_height = torch.zeros_like(x)
    inside = torch.zeros_like(x, dtype=torch.bool)
    segments = [
        (0.0, COMPACT_9X9.approach_end_x_m, 0.0),
        (COMPACT_9X9.approach_end_x_m, COMPACT_9X9.slope_end_x_m,
         slope_top * (x - COMPACT_9X9.approach_end_x_m)
         / (COMPACT_9X9.slope_end_x_m - COMPACT_9X9.approach_end_x_m)),
        (COMPACT_9X9.slope_end_x_m, COMPACT_9X9.transition_end_x_m, slope_top),
    ]
    segments.extend((COMPACT_9X9.transition_end_x_m + i * COMPACT_9X9.stair_tread_m,
                     COMPACT_9X9.transition_end_x_m + (i + 1) * COMPACT_9X9.stair_tread_m,
                     slope_top + (i + 1) * step_height)
                    for i in range(COMPACT_9X9.stair_count))
    segments.append((COMPACT_9X9.final_start_x_m, COMPACT_9X9.size_x_m,
                     slope_top + COMPACT_9X9.stair_count * step_height))
    for start, end, height in segments:
        segment = (x >= start + SURFACE_INSET_M) & (x <= end - SURFACE_INSET_M)
        inside |= segment
        expected_height = torch.where(segment, height, expected_height)
    inside &= ((y >= COMPACT_9X9.tile_border_m + SURFACE_INSET_M)
               & (y <= COMPACT_9X9.size_y_m - COMPACT_9X9.tile_border_m
                  - SURFACE_INSET_M))
    return (inside & torch.isfinite(wheel_tile).all(dim=-1)
            & torch.isfinite(forces_w).all(dim=-1)
            & (forces_w[..., 2] > MIN_VERTICAL_SUPPORT_N)
            & ((wheel_bottom - expected_height).abs() <= HEIGHT_TOLERANCE_M))


def terrain_contact_forces(ground_contacts, ids, num_envs):
    """Isaac ContactSensor force_matrix_w contract: four [E,1,1,3] tensors.

    Each sensor has one foot-link body and exactly one filtered terrain mesh;
    its final dimension is the world-frame reaction on that foot link.
    """
    if len(ground_contacts) != 4:
        raise ValueError("Composite route requires exactly four terrain-filtered foot sensors")
    matrices = [sensor.data.force_matrix_w for sensor in ground_contacts]
    if any(matrix is None or matrix.shape != (num_envs, 1, 1, 3)
           for matrix in matrices):
        raise ValueError("Expected one terrain-filtered [E,1,1,3] force_matrix_w per wheel")
    return torch.stack([matrix[ids, 0, 0] for matrix in matrices], dim=1)


class CompositeRuntime:
    def __init__(self, env, command):
        self.env, self.command = env, command
        self.device = env.device
        self.course = CompositeCourseState(env.num_envs, device=env.device)
        self.held_speed = torch.zeros(env.num_envs, device=env.device)
        self.speed_bin = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.needs_speed = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.stop_on_platform = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.outside = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.physical = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.success_streak = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.failure_streak = torch.zeros_like(self.success_streak)
        # Source-robot collision hull, not root position, sets the minimum
        # requested distance for a time-feasible full-route plus 2s dwell.
        pose = torch.zeros(1, 7, device=env.device)
        pose[:, 3] = 1.
        wheel_points = command.pit.geometry.wheel_points
        if not bool(wheel_points.any()) or not bool((~wheel_points).any()):
            raise ValueError("Composite collision sampler must include wheel and body surfaces")
        default_hull = command.pit.geometry.world_points(
            command.robot.data.default_joint_pos[:1], pose)
        if wheel_points.shape != (default_hull.shape[1],):
            raise ValueError("Composite wheel/body flags do not cover the collision samples")
        rear_extent = max(0., -float(default_hull[..., 0].amin()))
        self.required_progress_m = (COMPACT_9X9.final_start_x_m
                                    + self.course.acceptance.edge_margin_m + rear_extent
                                    - COMPACT_9X9.spawn_x_m)
        if not 4.5 < self.required_progress_m < 7.0:
            raise ValueError("Unexpected default Thunder collision hull rear extent")
        remaining_s = env.max_episode_length * env.step_dt - self.course.acceptance.dwell_required_s
        if remaining_s <= 0:
            raise ValueError("Composite episode leaves no time for the required dwell")
        # 0.2 m/s cannot traverse the full route within this episode. Keep it
        # for explicit diagnostics, not stochastic training samples.
        self.minimum_training_speed_mps = max(0.2, self.required_progress_m * 1.25 / remaining_s + 0.01)
        if self.minimum_training_speed_mps >= 0.4:
            raise ValueError("Composite first speed bin has no time-feasible range")
        # learner, level: episodes, completed, graded failure, up, down,
        # time-ineligible timeout (recorded but cannot lower the curriculum).
        self.level_totals = torch.zeros((2, GEOMETRY_LEVELS, 6), device=env.device,
                                        dtype=torch.float64)
        self.last_refresh = -1
        self.refresh_membership()

    @property
    def active(self):
        return self.course.active

    @property
    def success(self):
        return self.course.success

    def refresh_membership(self):
        route_type = self.command.terrain_type2idx["composite"]
        next_active = self.command.terrain_idxs == route_type
        changed = self.course.active ^ next_active
        self.course.set_active(next_active)
        self.success_streak[changed] = 0
        self.failure_streak[changed] = 0
        self.last_refresh = -1

    def reset(self, ids):
        self.course.reset(ids)
        self.needs_speed[ids] = True
        self.speed_bin[ids] = -1
        self.stop_on_platform[ids] = False
        self.outside[ids] = False
        self.physical[ids] = False
        self.last_refresh = -1

    def resample(self, ids):
        fresh = ids[self.needs_speed[ids]]
        bins = torch.randint(4, (len(fresh),), device=self.device)
        edges = torch.tensor((self.minimum_training_speed_mps, 0.4, 0.6, 0.8, 1.0),
                             device=self.device)
        speeds = edges[bins] + torch.rand(len(fresh), device=self.device) * (edges[bins + 1] - edges[bins])
        self.speed_bin[fresh], self.held_speed[fresh] = bins, speeds
        self.needs_speed[fresh] = False
        self.command.commands[ids] = 0
        self.command.commands[ids, 0] = self.held_speed[ids]
        self.command.last_is_limit_vel[ids] = False
        self.command.time_left[ids] = 1.0e9

    def refresh(self):
        env = self.env
        if self.last_refresh == env.common_step_counter:
            return
        self.last_refresh = env.common_step_counter
        ids = self.active.nonzero().flatten()
        if not len(ids):
            return
        pit = self.command.pit
        data = self.command.robot.data
        origins = env.scene.env_origins[ids]
        wheel_world = data.body_pos_w[ids][:, pit.wheel_ids]
        wheel_tile = wheel_world - origins[:, None]
        wheel_tile[..., :2] += torch.tensor(
            (COMPACT_9X9.spawn_x_m, COMPACT_9X9.size_y_m / 2), device=self.device)
        wheel_xy = wheel_tile[..., :2]
        forces = terrain_contact_forces(pit.ground_contacts, ids, env.num_envs)
        wheel_support = terrain_wheel_support(
            wheel_tile, forces, env.scene.terrain.terrain_levels[ids])
        supported_x = wheel_xy[..., 0]
        self.course.seen_slope[ids] |= (
            wheel_support & (supported_x >= COMPACT_9X9.approach_end_x_m)
            & (supported_x < COMPACT_9X9.slope_end_x_m)).any(dim=1)
        self.course.seen_stair[ids] |= (
            wheel_support & (supported_x >= COMPACT_9X9.transition_end_x_m)
            & (supported_x < COMPACT_9X9.final_start_x_m)).any(dim=1)
        base_contact = pit.contact.data.net_forces_w_history[ids][:, :, pit.base_contact_ids]
        body_contact = torch.linalg.vector_norm(base_contact, dim=-1).amax(dim=(1, 2)) > 1.0
        self.physical[ids] |= env.actuator_guard.violation[ids] | body_contact
        points_world = pit.geometry.world_points(
            data.joint_pos[ids], data.root_state_w[ids, :7])
        hull_xy = points_world[..., :2] - origins[:, None, :2]
        hull_xy += torch.tensor(
            (COMPACT_9X9.spawn_x_m, COMPACT_9X9.size_y_m / 2), device=self.device)
        # world_points includes every sampled articulated collision surface,
        # including all four wheel cylinders. A root inside the tile is not
        # enough when the first wheel/corner has already crossed its border.
        self.outside[ids] |= (
            ~torch.isfinite(hull_xy).all(dim=(1, 2))
            | (hull_xy[..., 0].amin(dim=1) < COMPACT_9X9.tile_border_m)
            | (hull_xy[..., 0].amax(dim=1) > COMPACT_9X9.size_x_m - COMPACT_9X9.tile_border_m)
            | (hull_xy[..., 1].amin(dim=1) < COMPACT_9X9.tile_border_m)
            | (hull_xy[..., 1].amax(dim=1) > COMPACT_9X9.size_y_m - COMPACT_9X9.tile_border_m))
        terminal = wheel_xy[..., 0].amin(dim=1) >= COMPACT_9X9.final_start_x_m
        candidates = ids[terminal]
        pending = ids[~terminal]
        self.course.dwell_s[pending] = 0
        self.course.full_body_platform[pending] = False
        if not len(candidates):
            return
        local = terminal.nonzero().flatten()
        tilt = torch.acos((-data.projected_gravity_b[candidates, 2]).clamp(-1., 1.))
        self.course.update_subset(
            candidates, wheel_xy=wheel_xy[local], body_hull_xy=hull_xy[local],
            wheel_support=wheel_support[local],
            translation_speed_mps=data.root_lin_vel_w[candidates, :2].norm(dim=1),
            yaw_rate_rps=data.root_ang_vel_w[candidates, 2], body_tilt_rad=tilt,
            physical_failure=self.physical[candidates] | self.outside[candidates],
            dt_s=env.step_dt,
        )
        self.stop_on_platform[candidates] |= (
            self.course.full_body_platform[candidates]
            & self.course.seen_slope[candidates] & self.course.seen_stair[candidates]
            & ~self.physical[candidates] & ~self.outside[candidates])

    def update_command(self):
        self.refresh()
        ids = self.active.nonzero().flatten()
        self.command.commands[ids] = 0
        self.command.commands[ids, 0] = torch.where(
            self.stop_on_platform[ids], 0.0, self.held_speed[ids])

    def record_episodes(self, ids):
        ids = ids[(self.active[ids]) & (self.env.episode_length_buf[ids] > 0)]
        if not len(ids):
            return
        self.refresh()
        completed = self.success[ids] & ~self.physical[ids] & ~self.outside[ids]
        # Terminal physics includes the existing illegal-contact and actuator
        # terms. Timeout and leaving the tile are distinct from success.
        terminated_without_success = self.env.termination_manager.terminated[ids] & ~completed
        timeout = self.env.episode_length_buf[ids] >= self.env.max_episode_length - 1
        remaining_s = max(0., self.env.max_episode_length * self.env.step_dt
                          - self.course.acceptance.dwell_required_s)
        eligible = self.held_speed[ids] * remaining_s >= self.required_progress_m * 1.25
        ineligible_timeout = timeout & ~eligible & ~completed
        failed = (self.physical[ids] | self.outside[ids] | terminated_without_success
                  | (timeout & eligible)) & ~completed
        all_failed = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        all_failed[ids] = failed | ineligible_timeout
        self.course.record_episodes(ids, self.command.learner, all_failed)
        terrain = self.env.scene.terrain
        prior = terrain.terrain_levels[ids].clone()
        if bool(((prior < 0) | (prior >= GEOMETRY_LEVELS)).any()):
            raise ValueError("Route curriculum received a level outside 0..9")
        self.success_streak[ids] = torch.where(completed,
                                                self.success_streak[ids] + 1, 0)
        self.failure_streak[ids] = torch.where(failed,
                                                self.failure_streak[ids] + 1, 0)
        success_gate = self.success_streak[ids] >= 2
        failure_gate = self.failure_streak[ids] >= 2
        upgrade = success_gate & (prior < GEOMETRY_LEVELS - 1)
        downgrade = failure_gate & (prior > 0)
        next_level = prior + upgrade.long() - downgrade.long()
        self.success_streak[ids[success_gate]] = 0
        self.failure_streak[ids[failure_gate]] = 0
        terrain.terrain_levels[ids] = next_level
        key = self.command.learner[ids].long() * GEOMETRY_LEVELS + prior
        values = torch.stack((torch.ones_like(completed), completed, failed,
                              upgrade, downgrade, ineligible_timeout), dim=-1).to(torch.float64)
        self.level_totals.view(-1, 6).index_add_(0, key, values)
        terrain.env_origins[ids] = terrain.terrain_origins[next_level, terrain.terrain_types[ids]]
        self.env.scene.env_origins[ids] = terrain.env_origins[ids]

    def state_dict(self):
        return dict(course=self.course.state_dict(),
                    held_speed=self.held_speed.clone(), speed_bin=self.speed_bin.clone(),
                    stop_on_platform=self.stop_on_platform.clone(),
                    required_progress_m=self.required_progress_m,
                    success_streak=self.success_streak.clone(),
                    failure_streak=self.failure_streak.clone(),
                    level_totals=self.level_totals.clone())

    def snapshot(self):
        return dict(profile="composite_flat35_v25", count=int(self.active.sum()),
                    seen_slope=int((self.course.seen_slope & self.active).sum()),
                    seen_stair=int((self.course.seen_stair & self.active).sum()),
                    full_body_platform=int((self.course.full_body_platform & self.active).sum()),
                    stop_command=int((self.stop_on_platform & self.active).sum()),
                    continuous_two_second_success=int((self.success & self.active).sum()),
                    physical_failure=int((self.physical & self.active).sum()),
                    outside=int((self.outside & self.active).sum()),
                    active_level_counts=torch.bincount(
                        self.env.scene.terrain.terrain_levels[self.active],
                        minlength=GEOMETRY_LEVELS).cpu().tolist(),
                    geometric_level_episode_totals=self.level_totals.cpu().tolist(),
                    nominal_whole_body_required_progress_m=self.required_progress_m,
                    minimum_training_speed_mps=self.minimum_training_speed_mps,
                    timeout_grading_margin=1.25,
                    cumulative_episode_totals=self.course.totals.cpu().tolist(),
                    statistic_scope="stochastic training episodes, not held-out route success")

    def load_state_dict(self, saved, *, environment_state_restored=False):
        if abs(saved["required_progress_m"] - self.required_progress_m) > 1e-6:
            raise ValueError("Robot collision hull changed across composite route resume")
        self.course.load_state_dict(saved["course"],
                                    environment_state_restored=environment_state_restored)
        self.success_streak.copy_(saved["success_streak"].to(self.device))
        self.failure_streak.copy_(saved["failure_streak"].to(self.device))
        self.level_totals.copy_(saved["level_totals"].to(self.device))
        if environment_state_restored:
            self.held_speed.copy_(saved["held_speed"].to(self.device))
            self.speed_bin.copy_(saved["speed_bin"].to(self.device))
            self.stop_on_platform.copy_(saved["stop_on_platform"].to(self.device))
        else:
            # Checkpoint reload creates new Isaac episodes; per-env command
            # phases cannot be restored without their full physical state.
            self.reset(self.active.nonzero().flatten())


def composite_success(env):
    command = env.command_manager.get_term("base_velocity")
    command.composite.refresh()
    return command.composite.active & command.composite.success


def composite_outside(env):
    command = env.command_manager.get_term("base_velocity")
    command.composite.refresh()
    return command.composite.active & command.composite.outside


def configure_composite_task(cfg):
    from isaaclab.managers import TerminationTermCfg

    terrain = cfg.scene.terrain.terrain_generator
    if terrain.num_cols != 42 or "composite" not in terrain.sub_terrains:
        raise ValueError("Configure composite task after explicit 42-column atlas")
    # Isaac's force_matrix_w is only a terrain-support signal for these
    # one-body, one-filter sensors. A changed filter could silently count a
    # wheel rubbing another robot part or obstacle as route support.
    for side in ("fr", "fl", "rr", "rl"):
        sensor = getattr(cfg.scene, f"pit_contact_{side}")
        if (not sensor.prim_path.endswith(f"/Robot/{side}_foot_Link")
                or sensor.filter_prim_paths_expr != ["/World/ground/terrain/mesh"]
                or sensor.update_period != 0.):
            raise ValueError(f"Composite {side} contact sensor lost the terrain force contract")
    # The inherited range still describes the optional 0.2 m/s diagnostic;
    # CompositeRuntime.resample uses its measured, time-feasible training floor.
    cfg.commands.base_velocity.terrain_max_command_ranges["composite"] = {
        "lin_vel_x": (.2, 1.), "lin_vel_y": (0., 0.), "ang_vel_yaw": (0., 0.)}
    cfg.terminations.composite_success = TerminationTermCfg(func=composite_success, time_out=False)
    cfg.terminations.composite_outside = TerminationTermCfg(func=composite_outside, time_out=True)
