"""Isaac Lab adapter for mixed stairs; frozen V23 sources are not edited here.

Call ``configure`` after ``configure_mixed`` and V21 stair coverage.  The
contact_provider must return measured world-XY terrain contact *points* and
terrain-filtered forces for the four wheels.  No wheel-centre proxy is used.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from mixed_stair_logic import (
    GROUP_NAMES, PROFILE, STRAIGHT, MixedStairCourse, StairGeometry,
    actual_pose_in_stair_frame, sample_cohorts, scheduled_command, signed_path_error,
)


TERMINAL_COLUMNS = (8, 11)  # V21 selected outer-platform columns, up/down.
AXIS_NAMES = ("positive_x", "positive_y", "negative_x", "negative_y")


def checkpoint_payload(command) -> dict:
    return {"mixed_stair_profile": PROFILE, "mixed_stair_course": command.mixed_stair_course.state_dict(),
            "mixed_stair_support_contract": getattr(command, "_mixed_support_contract", None)}


def restore_checkpoint(saved: dict, command) -> dict:
    """Restore new statistics exactly, or explicitly migrate old straight-only stats."""
    profile = saved.get("mixed_stair_profile")
    if profile is None:
        if saved.get("stair_command_profile") != "straight_speed_v1":
            raise ValueError("Mixed stairs need the mature straight stair source")
        command._mixed_support_contract = None
        return {"from": "straight_speed_v1", "to": PROFILE,
                "legacy_episode_totals": "retained in stair_episode_totals",
                "new_episode_totals": "initialized empty", "active_episodes": "reset on resume"}
    if profile != PROFILE:
        raise ValueError("Unknown saved mixed stair profile")
    command.mixed_stair_course.load_state_dict(saved["mixed_stair_course"])
    command._mixed_support_contract = saved.get("mixed_stair_support_contract")
    return {"from": PROFILE, "to": PROFILE,
            "new_episode_totals": "restored exactly", "active_episodes": "reset on resume"}


class MixedStairCommandMixin:
    """Put before ``MixedCommand`` in the class MRO.

    The original StraightStairCommand._update_command runs first; this mixin
    then replaces commands only in V21 terminal cells.  It therefore prevents
    the inherited zero-vy/zero-yaw assignment from erasing mixed pulses.
    """

    _mixed_stair_contact_provider: Callable | None = None

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.mixed_stair_course = MixedStairCourse(self.num_envs, self.device)
        self.mixed_platform_start = torch.zeros(self.num_envs, device=self.device)
        self.mixed_platform_end = torch.zeros(self.num_envs, device=self.device)
        self.mixed_platform_lane = torch.zeros(self.num_envs, device=self.device)
        self._mixed_support_contract = None
        receipts = env.scene.terrain.cfg.terrain_generator.stair_transition_coverage_receipts
        self._mixed_receipts = {(r["row"], r["column"]): r for r in receipts if r["selected"]}
        expected = {(row, col) for row in range(22) for col in TERMINAL_COLUMNS}
        if set(self._mixed_receipts) != expected:
            raise ValueError("V21 terminal-platform receipt set changed")

    def mixed_terminal_mask(self):
        column = self._env.scene.terrain.terrain_types
        return self.stair_mask() & ((column == TERMINAL_COLUMNS[0]) | (column == TERMINAL_COLUMNS[1]))

    def reset(self, env_ids=None):
        if hasattr(self, "mixed_stair_course"):
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long)
            self.mixed_stair_course.active[ids] = False
        return super().reset(env_ids)

    def _resample(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        fresh = ids[self.mixed_terminal_mask()[ids] & self.needs_speed[ids]]
        super()._resample(env_ids)
        if not len(fresh):
            return
        group, sign, speed_bin, speed = sample_cohorts(len(fresh), self.device)
        initial = actual_pose_in_stair_frame(
            self.robot.data.root_pos_w[fresh, :2], self.robot.data.root_quat_w[fresh],
            self._env.scene.env_origins[fresh, :2], self._env.stair_axis[fresh])
        self.mixed_stair_course.start(fresh, group, sign, speed_bin, speed, initial)
        self.speed_bins[fresh], self.held_speed[fresh] = speed_bin, speed
        self.time_left[fresh] = 1.0e9
        # Re-assert the first command after the parent sets straight-only values.
        self.commands[fresh] = scheduled_command(group, sign, speed, torch.zeros_like(speed))
        rows = self._env.scene.terrain.terrain_levels[fresh].tolist()
        cols = self._env.scene.terrain.terrain_types[fresh].tolist()
        axes = self._env.stair_axis[fresh].tolist()
        for env_id, row, col, axis in zip(fresh.tolist(), rows, cols, axes):
            geometry = StairGeometry.from_coverage_receipt(self._mixed_receipts[(row, col)], AXIS_NAMES[axis])
            self.mixed_platform_start[env_id] = geometry.platform_start_m
            self.mixed_platform_end[env_id] = geometry.platform_end_m
            self.mixed_platform_lane[env_id] = geometry.lane_half_width_m

    def _update_metrics(self):
        super()._update_metrics()
        course = self.mixed_stair_course
        ids = (self.mixed_terminal_mask() & course.active
               & (self._env.episode_length_buf > 0)).nonzero().flatten()
        if not len(ids):
            return
        actual = torch.cat((self.robot.data.root_lin_vel_b[ids, :2],
                            self.robot.data.root_ang_vel_b[ids, 2:3]), -1)
        course.observe_tracking(ids, actual, self.commands[ids])
        course.advance(ids, self._env.step_dt)
        mixed_ids = ids[course.group[ids] != STRAIGHT]
        if len(mixed_ids):
            provider = self._mixed_stair_contact_provider
            if provider is None:
                raise RuntimeError("Mixed stair promotion requires a validated platform-support provider")
            observation = provider(self._env, mixed_ids)
            if not isinstance(observation, dict):
                raise ValueError("Support provider must return an explicit measured or conservative record")
            kind = observation.get("kind")
            if kind == "measured_contact_points":
                contract = {"kind": kind}
                points_world_xy = observation["points_world_xy"]
                if points_world_xy.shape != (len(mixed_ids), 4, 2):
                    raise ValueError("Require four measured world-XY contact points")
            elif kind == "conservative_center_hull":
                collision_id = observation.get("collision_geometry_id")
                if observation.get("validated_collision_geometry") is not True or not isinstance(collision_id, str) or not collision_id:
                    raise ValueError("Conservative witness requires independently validated collision geometry")
                contract = {"kind": kind, "collision_geometry_id": collision_id,
                            "wheel_radius_m": float(observation["wheel_radius_m"])}
                wheel_centers_world_xyz = observation["wheel_centers_world_xyz"]
                if wheel_centers_world_xyz.shape != (len(mixed_ids), 4, 3):
                    raise ValueError("Require four physical wheel collider centres in world XYZ")
                points_world_xy = wheel_centers_world_xyz[..., :2]
            else:
                raise ValueError("Unknown platform support witness")
            if self._mixed_support_contract is None:
                self._mixed_support_contract = contract
            elif self._mixed_support_contract != contract:
                raise ValueError("Platform support geometry changed across checkpoint or rollout")
            contact_forces_n = observation["terrain_forces_n"]
            axis = self._env.stair_axis[mixed_ids]
            heading = axis * (math.pi / 2)
            delta = points_world_xy - self._env.scene.env_origins[mixed_ids, None, :2]
            forward = delta[..., 0] * torch.cos(heading[:, None]) + delta[..., 1] * torch.sin(heading[:, None])
            lateral = -delta[..., 0] * torch.sin(heading[:, None]) + delta[..., 1] * torch.cos(heading[:, None])
            local_points = torch.stack((forward, lateral), -1)
            start, end, lane = (self.mixed_platform_start[mixed_ids], self.mixed_platform_end[mixed_ids],
                                self.mixed_platform_lane[mixed_ids])
            physical = self._env.termination_manager.terminated[mixed_ids]
            if kind == "measured_contact_points":
                course.observe_platform(mixed_ids, local_points, contact_forces_n, start, end, lane, physical)
            else:
                hull_world_xy = observation["body_collision_hull_world_xy"]
                if hull_world_xy.ndim != 3 or hull_world_xy.shape[0] != len(mixed_ids) or hull_world_xy.shape[2] != 2:
                    raise ValueError("Require world-XY physical body collision hull")
                hull_delta = hull_world_xy - self._env.scene.env_origins[mixed_ids, None, :2]
                hull_forward = hull_delta[..., 0] * torch.cos(heading[:, None]) + hull_delta[..., 1] * torch.sin(heading[:, None])
                hull_lateral = -hull_delta[..., 0] * torch.sin(heading[:, None]) + hull_delta[..., 1] * torch.cos(heading[:, None])
                hull_local = torch.stack((hull_forward, hull_lateral), -1)
                course.observe_platform_conservative(
                    mixed_ids, local_points, hull_local, wheel_centers_world_xyz[..., 2],
                    observation["platform_world_z_m"], contact_forces_n, start, end, lane,
                    contract["wheel_radius_m"], physical)

    def _update_command(self):
        super()._update_command()
        course = self.mixed_stair_course
        ids = (self.mixed_terminal_mask() & course.active).nonzero().flatten()
        if len(ids):
            self.commands[ids] = scheduled_command(course.group[ids], course.sign[ids],
                                                   course.speed[ids], course.elapsed_s[ids])
            self.commands_xy_accumulation[ids] = course.desired_pose[ids, :2] / self.cfg.resampling_time

    def snapshot(self):
        # StraightStairCommand.snapshot asserts vy == yaw == 0 on all stairs;
        # that assertion is deliberately inapplicable to this opt-in cohort.
        course = self.mixed_stair_course
        active = self.mixed_terminal_mask() & course.active
        return {"profile": PROFILE, "active_counts_by_group": torch.bincount(
                    course.group[active], minlength=len(GROUP_NAMES)).cpu().tolist(),
                "episode_totals": course.episode_totals.cpu().tolist(),
                "legacy_straight_episode_totals_shape": list(self.episode_totals.shape),
                "completion": "four terrain supports on final platform for ten control steps",
                "support_witness": self._mixed_support_contract,
                "mixed_promotion": "full platform, relative path and three-axis tracking; never radial 4m"}


def mixed_stair_route_exit(env):
    from stair_speed import stair_route_exit

    command = env.command_manager.get_term("base_velocity")
    legacy = stair_route_exit(env)
    course = command.mixed_stair_course
    mixed = command.mixed_terminal_mask() & course.active & (course.group != STRAIGHT)
    ids = mixed.nonzero().flatten()
    if len(ids):
        actual = actual_pose_in_stair_frame(command.robot.data.root_pos_w[ids, :2],
                                            command.robot.data.root_quat_w[ids],
                                            env.scene.env_origins[ids, :2], env.stair_axis[ids])
        end = command.mixed_platform_end[ids]
        out_of_route = (actual[:, 0] > end - 0.15) | (actual[:, 1].abs() > command.mixed_platform_lane[ids] + 0.5)
        legacy[ids] = course.platform_complete[ids] | out_of_route
    return legacy


def mixed_stair_curriculum(env, env_ids):
    """Preserve pit/legacy straight grades; grade mixed cells on full platform."""
    from stair_speed import straight_stair_terrain_levels

    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    command = env.command_manager.get_term("base_velocity")
    pit = command.pit
    course = command.mixed_stair_course
    allocation = getattr(env, "terrain_allocation", None)
    if allocation is not None:
        allocation.record(ids)
    mixed = command.mixed_terminal_mask()[ids] & course.active[ids] & (course.group[ids] != STRAIGHT)
    legacy_ids = ids[~pit.active[ids] & ~mixed]
    if len(legacy_ids):
        straight_stair_terrain_levels(env, legacy_ids)
    selected = ids[mixed]
    if len(selected):
        actual = actual_pose_in_stair_frame(command.robot.data.root_pos_w[selected, :2],
                                            command.robot.data.root_quat_w[selected],
                                            env.scene.env_origins[selected, :2], env.stair_axis[selected])
        error = signed_path_error(actual, course.desired_pose[selected])
        tracking = course.tracking_error_sum[selected] / course.tracking_count[selected, None].clamp(min=1)
        eligible = course.speed[selected] * max(env.max_episode_length_s - 2.0, 0.0) >= (
            command.mixed_platform_start[selected] + 0.5) * 1.25
        physical = env.termination_manager.terminated[selected]
        complete = course.platform_complete[selected]
        good = (error[:, 1].abs() <= 0.35) & (error[:, 2].abs() <= 0.25) & (
            tracking[:, 0] <= 0.25) & (tracking[:, 1] <= 0.15) & (tracking[:, 2] <= 0.15)
        promote = complete & eligible & good & ~physical
        downgrade = physical | (eligible & ~promote)
        terrain = env.scene.terrain
        course.record(selected, command.learner[selected],
                      (command.terrain_idxs[selected] == command.terrain_type2idx["stairs_down"]).long(),
                      terrain.terrain_levels[selected], eligible, complete, promote, downgrade, physical,
                      error[:, 1], tracking)
        terrain.update_env_origins(selected, promote, downgrade)
    pit_ids = ids[pit.active[ids]]
    if len(pit_ids):
        pit.record_and_resample_levels(pit_ids)
    if allocation is not None:
        allocation.after_curriculum(ids)
    return env.scene.terrain.terrain_levels[~pit.active].float().mean()


def configure(cfg, contact_provider: Callable):
    """Install after V21 coverage; fail closed if measured contact data is absent."""
    from mixed_task import MixedCommand, mixed_curriculum
    from stair_speed import stair_route_exit

    if not callable(contact_provider):
        raise ValueError("A measured terrain-contact provider is required")
    if cfg.commands.base_velocity.class_type is not MixedCommand:
        raise ValueError("Expected frozen V23 MixedCommand before mixed-stair migration")
    if cfg.curriculum.terrain_levels.func is not mixed_curriculum:
        raise ValueError("Original mixed curriculum has changed")
    if cfg.terminations.stair_route_exit.func is not stair_route_exit:
        raise ValueError("Original 4m route termination has changed")
    runtime_type = type("MeasuredMixedStairCommand", (MixedStairCommandMixin, MixedCommand),
                        {"__module__": __name__, "_mixed_stair_contact_provider": staticmethod(contact_provider)})
    globals()[runtime_type.__name__] = runtime_type
    cfg.commands.base_velocity.class_type = runtime_type
    cfg.curriculum.terrain_levels.func = mixed_stair_curriculum
    cfg.terminations.stair_route_exit.func = mixed_stair_route_exit
    return {"profile": PROFILE, "terminal_columns": TERMINAL_COLUMNS,
            "straight_group_retains_legacy_curriculum": True,
            "mixed_group_uses_measured_terminal_platform": True}
