"""Nine terrain CTS task: original locomotion plus independently graded pit exits."""
import copy
import math

import numpy as np
import torch
import trimesh
from isaaclab.managers import RewardTermCfg, TerminationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import SubTerrainBaseCfg, TerrainImporter
from isaaclab.terrains.terrain_generator import TerrainGenerator

from pit_geometry import build_pit_meshes, height_for_level
from pit_task import PitCommand, pit_outside, pit_reward, pit_success, reset_pit_root
from stair_speed import StraightStairCommand, reset_stair_root, straight_stair_terrain_levels

PROFILE = "mixed9_v1"


def mixed_pit_terrain(difficulty, cfg):
    if tuple(cfg.size) != (9., 9.):
        raise ValueError("Mixed task requires the original 9m tile")
    level = min(int(math.floor(difficulty*22)), 21)
    meshes, origin = build_pit_meshes(height_for_level(level))
    # Embed the existing 8x6 pit in the original 9x9 tile, with a solid floor
    # under the full tile so the original teacher scanner always sees terrain.
    offset = np.array((.5, 1.5, 0.))
    meshes = meshes[1:]
    for mesh in meshes:
        mesh.apply_translation(offset)
    floor = trimesh.creation.box(extents=(9., 9., .2))
    floor.apply_translation((4.5, 4.5, -.1))
    return [floor, *meshes], origin+offset


class MixedTerrainGenerator(TerrainGenerator):
    """Keep original level 0..9 difficulty ranges while pit has 22 height rows."""
    def __init__(self, cfg, device="cpu"):
        self.slope_geometry_receipts = []
        for sub in cfg.sub_terrains.values():
            if hasattr(sub, "border_width"):
                sub.border_width = cfg.sub_terrain_border_width
        super().__init__(cfg, device)
        cfg.slope_geometry_receipts = self.slope_geometry_receipts

    def _get_terrain_mesh(self, difficulty, cfg):
        atlas_row = min(int(float(difficulty)*22), 21)
        original_slope_range = getattr(cfg, "slope_range", None)
        if getattr(cfg, "slope45_enabled", False):
            from slope_curriculum import slope_range_for_difficulty
            cfg.slope_range = slope_range_for_difficulty(difficulty)
        if cfg.function is not mixed_pit_terrain:
            scaled = float(difficulty)*22
            row = min(int(scaled), 21)
            fraction = scaled-row
            # Rows 10..21 for original types are never assigned. Repeat level 9
            # geometry there to keep the shared rectangular terrain atlas valid.
            difficulty = (min(row, 9) + (0. if self.cfg.use_gym_difficulty else fraction))/10
        original = getattr(cfg, "slope_threshold", None)
        override = getattr(cfg, "slope_threshold_override", None)
        if override is not None:
            cfg.slope_threshold = override
        try:
            result = super()._get_terrain_mesh(difficulty, cfg)
            if getattr(cfg, "slope45_enabled", False) and atlas_row in (0, 9):
                mesh, _ = result
                vertices = np.asarray(mesh.vertices)
                # The returned tile is centered on XY zero. Sample its center
                # line outside the 3m platform and away from the outer border.
                ys = np.unique(vertices[:, 1])
                center_y = ys[np.argmin(np.abs(ys))]
                points = vertices[np.isclose(vertices[:, 1], center_y)]
                points = points[(np.abs(points[:, 0]) > 1.6) & (np.abs(points[:, 0]) < 3.8)]
                points = points[np.argsort(points[:, 0])]
                dx, dz = np.diff(points[:, 0]), np.diff(points[:, 2])
                valid = (dx > .01) & (dx < .15) & (np.abs(dz) > .001)
                angles = np.degrees(np.arctan(np.abs(dz[valid]/dx[valid])))
                if not len(angles):
                    raise ValueError("Slope mesh has no measurable center-line gradient")
                self.slope_geometry_receipts.append(dict(
                    function=cfg.function.__name__, row=atlas_row,
                    nominal_angle_deg=math.degrees(math.atan(cfg.slope_range[0])),
                    centerline_angle_deg_median=float(np.median(angles)),
                    centerline_angle_deg_min=float(angles.min()),
                    centerline_angle_deg_max=float(angles.max()),
                    slope_threshold=cfg.slope_threshold,
                ))
            return result
        finally:
            if getattr(cfg, "slope45_enabled", False):
                cfg.slope_range = original_slope_range
            if override is not None:
                cfg.slope_threshold = original


class MixedTerrainImporter(TerrainImporter):
    def update_env_origins(self, env_ids, move_up, move_down):
        # Only the original terrain curriculum calls this method. Pit has its own
        # student gate and writes its row directly in PitCommand.
        if torch.any(self.terrain_types[env_ids] >= 20):
            raise ValueError("Original terrain curriculum received a pit environment")
        levels = self.terrain_levels[env_ids] + move_up.long() - move_down.long()
        self.terrain_levels[env_ids] = torch.where(
            levels >= 10, torch.randint_like(levels, 10), levels.clamp(min=0))
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]]


class MixedCommand(StraightStairCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.pit = PitCommand(copy.deepcopy(cfg), env)
        self.refresh_membership()

    def refresh_membership(self):
        self._init_terrain_infos()
        self._update_env_command_ranges()
        self.pit.active = self.terrain_idxs == self.terrain_type2idx["pit"]

    def set_student_indices(self, indices):
        super().set_student_indices(indices)
        self.pit.learner.zero_()
        self.pit.learner[indices] = 1

    def _update_metrics(self):
        super()._update_metrics()
        if hasattr(self._env, "terrain_allocation"):
            self._env.terrain_allocation.step()

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self.pit.reset(ids[self.pit.active[ids]])
        return super().reset(ids)

    def _resample(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        selected = self.pit.active[ids]
        super()._resample(ids[~selected])
        pit_ids = ids[selected]
        self.commands[pit_ids] = self.pit.commands[pit_ids]
        self.time_left[pit_ids] = 1e9

    def _update_command(self):
        super()._update_command()
        ids = self.pit.active.nonzero().flatten()
        self.commands[ids] = 0
        self.commands[ids, 0] = torch.where(self.pit.stop_on_platform[ids], 0., self.pit.held_speed[ids])


def mixed_reset_root(env, env_ids, pose_range, velocity_range, asset_cfg):
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    pit = env.command_manager.get_term("base_velocity").pit.active[ids]
    if len(ids[~pit]):
        reset_stair_root(env, ids[~pit], pose_range, velocity_range, asset_cfg)
    if len(ids[pit]):
        reset_pit_root(env, ids[pit])


def mixed_reset_joints(env, env_ids, position_range, velocity_range, asset_cfg=SceneEntityCfg("robot")):
    """Keep original joint draws on old terrains; use the tested pit spawn range."""
    from isaaclab.envs.mdp.events import reset_joints_by_scale
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    mask = env.command_manager.get_term("base_velocity").pit.active[ids]
    if len(ids[~mask]):
        reset_joints_by_scale(env, ids[~mask], position_range, velocity_range, asset_cfg)
    if len(ids[mask]):
        reset_joints_by_scale(env, ids[mask], (.95, 1.05), velocity_range, asset_cfg)


def mixed_push(env, env_ids, velocity_range, asset_cfg=SceneEntityCfg("robot")):
    from isaaclab.envs.mdp.events import push_by_setting_velocity
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    ids = ids[~env.command_manager.get_term("base_velocity").pit.active[ids]]
    if len(ids):
        push_by_setting_velocity(env, ids, velocity_range, asset_cfg)


def mixed_lin_vel_z(env):
    from robot_lab.tasks.go2.mdp.rewards import lin_vel_z_l2
    value = lin_vel_z_l2(env)
    return torch.where(env.command_manager.get_term("base_velocity").pit.active, 0., value)


def mixed_feet_regulation(env, base_height_target, wheel_radius, asset_cfg, sensor_cfg=None):
    from wheel_rewards import wheel_feet_regulation
    value = wheel_feet_regulation(env, base_height_target, wheel_radius, asset_cfg, sensor_cfg)
    return torch.where(env.command_manager.get_term("base_velocity").pit.active, 0., value)


def mixed_curriculum(env, env_ids):
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    pit = env.command_manager.get_term("base_velocity").pit
    allocation = getattr(env, "terrain_allocation", None)
    if allocation is not None:
        allocation.record(ids)
    if len(ids[~pit.active[ids]]):
        straight_stair_terrain_levels(env, ids[~pit.active[ids]])
    if len(ids[pit.active[ids]]):
        pit.record_and_resample_levels(ids[pit.active[ids]])
    if allocation is not None:
        allocation.after_curriculum(ids)
    return env.scene.terrain.terrain_levels[~pit.active].float().mean()


def configure_mixed(cfg, slope_profile="original", pit_training_profile="strict_v1"):
    if pit_training_profile not in ("strict_v1", "climb_first_v2"):
        raise ValueError("Unknown pit training profile")
    cfg.pit_training_profile = pit_training_profile
    cfg.mixed_profile, cfg.pit_max_level = PROFILE, 21
    terrain = cfg.scene.terrain.terrain_generator
    if terrain.num_cols != 20 or terrain.num_rows != 10 or tuple(terrain.size) != (9., 9.):
        raise ValueError("Expected the saved original terrain geometry layout")
    # Double columns: first 20 have exactly the previous terrain allocation.
    total = sum(s.proportion for s in terrain.sub_terrains.values())
    for sub in terrain.sub_terrains.values():
        sub.proportion /= 2*total
    terrain.sub_terrains["pit"] = SubTerrainBaseCfg(function=mixed_pit_terrain, proportion=.5)
    terrain.num_rows, terrain.num_cols = 22, 40
    terrain.class_type = MixedTerrainGenerator
    cfg.scene.terrain.class_type = MixedTerrainImporter
    cfg.scene.terrain.max_init_terrain_level = 2
    if slope_profile == "slope45_v1":
        for name in ("slope_up", "slope_down", "rough_slope"):
            terrain.sub_terrains[name].slope45_enabled = True
    elif slope_profile != "original":
        raise ValueError("Unknown slope curriculum")
    for side in ("fr", "fl", "rr", "rl"):
        setattr(cfg.scene, f"pit_contact_{side}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_foot_Link", update_period=0., history_length=1,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"]))
    cfg.commands.base_velocity.class_type = MixedCommand
    cfg.commands.base_velocity.terrain_max_command_ranges["pit"] = {
        "lin_vel_x": (.2, 1.), "lin_vel_y": (0., 0.), "ang_vel_yaw": (0., 0.)}
    cfg.events.reset_base.func = mixed_reset_root
    term = cfg.events.reset_robot_joints
    term.func = mixed_reset_joints
    push = cfg.events.randomize_push_robot
    push.func = mixed_push
    cfg.curriculum.terrain_levels.func = mixed_curriculum
    # The original global reward curriculum may change weights, so gate function
    # outputs rather than globally disabling these rewards in the mixed task.
    cfg.rewards.lin_vel_z_l2.func = mixed_lin_vel_z
    cfg.rewards.feet_regulation.func = mixed_feet_regulation
    cfg.rewards.pit_progress = RewardTermCfg(func=pit_reward, weight=3., params={"kind": "progress"})
    cfg.rewards.pit_success = RewardTermCfg(func=pit_reward, weight=10., params={"kind": "success"})
    cfg.rewards.pit_stationary = RewardTermCfg(func=pit_reward, weight=2., params={"kind": "stationary"})
    cfg.terminations.pit_success = TerminationTermCfg(func=pit_success, time_out=False)
    cfg.terminations.pit_outside = TerminationTermCfg(func=pit_outside, time_out=False)
    if pit_training_profile == "climb_first_v2":
        from pit_task import pit_settle_timeout
        cfg.rewards.pit_climb = RewardTermCfg(func=pit_reward, weight=5., params={"kind": "climb"})
        cfg.rewards.pit_stability = RewardTermCfg(func=pit_reward, weight=1., params={"kind": "stability"})
        cfg.terminations.pit_settle_timeout = TerminationTermCfg(func=pit_settle_timeout, time_out=True)


def validate_mixed_runtime(env, expected_pit_count=None):
    """Check live routing, caps and reward isolation without advancing simulation."""
    from types import SimpleNamespace
    command = env.command_manager.get_term("base_velocity")
    mask = command.pit.active
    levels = env.scene.terrain.terrain_levels
    expected_pit_count = env.num_envs//2 if expected_pit_count is None else expected_pit_count
    assert int(mask.sum()) == expected_pit_count
    assert torch.equal(mask, env.scene.terrain.terrain_types >= 20)
    assert bool(((levels[~mask] >= 0) & (levels[~mask] < 10)).all())
    assert bool(((levels[mask] >= 0) & (levels[mask] < 22)).all())
    assert bool((command.commands[mask, 1:] == 0).all())
    assert bool(torch.isfinite(env.scene["height_scanner"].data.ray_hits_w).all())
    reward_names = ["pit_progress", "pit_success", "pit_stationary"]
    if env.cfg.pit_training_profile == "climb_first_v2":
        reward_names.extend(("pit_climb", "pit_stability"))
    for name in reward_names:
        term = env.reward_manager.get_term_cfg(name)
        values = term.func(env, **term.params)
        assert bool(torch.isfinite(values).all())
        assert bool((values[~mask] == 0).all()), name
    for name in ("lin_vel_z_l2", "feet_regulation"):
        term = env.reward_manager.get_term_cfg(name)
        values = term.func(env, **term.params)
        assert bool(torch.isfinite(values).all())
        assert bool((values[mask] == 0).all()), name
    # Exercise level-9 promotion on an isolated importer state: it must wrap
    # within old levels even though the common terrain atlas has 22 rows.
    mock = SimpleNamespace(terrain_types=torch.zeros(128, dtype=torch.long, device=env.device),
                           terrain_levels=torch.full((128,), 9, device=env.device),
                           terrain_origins=env.scene.terrain.terrain_origins,
                           env_origins=torch.zeros((128, 3), device=env.device))
    ids = torch.arange(128, device=env.device)
    with torch.random.fork_rng(devices=[torch.device(env.device).index]):
        MixedTerrainImporter.update_env_origins(mock, ids, torch.ones_like(ids), torch.zeros_like(ids))
    assert bool(((mock.terrain_levels >= 0) & (mock.terrain_levels < 10)).all())
    geometry = env.cfg.scene.terrain.terrain_generator.slope_geometry_receipts
    if getattr(env.cfg.scene.terrain.terrain_generator.sub_terrains["slope_up"], "slope45_enabled", False):
        assert any(item["row"] == 9 and abs(item["nominal_angle_deg"]-45.) < 1e-6 for item in geometry)
    return dict(old_level_cap=9, pit_level_cap=21, pit_rewards_zero_on_old=True,
                old_vertical_and_wheel_rewards_zero_on_pit=True, scanner_finite=True,
                old_level9_promotion_bounded=True, pit_count=int(mask.sum()),
                shared_self_collisions=env.cfg.scene.robot.spawn.articulation_props.enabled_self_collisions,
                slope_geometry=geometry)
