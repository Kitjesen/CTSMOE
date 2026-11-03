"""Nine terrain CTS task: original locomotion plus independently graded pit exits."""
import copy
import math

import numpy as np
import torch
import trimesh
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
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
        for sub in cfg.sub_terrains.values():
            if hasattr(sub, "border_width"):
                sub.border_width = cfg.sub_terrain_border_width
        super().__init__(cfg, device)

    def _get_terrain_mesh(self, difficulty, cfg):
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
            return super()._get_terrain_mesh(difficulty, cfg)
        finally:
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


def mixed_event(env, env_ids, original, pit_position_range=None, **original_params):
    """Dispatch reset and interval events without changing original terrain draws."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    mask = env.command_manager.get_term("base_velocity").pit.active[ids]
    if len(ids[~mask]):
        original(env, ids[~mask], **original_params)
    if pit_position_range is not None and len(ids[mask]):
        original(env, ids[mask], **dict(original_params, position_range=pit_position_range))


def mixed_original_reward(env, original, **original_params):
    value = original(env, **original_params)
    return torch.where(env.command_manager.get_term("base_velocity").pit.active, 0., value)


def mixed_curriculum(env, env_ids):
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    pit = env.command_manager.get_term("base_velocity").pit
    if len(ids[~pit.active[ids]]):
        straight_stair_terrain_levels(env, ids[~pit.active[ids]])
    if len(ids[pit.active[ids]]):
        pit.record_and_resample_levels(ids[pit.active[ids]])
    return env.scene.terrain.terrain_levels[~pit.active].float().mean()


def configure_mixed(cfg):
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
    for side in ("fr", "fl", "rr", "rl"):
        setattr(cfg.scene, f"pit_contact_{side}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_foot_Link", update_period=0., history_length=1,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"]))
    cfg.commands.base_velocity.class_type = MixedCommand
    cfg.commands.base_velocity.terrain_max_command_ranges["pit"] = {
        "lin_vel_x": (.2, 1.), "lin_vel_y": (0., 0.), "ang_vel_yaw": (0., 0.)}
    cfg.events.reset_base.func = mixed_reset_root
    term = cfg.events.reset_robot_joints
    original, params = term.func, copy.deepcopy(term.params)
    term.func = mixed_event
    term.params = dict(original=original, pit_position_range=(.95, 1.05), **params)
    push = cfg.events.randomize_push_robot
    push.func, push.params = mixed_event, dict(original=push.func, **copy.deepcopy(push.params))
    cfg.curriculum.terrain_levels.func = mixed_curriculum
    # The original global reward curriculum may change weights, so gate function
    # outputs rather than globally disabling these rewards in the mixed task.
    for name in ("lin_vel_z_l2", "feet_regulation"):
        term = getattr(cfg.rewards, name)
        term.func, term.params = mixed_original_reward, dict(original=term.func, **term.params)
    cfg.rewards.pit_progress = RewardTermCfg(func=pit_reward, weight=3., params={"kind": "progress"})
    cfg.rewards.pit_success = RewardTermCfg(func=pit_reward, weight=10., params={"kind": "success"})
    cfg.rewards.pit_stationary = RewardTermCfg(func=pit_reward, weight=2., params={"kind": "stationary"})
    cfg.terminations.pit_success = TerminationTermCfg(func=pit_success, time_out=False)
    cfg.terminations.pit_outside = TerminationTermCfg(func=pit_outside, time_out=False)
