"""Yaw-aware terrain curriculum for Lope locomotion.

Promotion requires traversing beyond half of the usable tile length (``> 4.0 m``
on 9 m tiles with 0.5 m sub-terrain borders) without early termination so robots
must actually climb out cleanly to level up. Demotion triggers on early
termination (falls) or genuine forward-progress stalling along the commanded XY
direction, avoiding false demotions when robots obediently follow curved yaw
commands inside the central platform.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from go2_rl_lab.tasks.mixed.mdp.curriculums import TerrainLevelsVelGym

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_terrain_level_updates(
    maximum_distance,
    terrain_length: float,
    terminated,
    projected_move_distance,
    commanded_move_distance,
    min_commanded_distance: float = 1.0,
):
    """Compute ``(move_up, move_down)`` boolean masks for resetting environments."""
    move_up = (maximum_distance > (terrain_length / 2.0)) & (~terminated)
    stalled = (commanded_move_distance >= min_commanded_distance) & (
        projected_move_distance < commanded_move_distance * 0.5
    )
    move_down = (terminated | stalled) & (~move_up)
    return move_up, move_down


class LopeTerrainLevelsCurriculum(TerrainLevelsVelGym):
    """Terrain curriculum with strict clean-climb promotion, yaw-aware demotion, and L9/rate metrics."""

    def __init__(self, cfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        n_terrains = len(self._terrain_names)
        self._up_rate = torch.zeros(n_terrains, dtype=torch.float, device=env.device)
        self._down_rate = torch.zeros(n_terrains, dtype=torch.float, device=env.device)
        self._rate_seen = torch.zeros(n_terrains, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        terrain = env.scene.terrain
        command = env.command_manager.get_term("base_velocity")

        terrain_generator = terrain.cfg.terrain_generator
        if terrain_generator is None:
            return {"mean_level": torch.tensor(0.0, device=env.device)}

        sub_terrain_border_width = getattr(terrain_generator, "sub_terrain_border_width", 0.0) or 0.0
        terrain_length = max(
            0.0,
            terrain_generator.size[0] - 2.0 * sub_terrain_border_width,
        )

        maximum_distance = command.max_move_distance[env_ids]
        terminated_buf = getattr(env, "reset_terminated", None)
        if terminated_buf is None:
            terminated_buf = env.termination_manager.terminated
        terminated = terminated_buf[env_ids].bool()
        projected_distance = command.projected_move_distance[env_ids]
        commanded_distance = command.commanded_move_distance[env_ids]

        move_up, move_down = compute_terrain_level_updates(
            maximum_distance=maximum_distance,
            terrain_length=terrain_length,
            terminated=terminated,
            projected_move_distance=projected_distance,
            commanded_move_distance=commanded_distance,
        )
        terrain.update_env_origins(env_ids, move_up, move_down)

        if isinstance(env_ids, slice):
            resolved_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        else:
            resolved_ids = env_ids
        active_episode = env.episode_length_buf[resolved_ids] > 0

        levels = terrain.terrain_levels.float()
        max_level = float(max(0, terrain_generator.num_rows - 1))
        statistics = {"mean_level": torch.mean(levels)}
        for terrain_index, terrain_name in enumerate(self._terrain_names):
            terrain_mask = self._terrain_masks[terrain_name]
            if self._terrain_has_envs[terrain_name]:
                terrain_levels = levels[terrain_mask]
                mean_level = torch.mean(terrain_levels)
                current_max_level = torch.max(terrain_levels)
                self._max_levels_reached[terrain_index] = torch.maximum(
                    self._max_levels_reached[terrain_index], current_max_level
                )
                l9_ratio = torch.mean((terrain_levels >= max_level - 0.5).float())
                l7_plus_ratio = torch.mean((terrain_levels >= max(0.0, max_level - 2.5)).float())
                reset_mask = terrain_mask[resolved_ids] & active_episode
                if bool(reset_mask.any()):
                    batch_up = move_up[reset_mask].float().mean()
                    batch_down = move_down[reset_mask].float().mean()
                    if not bool(self._rate_seen[terrain_index]):
                        self._up_rate[terrain_index] = batch_up
                        self._down_rate[terrain_index] = batch_down
                        self._rate_seen[terrain_index] = True
                    else:
                        self._up_rate[terrain_index] = 0.9 * self._up_rate[terrain_index] + 0.1 * batch_up
                        self._down_rate[terrain_index] = 0.9 * self._down_rate[terrain_index] + 0.1 * batch_down
            else:
                mean_level = torch.tensor(-1.0, device=env.device)
                l9_ratio = torch.tensor(0.0, device=env.device)
                l7_plus_ratio = torch.tensor(0.0, device=env.device)

            statistics[f"{terrain_name}/mean_level"] = mean_level
            statistics[f"{terrain_name}/max_level_reached"] = self._max_levels_reached[terrain_index]
            statistics[f"{terrain_name}/l9_ratio"] = l9_ratio
            statistics[f"{terrain_name}/l7_plus_ratio"] = l7_plus_ratio
            statistics[f"{terrain_name}/up_rate"] = self._up_rate[terrain_index]
            statistics[f"{terrain_name}/down_rate"] = self._down_rate[terrain_index]
        return statistics
