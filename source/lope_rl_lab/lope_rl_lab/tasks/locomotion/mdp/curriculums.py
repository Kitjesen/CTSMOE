"""Yaw-aware terrain curriculum for Lope locomotion.

Promotion still strictly requires traversing beyond half of the usable tile
length (``> 4.0 m`` on 9 m tiles with 0.5 m sub-terrain borders) so robots must
actually climb out of stair/slope tiles to level up. Demotion triggers on early
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
    move_up = maximum_distance > (terrain_length / 2.0)
    stalled = (commanded_move_distance >= min_commanded_distance) & (
        projected_move_distance < commanded_move_distance * 0.5
    )
    move_down = (terminated | stalled) & (~move_up)
    return move_up, move_down


class LopeTerrainLevelsCurriculum(TerrainLevelsVelGym):
    """Terrain curriculum that keeps strict climb-out promotion without yaw-circle false demotions."""

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

        levels = terrain.terrain_levels.float()
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
            else:
                mean_level = torch.tensor(-1.0, device=env.device)

            statistics[f"{terrain_name}/mean_level"] = mean_level
            statistics[f"{terrain_name}/max_level_reached"] = self._max_levels_reached[terrain_index]
        return statistics
