"""Official mixed mesh layout with Lope stair and slope ceilings.

The generator, row count, and sub-terrain names come from
``MIXED_MESH_TERRAIN_CFG``. Stairs stay at 5–25 cm. Slopes stay at
``(0, tan(35°))``, a rise/run gradient. The wheels sub-terrain is removed and
its proportion is added to flat. ``stairs_up`` is the official inverted pyramid.
"""
import copy

from go2_rl_lab.tasks.mixed.terrain_mesh import MIXED_MESH_TERRAIN_CFG, MixedTerrainImporter
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from lope_lab3.spec import TERRAIN


class LopeTerrainImporter(MixedTerrainImporter):
    def _compute_env_origins_curriculum(self, num_envs, origins):
        result = super()._compute_env_origins_curriculum(num_envs, origins)
        level = self.cfg.fixed_level
        if level is None:
            return result
        if not 0 <= level < origins.shape[0]:
            raise ValueError("fixed terrain level outside generated rows")
        self.terrain_levels[:] = level
        return origins[self.terrain_levels, self.terrain_types]


@configclass
class LopeTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = LopeTerrainImporter
    fixed_level: int | None = None


def terrain_cfg():
    """Copy the official mesh generator, then apply Lope's locked ceilings."""
    cfg = copy.deepcopy(MIXED_MESH_TERRAIN_CFG)
    cfg.size = TERRAIN["size"]
    cfg.border_width = TERRAIN["border_width"]
    cfg.num_rows = TERRAIN["num_rows"]
    cfg.num_cols = TERRAIN["num_cols"]
    cfg.sub_terrains.pop("wheels", None)
    terrains = cfg.sub_terrains
    terrains["flat"].proportion = TERRAIN["flat_proportion"]
    slope = TERRAIN["slope_range"]
    for name in ("slope_up", "slope_down", "rough_slope"):
        terrains[name].slope_range = slope
    steps = TERRAIN["stair_step_height_range"]
    width = TERRAIN["stair_step_width"]
    for name in ("stairs_up", "stairs_down"):
        terrains[name].step_height_range = steps
        terrains[name].step_width = width
    terrains["stairs_up"].proportion = TERRAIN["stairs_up_proportion"]
    terrains["stairs_down"].proportion = TERRAIN["stairs_down_proportion"]
    terrains["obstacles"].obstacle_height_range = TERRAIN["obstacle_height_range"]
    terrains["obstacles"].obstacle_width_range = TERRAIN["obstacle_width_range"]
    terrains["wave"].amplitude_range = TERRAIN["wave_amplitude_range"]
    return cfg
