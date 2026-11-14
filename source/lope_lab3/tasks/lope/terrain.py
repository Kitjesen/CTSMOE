"""Lope rough terrain. Stairs are 5–25 cm. Slopes are a 0–35° gradient.

``slope_range`` is rise/run. Isaac Lab sets platform height to ``slope * run``,
so the upper value is ``tan(35°)``, not 35 and not 35° in radians.

Class-to-name assignment follows ``rough_cfg.py``. Isaac Lab 3 documents the
non-inverted pyramid as a high center platform, so Lope's ``stairs_up`` key is
the non-inverted mesh even though walking off that platform goes downward.
Both stair classes use the 5–25 cm step range. Playback selects those keys.
"""
import isaaclab.terrains as terrain_gen
from go2_rl_lab.tasks.mixed.terrain_mesh import MixedTerrainImporter
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
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
    slope = TERRAIN["slope_range"]
    steps = TERRAIN["stair_step_height_range"]
    return TerrainGeneratorCfg(
        seed=42,
        size=TERRAIN["size"],
        border_width=TERRAIN["border_width"],
        num_rows=TERRAIN["num_rows"],
        num_cols=TERRAIN["num_cols"],
        horizontal_scale=TERRAIN["horizontal_scale"],
        vertical_scale=TERRAIN["vertical_scale"],
        slope_threshold=TERRAIN["slope_threshold"],
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        curriculum=True,
        color_scheme="none",
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=TERRAIN["flat_proportion"]),
            "rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=TERRAIN["rough_proportion"],
                noise_range=TERRAIN["rough_noise_range"],
                noise_step=TERRAIN["rough_noise_step"],
                downsampled_scale=TERRAIN["rough_downsampled_scale"],
                border_width=TERRAIN["slope_border_width"],
            ),
            "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=TERRAIN["slope_proportion"],
                slope_range=slope,
                platform_width=TERRAIN["slope_platform_width"],
                border_width=TERRAIN["slope_border_width"],
            ),
            "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=TERRAIN["slope_proportion"],
                slope_range=slope,
                platform_width=TERRAIN["slope_platform_width"],
                border_width=TERRAIN["slope_border_width"],
            ),
            "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=TERRAIN["stair_proportion"],
                step_height_range=steps,
                step_width=TERRAIN["stair_step_width"],
                platform_width=TERRAIN["stair_platform_width"],
                border_width=TERRAIN["stair_border_width"],
                holes=False,
            ),
            "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=TERRAIN["stair_proportion"],
                step_height_range=steps,
                step_width=TERRAIN["stair_step_width"],
                platform_width=TERRAIN["stair_platform_width"],
                border_width=TERRAIN["stair_border_width"],
                holes=False,
            ),
            "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=TERRAIN["obstacle_proportion"],
                obstacle_width_range=TERRAIN["obstacle_width_range"],
                obstacle_height_range=TERRAIN["obstacle_height_range"],
                num_obstacles=TERRAIN["obstacle_count"],
                platform_width=TERRAIN["stair_platform_width"],
                border_width=TERRAIN["slope_border_width"],
            ),
            "wave": terrain_gen.HfWaveTerrainCfg(
                proportion=TERRAIN["wave_proportion"],
                amplitude_range=TERRAIN["wave_amplitude_range"],
                num_waves=TERRAIN["wave_count"],
                border_width=TERRAIN["slope_border_width"],
            ),
        },
    )
