"""Apply explicitly selected candidates before creating the environment."""
import math

from ..boxes import gradual_random_grid_terrain
from .height import HeightOptions


def configure(cfg, *, gradual_boxes=False, height_weight=0.0, height_options=None):
    """All defaults preserve the current configuration; negative height weight is a cost."""
    if not math.isfinite(height_weight) or height_weight > 0:
        raise ValueError("height_weight must be finite and nonpositive")
    if height_weight and not isinstance(height_options, HeightOptions):
        raise ValueError("an enabled height cost requires explicit HeightOptions")
    term = None
    if height_weight:
        from isaaclab.managers import RewardTermCfg, SceneEntityCfg
        from .isaac_height import CroppedGroundHeight
        if hasattr(cfg.rewards, "cropped_ground_height"):
            raise ValueError("cropped_ground_height is already configured")
        term = RewardTermCfg(
            func=CroppedGroundHeight, weight=height_weight,
            params={"options": height_options, "asset_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg("height_scanner"),
                    "x_range": (-0.2, 0.2), "y_range": (-0.15, 0.15)})
    if gradual_boxes:
        generator = cfg.scene.terrain.terrain_generator
        boxes = generator.sub_terrains["boxes"]
        boxes.function = gradual_random_grid_terrain
        # A changed generator must not reuse terrain cached by an old function.
        generator.use_cache = False
    if term is not None:
        cfg.rewards.cropped_ground_height = term
    return cfg
