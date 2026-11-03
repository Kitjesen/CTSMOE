"""Reuse the 20261102 gradual amplitude curve; keep the native mesh generator."""
import copy
import math


def amplitude(difficulty):
    d = float(difficulty)
    if not math.isfinite(d) or not 0 <= d <= 1:
        raise ValueError("difficulty must be finite and in [0, 1]")
    height = 0.01 + d * (0.20 - 0.01)
    if d > 0.1:
        height -= (15.0 / 14.0) * (d - 0.1) ** 2 * (1.0 - d) ** 2
    return height


def with_amplitude(difficulty, cfg, native_generator):
    local = copy.copy(cfg)
    height = amplitude(difficulty)
    local.grid_height_range = (height, height)
    return native_generator(difficulty, local)


def gradual_random_grid_terrain(difficulty, cfg):
    from isaaclab.terrains.trimesh.mesh_terrains import random_grid_terrain
    return with_amplitude(difficulty, cfg, random_grid_terrain)
