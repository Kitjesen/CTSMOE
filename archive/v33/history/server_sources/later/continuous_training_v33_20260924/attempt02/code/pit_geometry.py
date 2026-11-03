"""Single vertical exit from a closed pit; coordinates before Isaac tile translation."""
import math

import numpy as np
import trimesh

LEVEL_COUNT = 22
SIZE = (8.0, 6.0)
ORIGIN = (3.0, 3.0, 0.0)
EDGE_X = 4.5
EDGE_OFFSET = EDGE_X - ORIGIN[0]


def height_for_level(level):
    if int(level) != level or not 0 <= level < LEVEL_COUNT:
        raise ValueError("Pit level must be an integer in [0, 21]")
    return round(0.15 + 0.05 * int(level), 8)


def level_from_difficulty(difficulty):
    if not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("Difficulty must be finite and in [0, 1]")
    return min(int(math.floor(difficulty * LEVEL_COUNT)), LEVEL_COUNT - 1)


def pit_boxes(height):
    if not math.isfinite(height) or not 0.15 <= height <= 1.2:
        raise ValueError("Pit height outside designed range")
    return {
        "floor": ((0., 0., -0.2), (8., 6., 0.)),
        "platform": ((EDGE_X, 0., 0.), (8., 6., height)),
        "back": ((0., 0., 0.), (0.2, 6., height)),
        "left": ((0.2, 0., 0.), (EDGE_X, 0.2, height)),
        "right": ((0.2, 5.8, 0.), (EDGE_X, 6., height)),
    }


def build_pit_meshes(height):
    meshes = []
    for name, bounds in pit_boxes(height).items():
        lower, upper = np.asarray(bounds)
        transform = np.eye(4)
        transform[:3, 3] = (upper + lower) / 2
        mesh = trimesh.creation.box(extents=upper-lower, transform=transform)
        mesh.metadata["pit_part"] = name
        meshes.append(mesh)
    return meshes, np.asarray(ORIGIN)


def pit_terrain(difficulty, cfg):
    if tuple(cfg.size) != SIZE:
        raise ValueError("Pit MDP requires an 8 x 6 m tile")
    return build_pit_meshes(height_for_level(level_from_difficulty(difficulty)))
