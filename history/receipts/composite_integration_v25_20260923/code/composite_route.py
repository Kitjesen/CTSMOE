"""Explicit slope-to-stairs surfaces for isolated Thunder terrain experiments.

Coordinates are in the uncentered Isaac sub-terrain frame: x is forward,
y spans the tile, and the returned origin is the robot's flat starting point.
Isaac's TerrainGenerator later subtracts half the tile size from both mesh and
origin. The surface is open, as in Isaac Lab's height-field triangle meshes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import trimesh


@dataclass(frozen=True)
class RouteSpec:
    size_x_m: float
    size_y_m: float
    spawn_x_m: float
    approach_end_x_m: float
    slope_end_x_m: float
    transition_end_x_m: float
    slope_angle_deg: float = 10.0
    stair_count: int = 4
    stair_height_m: float = 0.08
    stair_tread_m: float = 0.31
    tile_border_m: float = 0.5

    def __post_init__(self) -> None:
        numbers = (
            self.size_x_m, self.size_y_m, self.spawn_x_m,
            self.approach_end_x_m, self.slope_end_x_m,
            self.transition_end_x_m, self.slope_angle_deg,
            self.stair_height_m, self.stair_tread_m, self.tile_border_m,
        )
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Route parameters must be finite")
        if not (0 < self.tile_border_m < self.spawn_x_m < self.approach_end_x_m
                < self.slope_end_x_m < self.transition_end_x_m < self.final_start_x_m
                < self.size_x_m - self.tile_border_m):
            raise ValueError("Route stations or spawn fall outside the usable tile")
        if self.size_y_m <= 2 * self.tile_border_m:
            raise ValueError("Route has no usable lateral width")
        if not (0 < self.slope_angle_deg <= 20 and self.stair_count == 4
                and 0 < self.stair_height_m <= 0.12 and self.stair_tread_m >= 0.31):
            raise ValueError("Route exceeds this first-stage geometry contract")
        if self.size_x_m - self.tile_border_m - self.final_start_x_m < 1.5:
            raise ValueError("Final usable platform is shorter than 1.5 m")

    @property
    def slope_top_m(self) -> float:
        return (self.slope_end_x_m - self.approach_end_x_m) * math.tan(
            math.radians(self.slope_angle_deg))

    @property
    def final_start_x_m(self) -> float:
        return self.transition_end_x_m + self.stair_count * self.stair_tread_m

    @property
    def final_height_m(self) -> float:
        return self.slope_top_m + self.stair_count * self.stair_height_m

    @property
    def final_usable_length_m(self) -> float:
        return self.size_x_m - self.tile_border_m - self.final_start_x_m


IDEAL_12X9 = RouteSpec(
    size_x_m=12.0, size_y_m=9.0, spawn_x_m=1.0,
    approach_end_x_m=2.0, slope_end_x_m=4.0,
    transition_end_x_m=5.2,
)

COMPACT_9X9 = RouteSpec(
    size_x_m=9.0, size_y_m=9.0, spawn_x_m=1.1,
    approach_end_x_m=2.1, slope_end_x_m=3.9,
    transition_end_x_m=4.9,
)

GEOMETRY_LEVELS = 10


def spec_for_level(level: int, base: RouteSpec = COMPACT_9X9) -> RouteSpec:
    """Bounded independent 0..9 route curriculum; stations never move."""
    if not isinstance(level, int) or not 0 <= level < GEOMETRY_LEVELS:
        raise ValueError("Composite difficulty level must be an integer 0..9")
    return replace(base, slope_angle_deg=10.0 + 10.0 * level / 9,
                   stair_height_m=0.08 + 0.04 * level / 9)


def profile_stations(spec: RouteSpec) -> np.ndarray:
    """Piecewise x,z top profile; duplicate x marks a vertical 8 cm riser."""
    profile = [
        (0.0, 0.0),
        (spec.approach_end_x_m, 0.0),
        (spec.slope_end_x_m, spec.slope_top_m),
        (spec.transition_end_x_m, spec.slope_top_m),
    ]
    for step in range(spec.stair_count):
        # Reuse the prior tread endpoint bit-for-bit. Recomputing the same X
        # through another arithmetic path can create a microscopic sloped face
        # instead of an exactly vertical riser in the exported collision mesh.
        riser_x = profile[-1][0]
        height = spec.slope_top_m + (step + 1) * spec.stair_height_m
        profile.append((riser_x, height))
        profile.append((spec.transition_end_x_m + (step + 1) * spec.stair_tread_m, height))
    profile.append((spec.size_x_m, spec.final_height_m))
    return np.asarray(profile, dtype=np.float64)


def build_route_mesh(spec: RouteSpec) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Create an exact strip mesh with sloped faces and vertical stair risers.

    No height-field threshold approximates the risers into diagonal ramps.
    The cross-section is identical at y=0, y=mid, and y=tile width.
    """
    profile = profile_stations(spec)
    y_stations = np.asarray((0.0, spec.size_y_m / 2, spec.size_y_m))
    vertices = np.asarray(
        [(x, y, z) for x, z in profile for y in y_stations], dtype=np.float64)
    faces = []
    ny = len(y_stations)
    for i in range(len(profile) - 1):
        for j in range(ny - 1):
            low, high = i * ny + j, (i + 1) * ny + j
            faces.extend(((low, high, high + 1), (low, high + 1, low + 1)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    mesh.metadata["route_geometry"] = "slope_transition_four_stairs_platform_v1"
    origin = np.asarray((spec.spawn_x_m, spec.size_y_m / 2, 0.0))
    return [mesh], origin


def _terrain(difficulty: float, cfg: object, spec: RouteSpec):
    if tuple(getattr(cfg, "size", ())) != (spec.size_x_m, spec.size_y_m):
        raise ValueError("Isaac sub-terrain size does not match route geometry")
    if not math.isfinite(float(difficulty)) or not 0 <= difficulty <= 1:
        raise ValueError("Difficulty must be finite and in [0, 1]")
    # MixedTerrainGenerator passes the raw 22-row difficulty to this function.
    # The first ten rows are the bounded route course; the spare atlas rows
    # duplicate L9 and are never assigned route environments.
    atlas_row = min(int(float(difficulty) * 22), 21)
    return build_route_mesh(spec_for_level(min(atlas_row, 9), spec))


def composite_terrain_12x9(difficulty: float, cfg: object):
    """Isaac SubTerrainBaseCfg(function=...) entry point for a 12x9 atlas."""
    return _terrain(difficulty, cfg, IDEAL_12X9)


def composite_terrain_9x9(difficulty: float, cfg: object):
    """Isaac SubTerrainBaseCfg(function=...) entry point for the V23 9x9 atlas."""
    return _terrain(difficulty, cfg, COMPACT_9X9)


def world_to_tile_xy(points_world_xy, env_origin_world_xy, spec: RouteSpec):
    """Translate world points into route coordinates; the route always faces +X.

    Isaac env_origin is the returned spawn origin after atlas centering. Do not
    rotate by robot yaw: that would make a drifted robot appear on-route.
    """
    points = np.asarray(points_world_xy, dtype=np.float64)
    origin = np.asarray(env_origin_world_xy, dtype=np.float64)
    if points.shape[-1] != 2 or origin.shape != (2,) or not np.isfinite(points).all():
        raise ValueError("Expected finite XY points and one XY environment origin")
    return points - origin + np.asarray((spec.spawn_x_m, spec.size_y_m / 2))
