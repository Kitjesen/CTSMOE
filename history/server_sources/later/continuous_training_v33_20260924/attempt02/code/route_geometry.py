"""Watertight CPU collision meshes for six continuous-route curricula.

Every course is one closed height-field solid.  In particular, asymmetric
courses do not concatenate independently closed left and right strips, so
there are no coincident walls at the lane centreline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import trimesh


CELL_LENGTH_M = 9.0
CELL_WIDTH_M = 9.0
SPAWN = (-3.5, 0.0, 0.0)
FINISH_X_M = 3.3
EXIT_BOUND_X_M = 4.4
BOTTOM_Z_M = -1.5
RISER_RUN_M = 0.03
FAMILIES = (
    "continuous_curbs",
    "continuous_blocks",
    "continuous_ramp_stairs",
    "continuous_split_height",
    "continuous_cross_bumps",
    "continuous_split_route",
)


@dataclass(frozen=True)
class TerrainSpec:
    family: str
    level: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown terrain family: {self.family}")
        if not isinstance(self.level, int) or isinstance(self.level, bool) or not 0 <= self.level <= 9:
            raise ValueError("level must be an integer from 0 through 9")


def _ramp01(value: np.ndarray, start: float, end: float) -> np.ndarray:
    if end <= start:
        raise ValueError("ramp end must follow its start")
    return np.clip((value - start) / (end - start), 0.0, 1.0)


def _window(value: np.ndarray, start: float, rise: float, end: float, fall: float) -> np.ndarray:
    return np.minimum(_ramp01(value, start, start + rise), 1.0 - _ramp01(value, end, end + fall))


def _event(x: float, name: str, height: float, **values: float | int | str) -> dict:
    item = {"x_m": float(x), "name": name, "height_m": float(height)}
    item.update(values)
    return item


def _course_surface(spec: TerrainSpec, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[dict], dict]:
    p = spec.level / 9.0
    events: list[dict] = []
    targets: dict[str, float | int | str] = {}

    if spec.family == "continuous_curbs":
        height = 0.08 + 0.32 * p
        z = np.zeros_like(x)
        for index, start in enumerate((-1.9, 0.2)):
            end = start + 0.85
            z += height * _window(x, start, RISER_RUN_M, end, RISER_RUN_M)
            angle = math.degrees(math.atan2(height, RISER_RUN_M))
            events.extend((_event(start, f"curb_up_{index}", height, actual_slope_deg=angle),
                           _event(end, f"curb_drop_{index}", -height, actual_slope_deg=-angle)))
        events.append(_event(end + RISER_RUN_M, "route_end", 0.0))
        targets = {"count": 2, "target_max_height_m": 0.4}

    elif spec.family == "continuous_blocks":
        rng = np.random.default_rng(spec.seed)
        peak = 0.09 + 0.31 * p
        ratios = np.asarray((1.0, rng.uniform(0.68, 0.9)))
        lengths = rng.uniform(0.75, 1.0, 2)
        starts = (-1.9, 0.25)
        z = np.zeros_like(x)
        for index, (start, length, ratio) in enumerate(zip(starts, lengths, ratios)):
            height = peak * ratio
            end = start + float(length)
            z += height * _window(x, start, RISER_RUN_M, end, RISER_RUN_M)
            angle = math.degrees(math.atan2(height, RISER_RUN_M))
            events.extend((_event(start, f"block_up_{index}", height, actual_slope_deg=angle,
                                  length_m=float(length)),
                           _event(end, f"block_drop_{index}", -height, actual_slope_deg=-angle)))
        events.append(_event(end + RISER_RUN_M, "route_end", 0.0))
        targets = {"count": 2, "target_max_height_m": 0.4, "seed": spec.seed}

    elif spec.family == "continuous_ramp_stairs":
        stair = 0.06 + 0.24 * p
        requested_slope = 15.0 + 30.0 * p
        run = stair / math.tan(math.radians(requested_slope))
        start = -2.0
        stair_one = start + run + 0.35
        stair_two = stair_one + 0.48
        descent = stair_two + 0.65
        finish = descent + 3.0 * stair / math.tan(math.radians(requested_slope))
        z = stair * _ramp01(x, start, start + run)
        z += stair * _ramp01(x, stair_one, stair_one + RISER_RUN_M)
        z += stair * _ramp01(x, stair_two, stair_two + RISER_RUN_M)
        z -= 3.0 * stair * _ramp01(x, descent, finish)
        riser_angle = math.degrees(math.atan2(stair, RISER_RUN_M))
        events = [_event(start, "up_ramp_start", 0.0, actual_slope_deg=requested_slope),
                  _event(stair_one, "stair_0", stair, actual_slope_deg=riser_angle),
                  _event(stair_two, "stair_1", stair, actual_slope_deg=riser_angle),
                  _event(descent, "down_ramp_start", 3.0 * stair, actual_slope_deg=-requested_slope),
                  _event(finish, "route_end", 0.0)]
        targets = {"count": 2, "target_stair_height_m": 0.3, "target_ramp_slope_deg": 45.0}

    elif spec.family == "continuous_split_height":
        difference = 0.03 + 0.17 * p
        left_section = _window(x, -2.0, 0.35, -0.45, 0.35)
        right_section = _window(x, 0.1, 0.35, 1.65, 0.35)
        # A 32 cm centre transition gives wheels at Y=+/-0.25 m the full
        # commanded height difference while keeping one connected top sheet.
        lateral = _ramp01(y, -0.16, 0.16)
        z = difference * (left_section * lateral + right_section * (1.0 - lateral))
        cross_angle = math.degrees(math.atan2(difference, 0.32))
        events = [_event(-2.0, "height_difference_entry", 0.0),
                  _event(-1.65, "left_high_section", difference, side="left",
                         actual_cross_slope_deg=cross_angle),
                  _event(0.45, "right_high_section", difference, side="right",
                         actual_cross_slope_deg=-cross_angle),
                  _event(2.0, "route_end", 0.0)]
        targets = {"target_left_right_difference_m": 0.2, "sides_in_route": "left then right",
                   "wheel_track_y_m": [-0.25, 0.25]}

    elif spec.family == "continuous_cross_bumps":
        cross_slope = 4.0 + 10.0 * p
        bump_height = 0.035 + 0.115 * p
        longitudinal = _window(x, -2.0, 0.4, 1.8, 0.4)
        z = np.tan(np.deg2rad(cross_slope)) * np.clip(y, -1.5, 1.5) * longitudinal
        bump_centres = ((-1.0, 0.25), (0.1, -0.25), (1.1, 0.25))
        for bx, by in bump_centres:
            radius_x, radius_y = 0.42, 0.24
            inside = ((x - bx) / radius_x) ** 2 + ((y - by) / radius_y) ** 2
            bump = np.where(inside < 1.0, bump_height * 0.5 * (1.0 + np.cos(np.pi * np.sqrt(np.minimum(inside, 1.0)))), 0.0)
            z += bump
            events.append(_event(bx, "smooth_bump", bump_height, y_m=by))
        events[:0] = [_event(-2.0, "cross_slope_entry", 0.0, actual_cross_slope_deg=cross_slope)]
        events.append(_event(2.2, "route_end", 0.0))
        targets = {"target_cross_slope_deg": 14.0, "target_bump_height_m": 0.15, "count": 3}

    else:  # continuous_split_route
        stair = 0.06 + 0.24 * p
        ramp_slope = 15.0 + 30.0 * p
        total = 2.0 * stair
        ramp_run = total / math.tan(math.radians(ramp_slope))
        left_start = -2.0
        left_down = 0.6
        left = total * _window(x, left_start, ramp_run, left_down, ramp_run)
        right_start = -1.5  # 50 cm longitudinal offset is deliberate.
        second = right_start + 0.55
        right_down_0 = 0.65
        right_down_1 = right_down_0 + 0.55
        right = stair * _ramp01(x, right_start, right_start + RISER_RUN_M)
        right += stair * _ramp01(x, second, second + RISER_RUN_M)
        right -= stair * _ramp01(x, right_down_0, right_down_0 + RISER_RUN_M)
        right -= stair * _ramp01(x, right_down_1, right_down_1 + RISER_RUN_M)
        blend = _ramp01(y, -0.16, 0.16)
        z = right * (1.0 - blend) + left * blend
        riser_angle = math.degrees(math.atan2(stair, RISER_RUN_M))
        events = [_event(left_start, "left_ramp_start", 0.0, side="left", actual_slope_deg=ramp_slope),
                  _event(right_start, "right_stair_0", stair, side="right", actual_slope_deg=riser_angle),
                  _event(second, "right_stair_1", stair, side="right", actual_slope_deg=riser_angle),
                  _event(left_down, "left_ramp_down", total, side="left", actual_slope_deg=-ramp_slope),
                  _event(right_down_0, "right_stair_down_0", -stair, side="right", actual_slope_deg=-riser_angle),
                  _event(right_down_1 + RISER_RUN_M, "route_end", 0.0)]
        targets = {"count": 2, "offset_m": 0.5, "target_stair_height_m": 0.3,
                   "target_ramp_slope_deg": 45.0}

    return z, events, targets


def _critical_x(spec: TerrainSpec) -> list[float]:
    """Return exact profile breaks so short risers survive the coarse base grid."""
    p = spec.level / 9.0
    if spec.family == "continuous_curbs":
        starts = (-1.9, 0.2)
        ends = tuple(start + 0.85 for start in starts)
        return [value for pair in zip(starts, ends) for value in pair]
    if spec.family == "continuous_blocks":
        rng = np.random.default_rng(spec.seed)
        rng.uniform(0.68, 0.9)
        lengths = rng.uniform(0.75, 1.0, 2)
        starts = (-1.9, 0.25)
        return [value for pair in zip(starts, (start + float(length) for start, length in zip(starts, lengths)))
                for value in pair]
    if spec.family == "continuous_ramp_stairs":
        stair = 0.06 + 0.24 * p
        slope = 15.0 + 30.0 * p
        start = -2.0
        ramp_end = start + stair / math.tan(math.radians(slope))
        stair_one = ramp_end + 0.35
        stair_two = stair_one + 0.48
        descent = stair_two + 0.65
        finish = descent + 3.0 * stair / math.tan(math.radians(slope))
        return [start, ramp_end, stair_one, stair_two, descent, finish]
    if spec.family == "continuous_split_route":
        stair = 0.06 + 0.24 * p
        slope = 15.0 + 30.0 * p
        run = 2.0 * stair / math.tan(math.radians(slope))
        return [-2.0, -2.0 + run, 0.6, 0.6 + run, -1.5, -0.95, 0.65, 1.2]
    return [-2.0, -1.65, -1.0, -0.45, 0.1, 0.45, 1.0, 1.65, 1.8, 2.0, 2.2]


def _axes(spec: TerrainSpec) -> tuple[np.ndarray, np.ndarray]:
    # A fixed CPU grid makes all families share the same integration contract.
    # 2 cm samples around each nominal event preserve the 3 cm collision risers.
    base_x = np.linspace(-4.5, 4.5, 46)
    event_x = np.asarray([-2.0, -1.9, -1.65, -1.5, -1.0, -0.95, -0.45, 0.1,
                          0.2, 0.25, 0.45, 0.6, 0.65, 1.05, 1.1, 1.2, 1.65, 1.8, 2.0, 2.2])
    critical = np.asarray(_critical_x(spec))
    fine = np.concatenate((event_x, event_x + RISER_RUN_M,
                           critical, critical + RISER_RUN_M))
    xs = np.unique(np.round(np.concatenate((base_x, fine)), 8))
    ys = np.unique(np.round(np.concatenate((np.linspace(-4.5, 4.5, 46),
                                            [-0.25, -0.16, 0.0, 0.16, 0.25])), 8))
    return xs, ys


def _solid(xs: np.ndarray, ys: np.ndarray, top: np.ndarray) -> tuple[trimesh.Trimesh, int]:
    nx, ny = len(xs), len(ys)
    top_vertices = np.column_stack((np.repeat(xs, ny), np.tile(ys, nx), top.reshape(-1)))
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 2] = BOTTOM_Z_M
    vertices = np.vstack((top_vertices, bottom_vertices))
    top_faces: list[tuple[int, int, int]] = []
    bottom_faces: list[tuple[int, int, int]] = []
    side_faces: list[tuple[int, int, int]] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b, c, d = a + ny, a + ny + 1, a + 1
            top_faces.extend(((a, b, c), (a, c, d)))
            bottom_faces.extend(((a + nx * ny, c + nx * ny, b + nx * ny),
                                 (a + nx * ny, d + nx * ny, c + nx * ny)))

    boundary = ([j for j in range(ny)]
                + [i * ny + ny - 1 for i in range(1, nx)]
                + [(nx - 1) * ny + j for j in range(ny - 2, -1, -1)]
                + [i * ny for i in range(nx - 2, 0, -1)])
    offset = nx * ny
    for index, a in enumerate(boundary):
        b = boundary[(index + 1) % len(boundary)]
        side_faces.extend(((a, b + offset, a + offset), (a, b, b + offset)))

    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=np.asarray(top_faces + bottom_faces + side_faces),
                           process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0.0:
        raise ValueError("terrain must be one closed, consistently oriented positive solid")
    if mesh.area_faces.min() <= 1e-10:
        raise ValueError("terrain contains a degenerate collision triangle")
    return mesh, len(top_faces)


def generate(spec: TerrainSpec) -> tuple[trimesh.Trimesh, dict]:
    """Generate one CPU mesh and truthful collision metadata."""
    xs, ys = _axes(spec)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    top, events, targets = _course_surface(spec, grid_x, grid_y)
    mesh, top_face_count = _solid(xs, ys, top)

    top_normals = mesh.face_normals[:top_face_count]
    slopes = np.degrees(np.arccos(np.clip(top_normals[:, 2], -1.0, 1.0)))
    surface_min = float(top.min())
    surface_max = float(top.max())
    metadata = {
        "config": asdict(spec),
        "units": "m",
        "up_axis": "Z",
        "forward_axis": "X",
        "cell_size_m": [CELL_LENGTH_M, CELL_WIDTH_M],
        "spawn": list(SPAWN),
        "final_obstacle_x": float(max(event["x_m"] for event in events)),
        "finish_x": FINISH_X_M,
        "exit_bound_x": EXIT_BOUND_X_M,
        "cell_bounds": [[-4.5, -4.5], [4.5, 4.5]],
        "atlas_columns": list(range(20, 26)),
        "route_events": events,
        "collision": {
            "mesh_contract": "single closed height-field solid; external boundary only",
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "positive_volume": bool(mesh.volume > 0.0),
            "surface_min_height_m": surface_min,
            "surface_max_height_m": surface_max,
            "surface_height_range_m": surface_max - surface_min,
            "max_surface_triangle_slope_deg": float(slopes.max()),
            "riser_model": "finite 3 cm chamfer; not an exact vertical face",
            "riser_run_m": RISER_RUN_M,
            "has_exact_vertical_risers": False,
            "triangle_count": int(len(mesh.faces)),
        },
        "design_targets_at_level_9": targets,
        "training_integrated": False,
    }
    return mesh, metadata


def mesh_for(family: str, level: int, *, seed: int = 0) -> tuple[trimesh.Trimesh, dict]:
    """Runtime-facing constructor for one of the 60 dedicated atlas cells."""
    return generate(TerrainSpec(family=family, level=level, seed=seed))
