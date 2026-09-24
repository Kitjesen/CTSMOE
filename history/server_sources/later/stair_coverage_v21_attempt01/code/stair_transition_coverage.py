"""Geometry-only coverage for two existing stair columns; NumPy helpers are CPU-only."""
from __future__ import annotations

import hashlib

import numpy as np

PROFILE = 'stair_terminal_columns_v21'
COLUMNS = {'stairs_up': (8,), 'stairs_down': (11,)}
ROWS = tuple(range(22))
INNER_RADIUS_M = 2.7
EXPECTED_COLUMNS = (
    'wave', 'slope_up', 'slope_up', 'slope_down', 'slope_down', 'rough_slope',
    *('stairs_up',) * 5, *('stairs_down',) * 2, *('obstacles',) * 4,
    *('flat',) * 3, *('pit',) * 20,
)
AREA_EPS = 1e-10


def _hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array)
        value = np.ascontiguousarray(value, dtype='<i8' if value.dtype.kind in 'iu' else '<f8')
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.dtype.str.encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def mesh_hash(vertices, faces):
    """Order-sensitive canonical numeric hash, before atlas placement."""
    return _hash(vertices, faces)


def _areas(vertices, faces):
    triangle = vertices[faces]
    return np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0],
                                   triangle[:, 2] - triangle[:, 0]), axis=1) / 2


def _profile(vertices, axis, sign):
    """Measured cardinal surface envelope, compressed into constant-height runs."""
    other = 1 - axis
    nearest = np.min(np.abs(vertices[:, other]))
    subset = vertices[np.isclose(np.abs(vertices[:, other]), nearest, atol=1e-7, rtol=0)]
    distance = sign * subset[:, axis]
    subset, distance = subset[distance >= -1e-7], distance[distance >= -1e-7]
    x, inverse = np.unique(np.round(distance, 7), return_inverse=True)
    z = np.full(len(x), -np.inf)
    np.maximum.at(z, inverse, subset[:, 2])
    boundaries = np.r_[0, np.flatnonzero(np.abs(np.diff(z)) > 1e-6) + 1, len(x)]
    runs = [dict(from_m=float(x[a]), to_m=float(x[b-1]), z_m=float(z[a]))
            for a, b in zip(boundaries[:-1], boundaries[1:])]
    inside = z[x < INNER_RADIUS_M]
    jumps = np.abs(np.diff(inside))
    return dict(lateral_offset_abs_m=float(nearest), surface_runs=runs,
                inner_riser_height_m=sorted(set(np.round(jumps[jumps > 1e-6], 7).tolist())))


def cap_mesh_arrays(vertices, faces, kind):
    """Apply the frozen phase_landing cap with exact inner-geometry evidence.

    Keeps original inner vertex positions and nondegenerate triangles wholly within
    radius 2.7 m. It shortens the flight; it does not preserve its total rise.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if kind not in COLUMNS:
        raise ValueError('Only stairs_up/stairs_down can receive a landing')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Expected finite (N,3) vertices')
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Expected valid integer triangular faces')
    if not np.allclose(vertices[:, :2].min(0), [-4.5, -4.5], atol=.2, rtol=0):
        raise ValueError('Expected centered original 9m stair tile lower bounds')
    if not np.allclose(vertices[:, :2].max(0), [4.5, 4.5], atol=.2, rtol=0):
        raise ValueError('Expected centered original 9m stair tile upper bounds')
    radius = np.abs(vertices[:, :2]).max(-1)
    band = (radius >= 2.7) & (radius <= 2.9)
    values, counts = np.unique(np.round(vertices[band, 2], 5), return_counts=True)
    if not len(values) or not np.any(radius < 1.):
        raise ValueError('Missing landing band or center surface')
    cap, center = float(values[counts.argmax()]), float(np.median(vertices[radius < 1., 2]))
    if (kind == 'stairs_up' and cap <= center) or (kind == 'stairs_down' and cap >= center):
        raise ValueError('Stair direction does not match outward traversal')
    changed = vertices.copy()
    changed[:, 2] = np.minimum(vertices[:, 2], cap) if kind == 'stairs_up' else np.maximum(vertices[:, 2], cap)
    if np.array_equal(changed, vertices):
        raise ValueError('Selected tile has no outer stair geometry to cap')
    inner = radius < INNER_RADIUS_M
    if not np.array_equal(vertices[inner], changed[inner]):
        raise ValueError('Landing would change the protected inner stair vertices')
    if not np.array_equal(vertices[:, :2], changed[:, :2]):
        raise ValueError('Landing changed XY geometry')
    keep = _areas(changed, faces) > AREA_EPS
    inner_faces = inner[faces].all(-1) & (_areas(vertices, faces) > AREA_EPS)
    if not np.all(keep[inner_faces]):
        raise ValueError('Landing removed an inner nondegenerate triangle')
    kept_faces = faces[keep]
    used, inverse = np.unique(kept_faces.reshape(-1), return_inverse=True)
    output_vertices, output_faces = changed[used].copy(), inverse.reshape(-1, 3)
    before_inner = _hash(vertices[faces[inner_faces]])
    after_inner = _hash(output_vertices[output_faces[inner_faces[keep]]])
    if before_inner != after_inner:
        raise ValueError('Protected inner triangle coordinates changed')
    protected_vertex_ids = np.unique(faces[inner_faces])
    actual_inner_vertices = output_vertices[np.searchsorted(used, protected_vertex_ids)]
    directions = [('positive_x', 0, 1), ('negative_x', 0, -1),
                  ('positive_y', 1, 1), ('negative_y', 1, -1)]
    receipt = dict(
        kind=kind, coordinate_frame='centered tile, before atlas placement',
        before_sha256=mesh_hash(vertices, faces), after_sha256=mesh_hash(output_vertices, output_faces),
        inner_radius_strict_less_m=INNER_RADIUS_M,
        inner_vertices_before_sha256=_hash(vertices[inner]),
        inner_vertices_after_sha256=_hash(changed[inner]),
        inner_triangle_vertex_positions_before_sha256=_hash(vertices[protected_vertex_ids]),
        inner_triangle_vertex_positions_after_sha256=_hash(actual_inner_vertices),
        inner_nondegenerate_triangles_before_sha256=before_inner,
        inner_nondegenerate_triangles_after_sha256=after_inner,
        inner_vertices_unchanged=True, inner_nondegenerate_triangles_unchanged=True, xy_unchanged=True,
        cap_z_m=cap, center_z_m=center, cap_minus_center_m=cap-center,
        cap_selection_radius_m=[2.7, 2.9],
        changed_vertex_count=int(np.count_nonzero(vertices[:, 2] != changed[:, 2])),
        removed_degenerate_faces=int((~keep).sum()),
        before_vertex_count=len(vertices), after_vertex_count=len(output_vertices),
        before_face_count=len(faces), after_face_count=len(output_faces),
        xy_bounds_m=[vertices[:, :2].min(0).tolist(), vertices[:, :2].max(0).tolist()],
        geometry_before={name: _profile(vertices, axis, sign) for name, axis, sign in directions},
        geometry_after={name: _profile(output_vertices, axis, sign) for name, axis, sign in directions},
        difficulty_scope='original row/riser/tread retained inside protected region; outer steps and total rise reduced',
    )
    return output_vertices, output_faces, receipt


def column_names(cfg):
    """Use the pinned curriculum generator's proportion-to-column rule."""
    names = list(cfg.sub_terrains)
    values = np.asarray([cfg.sub_terrains[name].proportion for name in names], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError('Invalid terrain proportions')
    cumulative = np.cumsum(values / values.sum())
    return tuple(names[int(np.flatnonzero(col / cfg.num_cols + .001 < cumulative)[0])]
                 for col in range(cfg.num_cols))


def validate_config(cfg):
    if (tuple(cfg.size) != (9., 9.) or cfg.num_rows != 22 or cfg.num_cols != 40
            or not getattr(cfg, 'curriculum', False)):
        raise ValueError('Configure coverage after configure_mixed: expected curriculum 22x40 atlas of 9m tiles')
    if getattr(cfg, 'sub_terrain_border_width', None) != .5:
        raise ValueError('Expected original 0.5m sub-terrain border')
    if column_names(cfg) != EXPECTED_COLUMNS:
        raise ValueError('Pinned nine-terrain column mapping changed')
    for kind in COLUMNS:
        sub = cfg.sub_terrains[kind]
        if (tuple(sub.step_height_range) != (.05, .257) or sub.step_width != .31
                or sub.platform_width != 3.):
            raise ValueError('Original stair riser/tread/center-platform configuration changed')
    if any(getattr(sub, 'phase_landing_kind', None) for sub in cfg.sub_terrains.values()):
        raise ValueError('Existing global phase_landing cap would duplicate the coverage transform')
    return dict(profile=PROFILE, columns_by_kind={k: list(v) for k, v in COLUMNS.items()},
                generated_rows=list(ROWS), active_original_rows=list(range(10)),
                inactive_original_rows=list(range(10, 22)),
                expected_column_names=list(EXPECTED_COLUMNS),
                geometry_only=True, rewards_commands_terminations_changed=False,
                inner_radius_m=INNER_RADIUS_M, cap_radius_band_m=[2.7, 2.9])


class _CoverageMixin:
    def __init__(self, cfg, device='cpu'):
        self.coverage_config = validate_config(cfg)
        self.coverage_mesh_receipts = []
        super().__init__(cfg, device=device)
        if len(self.coverage_mesh_receipts) != cfg.num_rows * cfg.num_cols:
            raise RuntimeError('Did not audit every generated tile')
        changed = {(r['row'], r['column']) for r in self.coverage_mesh_receipts if r['selected']}
        expected = {(row, col) for row in ROWS for columns in COLUMNS.values() for col in columns}
        if changed != expected:
            raise RuntimeError('Generated coverage cells differ from the explicit whitelist')
        cfg.stair_transition_coverage_receipts = self.coverage_mesh_receipts

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        if row not in ROWS or not 0 <= col < len(EXPECTED_COLUMNS):
            raise ValueError('Unexpected terrain cell')
        kind = EXPECTED_COLUMNS[col]
        if self.cfg.sub_terrains[kind] is not sub_terrain_cfg:
            raise ValueError('Sub-terrain config does not match the pinned column')
        selected = col in COLUMNS.get(kind, ())
        before = mesh_hash(mesh.vertices, mesh.faces)
        receipt = dict(row=int(row), column=int(col), kind=kind, selected=selected,
                       effective_original_level=min(int(row), 9) if kind != 'pit' else int(row),
                       active_original_row=row < 10, origin_before=np.asarray(origin).tolist())
        if selected:
            vertices, faces, evidence = cap_mesh_arrays(mesh.vertices, mesh.faces, kind)
            mesh = mesh.copy()
            mesh.vertices, mesh.faces = vertices, faces
            receipt.update(evidence)
        else:
            receipt.update(before_sha256=before, after_sha256=mesh_hash(mesh.vertices, mesh.faces))
            if receipt['before_sha256'] != receipt['after_sha256']:
                raise RuntimeError('Nonselected geometry changed')
        receipt['origin_after'] = np.asarray(origin).tolist()
        receipt['origin_unchanged'] = True
        self.coverage_mesh_receipts.append(receipt)
        return super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)


def configure(cfg):
    """Call immediately AFTER configure_mixed and BEFORE constructing Go2Env."""
    from mixed_task import MixedTerrainGenerator

    terrain = cfg.scene.terrain.terrain_generator
    contract = validate_config(terrain)
    if terrain.class_type is not MixedTerrainGenerator:
        raise ValueError('Coverage must wrap the original MixedTerrainGenerator exactly once')
    runtime_type = type('StairTransitionCoverageTerrainGenerator',
                        (_CoverageMixin, MixedTerrainGenerator), {'__module__': __name__})
    # Make the configured class resolve by its module/name when configuration is serialized.
    globals()[runtime_type.__name__] = runtime_type
    terrain.class_type = runtime_type
    terrain.stair_transition_coverage = contract
    return contract


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def coverage_counts(levels, columns, learner):
    """Actual restored assignment counts; does not mutate or reassign environments."""
    levels, columns, learner = (_numpy(value) for value in (levels, columns, learner))
    if (levels.ndim != 1 or levels.shape != columns.shape or levels.shape != learner.shape
            or any(value.dtype.kind not in 'iu' for value in (levels, columns, learner))
            or np.any((levels < 0) | (levels >= 22)) or np.any((columns < 0) | (columns >= 40))
            or np.any((learner < 0) | (learner > 1))):
        raise ValueError('Expected aligned integer level/column/learner assignment arrays')
    types = np.asarray(EXPECTED_COLUMNS)[columns]
    if np.any((types != 'pit') & (levels > 9)):
        raise ValueError('Original terrain assigned to inactive level above 9')
    selected = np.isin(columns, [col for values in COLUMNS.values() for col in values])
    result = {}
    for value, label in ((0, 'teacher'), (1, 'student')):
        group = learner == value
        by_type = {}
        for kind in dict.fromkeys(EXPECTED_COLUMNS):
            mask = group & (types == kind)
            covered = mask & selected
            by_type[kind] = dict(total=int(mask.sum()), coverage=int(covered.sum()),
                                 original=int((mask & ~selected).sum()),
                                 coverage_by_level=np.bincount(levels[covered], minlength=22).tolist())
        result[label] = dict(total=int(group.sum()), coverage=int((group & selected).sum()), by_type=by_type)
    return dict(total=len(levels), coverage_total=int(selected.sum()),
                fraction=float(selected.mean()) if len(levels) else 0., by_learner=result)


def snapshot(env):
    """Call after restored terrain assignments and learner IDs are installed."""
    terrain = env.scene.terrain
    cfg = terrain.cfg.terrain_generator
    command = env.command_manager.get_term('base_velocity')
    receipts = cfg.stair_transition_coverage_receipts
    counts = coverage_counts(terrain.terrain_levels, terrain.terrain_types, command.learner)
    return dict(profile=PROFILE, config=cfg.stair_transition_coverage, assignments=counts,
                generated_cell_count=len(receipts), selected_cell_count=sum(r['selected'] for r in receipts),
                mesh_receipts=receipts,
                scope='geometry and actual environment coverage only; not a behavior or success claim')
