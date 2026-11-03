import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import stair_transition_coverage as coverage


def mesh_arrays(sign=1):
    axis = np.linspace(-4.5, 4.5, 91)
    x, y = np.meshgrid(axis, axis)
    radius = np.maximum(np.abs(x), np.abs(y))
    z = sign * np.maximum(np.floor((radius - 1.5) / .31), 0.) * .2
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    lower = (np.arange(90)[:, None] * 91 + np.arange(90)[None]).ravel()
    faces = np.vstack((np.column_stack((lower, lower + 1, lower + 91)),
                       np.column_stack((lower + 1, lower + 92, lower + 91))))
    return vertices, faces


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices, self.faces = vertices, faces

    def copy(self):
        return copy.deepcopy(self)


def config():
    names, counts = np.unique(coverage.EXPECTED_COLUMNS, return_counts=True)
    sizes = dict(zip(names, counts.tolist()))
    subs = {name: SimpleNamespace(proportion=sizes[name] / 40, step_height_range=(.05, .257),
                                  step_width=.31, platform_width=3.)
            for name in dict.fromkeys(coverage.EXPECTED_COLUMNS)}
    return SimpleNamespace(size=(9., 9.), num_rows=22, num_cols=40, curriculum=True,
                           sub_terrain_border_width=.5, sub_terrains=subs)


@pytest.mark.parametrize(('kind', 'sign'), [('stairs_up', 1), ('stairs_down', -1)])
def test_real_mesh_arrays_preserve_inner_geometry_and_record_actual_platform(kind, sign):
    vertices, faces = mesh_arrays(sign)
    original = vertices.copy()
    output, output_faces, receipt = coverage.cap_mesh_arrays(vertices, faces, kind)
    np.testing.assert_array_equal(vertices, original)
    assert receipt['before_sha256'] != receipt['after_sha256']
    assert receipt['inner_vertices_before_sha256'] == receipt['inner_vertices_after_sha256']
    assert receipt['inner_nondegenerate_triangles_before_sha256'] == receipt['inner_nondegenerate_triangles_after_sha256']
    assert receipt['changed_vertex_count'] > 0
    assert receipt['after_sha256'] == coverage.mesh_hash(output, output_faces)
    assert sign * receipt['cap_minus_center_m'] > 0
    for direction, profile in receipt['geometry_after'].items():
        assert profile['inner_riser_height_m'] == [.2]
        assert profile['surface_runs'][-1]['z_m'] == receipt['cap_z_m']
        assert profile['surface_runs'][-1]['to_m'] == 4.5
        assert receipt['geometry_before'][direction]['inner_riser_height_m'] == profile['inner_riser_height_m']


def test_matches_frozen_phase_landing_height_cap_without_importing_isaac():
    vertices, faces = mesh_arrays()
    output, _, receipt = coverage.cap_mesh_arrays(vertices, faces, 'stairs_up')
    radius = np.abs(vertices[:, :2]).max(-1)
    values, counts = np.unique(np.round(vertices[(radius >= 2.7) & (radius <= 2.9), 2], 5), return_counts=True)
    expected = vertices.copy()
    expected[:, 2] = np.minimum(expected[:, 2], values[counts.argmax()])
    assert receipt['cap_z_m'] == values[counts.argmax()]
    # This fixture has no capping-degenerate faces, so no vertices need compaction.
    np.testing.assert_array_equal(output, expected)


def test_invalid_direction_layout_and_existing_flat_capping_are_explicit_errors():
    vertices, faces = mesh_arrays()
    with pytest.raises(ValueError, match='direction'):
        coverage.cap_mesh_arrays(vertices, faces, 'stairs_down')
    with pytest.raises(ValueError, match='9m'):
        coverage.cap_mesh_arrays(vertices + np.array([1., 0., 0.]), faces, 'stairs_up')
    with pytest.raises(ValueError, match='integer'):
        coverage.cap_mesh_arrays(vertices, faces.astype(float), 'stairs_up')
    capped, cap_faces, _ = coverage.cap_mesh_arrays(vertices, faces, 'stairs_up')
    with pytest.raises(ValueError, match='no outer stair'):
        coverage.cap_mesh_arrays(capped, cap_faces, 'stairs_up')


def test_config_rejects_early_call_changed_stairs_or_global_landing():
    cfg = config()
    assert coverage.column_names(cfg) == coverage.EXPECTED_COLUMNS
    assert coverage.validate_config(cfg)['geometry_only']
    cfg.num_rows, cfg.num_cols = 10, 20
    with pytest.raises(ValueError, match='after configure_mixed'):
        coverage.validate_config(cfg)
    cfg = config()
    cfg.sub_terrains['stairs_up'].step_width = .2
    with pytest.raises(ValueError, match='riser/tread'):
        coverage.validate_config(cfg)
    cfg = config()
    cfg.sub_terrains['stairs_down'].phase_landing_kind = 'stairs_down'
    with pytest.raises(ValueError, match='duplicate'):
        coverage.validate_config(cfg)
    cfg = config()
    cfg.sub_terrains['stairs_up'].proportion += .1
    with pytest.raises(ValueError, match='mapping'):
        coverage.validate_config(cfg)


class BaseGenerator:
    def __init__(self, cfg, device='cpu'):
        self.cfg, self.received = cfg, []
        meshes = {kind: Mesh(*mesh_arrays(-1 if kind == 'stairs_down' else 1))
                  for kind in dict.fromkeys(coverage.EXPECTED_COLUMNS)}
        for col, kind in enumerate(coverage.EXPECTED_COLUMNS):
            for row in coverage.ROWS:
                self._add_sub_terrain(meshes[kind], np.array([0., 0., 0.]), row, col, cfg.sub_terrains[kind])

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        self.received.append((row, col, coverage.mesh_hash(mesh.vertices, mesh.faces)))


def test_mixed_subclass_only_changes_whitelisted_cells_and_no_other_config(monkeypatch):
    monkeypatch.setitem(sys.modules, 'mixed_task', SimpleNamespace(MixedTerrainGenerator=BaseGenerator))
    terrain = config()
    terrain.class_type = BaseGenerator
    cfg = SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(terrain_generator=terrain)),
                           rewards=object(), commands=object(), terminations=object())
    unrelated = (cfg.rewards, cfg.commands, cfg.terminations)
    contract = coverage.configure(cfg)
    assert unrelated == (cfg.rewards, cfg.commands, cfg.terminations)
    assert issubclass(terrain.class_type, BaseGenerator)
    assert contract['columns_by_kind'] == {'stairs_up': [8], 'stairs_down': [11]}
    generator = terrain.class_type(terrain)
    receipts = terrain.stair_transition_coverage_receipts
    assert len(receipts) == len(generator.received) == 880
    selected = [r for r in receipts if r['selected']]
    assert len(selected) == 44
    assert {r['column'] for r in selected} == {8, 11}
    assert {r['row'] for r in selected} == set(range(22))
    assert all(r['origin_unchanged'] and r['origin_before'] == r['origin_after'] for r in receipts)
    assert all(r['before_sha256'] == r['after_sha256'] for r in receipts if not r['selected'])
    assert all(r['effective_original_level'] == 9 for r in selected if r['row'] >= 10)
    with pytest.raises(ValueError, match='exactly once'):
        coverage.configure(cfg)


def test_actual_coverage_is_counted_by_learner_type_and_level_without_mutation():
    levels, columns = np.array([3, 7, 8, 4, 2, 20]), np.array([8, 6, 11, 12, 17, 20])
    learner = np.array([0, 0, 1, 1, 1, 0])
    saved = [array.copy() for array in (levels, columns, learner)]
    result = coverage.coverage_counts(levels, columns, learner)
    assert result['coverage_total'] == 2
    assert result['fraction'] == pytest.approx(1/3)
    assert result['by_learner']['teacher']['by_type']['stairs_up']['coverage'] == 1
    assert result['by_learner']['student']['by_type']['stairs_down']['coverage_by_level'][8] == 1
    assert result['by_learner']['student']['by_type']['flat']['coverage'] == 0
    for actual, expected in zip((levels, columns, learner), saved):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='inactive'):
        coverage.coverage_counts(np.array([10]), np.array([8]), np.array([0]))


def test_degenerate_outer_faces_removed_while_protected_triangle_hash_stays_exact():
    vertices, faces = mesh_arrays()
    faces = np.vstack((faces, [0, 0, 1]))
    output, output_faces, receipt = coverage.cap_mesh_arrays(vertices, faces, 'stairs_up')
    assert receipt['removed_degenerate_faces'] == 1
    assert len(output_faces) == len(faces) - 1
    assert receipt['inner_nondegenerate_triangles_before_sha256'] == receipt['inner_nondegenerate_triangles_after_sha256']
    assert np.all(coverage._areas(output, output_faces) > coverage.AREA_EPS)
