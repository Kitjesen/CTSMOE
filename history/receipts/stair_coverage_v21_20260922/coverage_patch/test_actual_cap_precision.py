"""Regression from the actual attempt01 environment-construction failure mesh."""
from pathlib import Path

import numpy as np
import pytest

import stair_transition_coverage as coverage

FIXTURE = Path(__file__).with_name('fixtures') / 'geometry_failure_attempt01.npz'


def actual_mesh():
    with np.load(FIXTURE, allow_pickle=False) as data:
        return data['vertices'], data['faces'], str(data['kind'].item())


def test_actual_failure_was_rounding_and_repair_keeps_exact_inner_geometry():
    vertices, faces, kind = actual_mesh()
    assert kind == 'stairs_down' and vertices.shape == (8227, 3) and faces.shape == (16200, 3)
    radius = np.abs(vertices[:, :2]).max(-1)
    old_changed = vertices.copy()
    old_changed[:, 2] = np.maximum(vertices[:, 2], .24)
    old_inner_changes = (radius < 2.7) & (old_changed[:, 2] != vertices[:, 2])
    assert np.count_nonzero(old_inner_changes) == 107
    assert np.abs(old_changed[old_inner_changes, 2] - vertices[old_inner_changes, 2]).max() == .24 - float(np.float32(.24))

    output, output_faces, receipt = coverage.cap_mesh_arrays(vertices, faces, kind)
    assert receipt['cap_rounded_mode_z_m'] == .24
    assert receipt['cap_z_m'] == float(np.float32(.24)) == .23999999463558197
    assert receipt['cap_original_level_count'] == 1
    assert receipt['changed_faces_outside_inner_square'] is True
    assert receipt['changed_face_count'] == 9000
    assert receipt['changed_faces_inner_square_overlap_count'] == 0
    assert receipt['changed_faces_min_axis_separation_m'] == pytest.approx(.3)
    for stem in ('inner_vertices', 'inner_triangle_vertex_positions', 'inner_nondegenerate_triangles'):
        assert receipt[stem + '_before_sha256'] == receipt[stem + '_after_sha256']
    assert coverage.mesh_hash(output, output_faces) == receipt['after_sha256']

    # Independent complete geometric argument: every changed triangle is outside
    # the square on a common axis, so no part of any changed triangle is internal.
    candidate = vertices.copy()
    candidate[:, 2] = np.maximum(vertices[:, 2], receipt['cap_z_m'])
    changed_faces = (candidate[:, 2] != vertices[:, 2])[faces].any(-1)
    tri = vertices[faces[changed_faces]]
    outside = ((tri[:, :, 0].min(1) >= 2.7) | (tri[:, :, 0].max(1) <= -2.7)
               | (tri[:, :, 1].min(1) >= 2.7) | (tri[:, :, 1].max(1) <= -2.7))
    assert outside.all()
    np.testing.assert_array_equal(candidate[faces[~changed_faces]], vertices[faces[~changed_faces]])


def test_real_inner_height_change_is_still_rejected_exactly():
    vertices, faces, kind = actual_mesh()
    radius = np.abs(vertices[:, :2]).max(-1)
    vertices[np.flatnonzero(radius < 1.)[0], 2] = .16
    with pytest.raises(ValueError, match='protected inner stair vertices'):
        coverage.cap_mesh_arrays(vertices, faces, kind)


def test_changed_triangle_crossing_inner_region_is_rejected_without_inner_vertices():
    vertices, faces, kind = actual_mesh()
    crossing = [int(vertices[:, 0].argmin()), int(vertices[:, 0].argmax()), int(vertices[:, 1].argmax())]
    faces = np.vstack((faces, crossing))
    with pytest.raises(ValueError, match='triangle may intersect'):
        coverage.cap_mesh_arrays(vertices, faces, kind)


def test_multiple_original_levels_in_one_rounded_bin_are_not_silently_averaged():
    vertices, faces, kind = actual_mesh()
    radius = np.abs(vertices[:, :2]).max(-1)
    candidates = np.flatnonzero((radius >= 2.7) & (radius <= 2.9) & (np.round(vertices[:, 2], 5) == .24))
    vertices[candidates[0], 2] += 1e-7
    with pytest.raises(ValueError, match='multiple original mesh heights'):
        coverage.cap_mesh_arrays(vertices, faces, kind)
