import numpy as np
import pytest
import torch

from pit_geometry import build_pit_meshes, height_for_level, level_from_difficulty, pit_boxes
from pit_logic import PitCourse, stable_success, top_support


@pytest.mark.parametrize("level", range(22))
def test_single_vertical_wall_and_bottom_origin(level):
    h = height_for_level(level)
    meshes, origin = build_pit_meshes(h)
    np.testing.assert_allclose(origin, [3, 3, 0])
    assert len(meshes) == 5
    for mesh, bounds in zip(meshes, pit_boxes(h).values(), strict=True):
        np.testing.assert_allclose(mesh.bounds, bounds)
        assert mesh.is_watertight
        assert np.all(np.max(np.abs(mesh.face_normals), axis=1) == 1)
    # Along the centerline there is only floor then a single platform.
    for x in (0.5, 3., 4.49, 4.51, 7.5):
        z = max(hi[2] for lo, hi in pit_boxes(h).values() if lo[0] < x < hi[0] and lo[1] < 3 < hi[1])
        assert z == (0 if x < 4.5 else h)
    assert level_from_difficulty((level+0.5)/22) == level
    assert level_from_difficulty((level+0.00001)/22) == level


def test_height_endpoints_and_invalid_inputs():
    assert height_for_level(21) == 1.2
    assert level_from_difficulty(1) == 21
    for d in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            level_from_difficulty(d)


def test_wall_contact_and_airborne_are_not_platform_support():
    pos = torch.tensor([[[1.7, 0, .343]]*4]*3)
    force = torch.tensor([[[0., 0., 30.]]*4]*3)
    pos[1, :, 0] = 1.5
    force[2] = 0
    supported = top_support(pos, force, torch.full((3,), 1.5), torch.full((3,), .25))
    assert supported.all(-1).tolist() == [True, False, False]


def test_success_needs_whole_body_and_uninterrupted_two_seconds():
    n = 6
    hold = torch.zeros(n)
    support = torch.ones(n, 4, dtype=torch.bool)
    support[1, -1] = False
    clear = torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.bool)
    violation = torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.bool)
    started = torch.tensor([1, 1, 1, 1, 0, 1], dtype=torch.bool)
    ones = torch.ones(n, dtype=torch.bool)
    for i in range(100):
        quiet = ones.clone()
        if i == 80:
            quiet[-1] = False
        hold, success = stable_success(hold, support, clear, ones, quiet, violation, started, .02)
    assert success.tolist() == [True, False, False, False, False, False]


def test_course_requires_all_bins_two_batches_and_survives_restore():
    course = PitCourse()
    for b in range(3):
        assert not course.observe([b]*64, [True]*64, [False]*64)
    assert course.frontier == 2 and course.streak == 0
    assert not course.observe([3]*64, [True]*64, [False]*64)
    course = PitCourse.restore(course.state_dict())
    assert course.streak == 1
    assert course.observe(list(range(4))*64, [True]*256, [False]*256)
    assert course.frontier == 3
    assert not course.observe(list(range(4))*64, [False]*256, [True]*256)
    assert course.frontier == 3 and course.streak == 0


def test_course_never_exceeds_validated_cap():
    course = PitCourse(frontier=4, max_level=4, min_samples=1)
    for _ in range(3):
        assert not course.observe([0, 1, 2, 3], [True]*4, [False]*4)
    assert course.frontier == 4
