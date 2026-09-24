import numpy as np
import pytest
from routes import Course, generate


@pytest.mark.parametrize('kind', ['curbs', 'blocks', 'ramp_stairs_ramp'])
@pytest.mark.parametrize('level', range(4))
def test_closed_geometry_and_spawn(kind, level):
    height = [.1, .2, .3, .4][level] if kind != 'ramp_stairs_ramp' else [.1, .15, .25, .3][level]
    mesh, data = generate(Course(kind, height=height, slope_deg=[15,25,35,45][level]))
    assert mesh.is_watertight and mesh.is_winding_consistent
    top = np.array(data['profile'])
    assert np.all(np.diff(top[:,0]) >= 0)
    assert top[0].tolist() == [0,0] and top[1].tolist() == [2,0]
    assert top[-1,1] == 0 and data['max_step_m'] <= .400001
    assert mesh.bounds[0,1] == -3 and mesh.bounds[1,1] == 3


def test_ramp_has_no_notch_and_angles_match():
    _, data = generate(Course('ramp_stairs_ramp', height=.25, slope_deg=45))
    points = np.array(data['profile'])
    d = np.diff(points, axis=0)
    ramps = d[(d[:,0] > 0) & (np.abs(d[:,1]) > 1e-8)]
    np.testing.assert_allclose(np.abs(ramps[:,1]/ramps[:,0]), 1.)
    assert (d[d[:,0] == 0,1] > 0).all()


def test_random_blocks_reproducible_but_not_identical():
    a, da = generate(Course('blocks', seed=4))
    b, _ = generate(Course('blocks', seed=4))
    _, dc = generate(Course('blocks', seed=5))
    np.testing.assert_array_equal(a.vertices,b.vertices)
    assert da['profile'] != dc['profile']


def test_yaw_and_time_constraints_are_not_hidden():
    _, d = generate(Course('curbs', height=.4))
    assert d['required_width_for_yaw45_m'] > 6
    assert d['seconds_at_03mps_without_margin'] > 25


def test_reject_illegal_height():
    with pytest.raises(ValueError):
        generate(Course('curbs', height=.8))
