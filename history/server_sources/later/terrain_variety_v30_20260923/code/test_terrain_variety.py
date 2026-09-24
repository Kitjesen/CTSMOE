import numpy as np
import pytest

from terrain_variety import VARIANTS, mesh_for, segments


@pytest.mark.parametrize('kind', list(VARIANTS.values()))
@pytest.mark.parametrize('level', [0, 5, 9])
def test_closed_surface_and_spawn(kind, level):
    mesh = mesh_for(kind, level)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert np.all(mesh.area_faces > 0)
    rings = segments(kind, level)
    assert rings[0] == (0., 1.5, 0., 0.)
    assert rings[-1][1] == 4.5
    assert all(abs(a[1] - b[0]) < 1e-8 for a, b in zip(rings, rings[1:]))
    assert np.allclose(mesh.bounds[:, :2], [[-4.5, -4.5], [4.5, 4.5]])
    jumps = [b[2] - a[3] for a, b in zip(rings, rings[1:])]
    assert max(abs(x) for x in jumps) <= .20000001
    if kind == 'short_tread_down':
        assert all(x <= 0 for x in jumps)
    else:
        assert all(x >= 0 for x in jumps)


def test_variants_keep_reference_stairs():
    assert 8 not in VARIANTS and 11 not in VARIANTS
    assert len(segments('single_curb_up', 9)) == 2
    assert segments('short_tread_up', 9)[1][1] == 1.75
    assert segments('long_tread_up', 9)[1][1] == 1.85
