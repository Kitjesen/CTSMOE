import numpy as np
import pytest
from geometry import height_field, max_adjacent_step_m


@pytest.mark.parametrize('difficulty', [0, .25, .5, .75, 1])
def test_signed_overlapping_boxes_keep_neighbor_and_spawn_limits(difficulty):
    for seed in range(100):
        field = height_field(difficulty, seed)
        assert field.shape == (90, 90)
        assert max_adjacent_step_m(field) <= .4 + 1e-12
        assert np.count_nonzero(field[30:60,30:60]) == 0
        assert abs(field).max() * .005 <= .4 + 1e-12
        assert not field[:5].any() and not field[-5:].any()
        assert not field[:,:5].any() and not field[:,-5:].any()


def test_full_difficulty_retains_bumps_and_depressions_and_40cm_steps():
    fields = [height_field(1, seed) for seed in range(100)]
    assert any(f.max() == 80 for f in fields)
    assert any(f.min() == -80 for f in fields)
    assert any(abs(max_adjacent_step_m(f)-.4) < 1e-12 for f in fields)
    np.testing.assert_array_equal(height_field(1, 42), height_field(1, 42))


@pytest.mark.parametrize('difficulty', [-.1, 1.1, np.nan])
def test_invalid_difficulty_rejected(difficulty):
    with pytest.raises(ValueError):
        height_field(difficulty, 0)
