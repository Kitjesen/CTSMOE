import numpy as np
import pytest
from audit_helpers import assign_columns, update_geometric_climb


def test_atlas_columns_are_not_terrain_ordinals():
    ranges = dict(wave=(0, 1), slope_up=(1, 3), slope_down=(3, 5),
                  rough_slope=(5, 6), stairs_up=(6, 11), stairs_down=(11, 13),
                  obstacles=(13, 17), stepping_stones=(17, 17), gap=(17, 17),
                  flat=(17, 20), pit=(20, 40))
    cases = [{"terrain": name} for name in ("flat", "pit", "stairs_up", "stairs_down")]
    assert assign_columns(cases, ranges) == [17, 20, 6, 11]
    with pytest.raises(ValueError):
        assign_columns([{"terrain": "pit", "terrain_column": 10}], ranges)
    with pytest.raises(ValueError):
        assign_columns([{"terrain": "gap"}], ranges)


def test_geometric_milestone_requires_continuity_and_remembers_climb():
    hold = np.zeros(2, dtype=np.float32)
    seen = np.zeros(2, dtype=bool)
    for step in range(10):
        eligible = np.array([True, step != 5])
        hold, seen = update_geometric_climb(hold, seen, eligible, .02)
    assert seen.tolist() == [True, False]
    hold, seen = update_geometric_climb(hold, seen, np.array([False, False]), .02)
    assert hold.tolist() == [0., 0.]
    assert seen.tolist() == [True, False]
