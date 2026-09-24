"""Pure CPU tests for the 45-degree slope course and saved-row migration."""

import math

import pytest

from slope_curriculum import (
    SLOPE_ANGLES_DEG,
    SLOPE_PROFILE,
    angle_for_level,
    level_from_difficulty,
    migrate_slope_levels,
    slope_range_for_difficulty,
)


def test_all_angles_are_monotonic_and_top_row_reaches_exact_45_degrees():
    assert list(SLOPE_ANGLES_DEG) == [5, 10, 15, 20, 25, 30, 35, 40, 42.5, 45]
    assert [angle_for_level(row) for row in range(22)] == list(SLOPE_ANGLES_DEG) + [45] * 12
    slopes = [slope_range_for_difficulty((row + 0.5) / 22) for row in range(22)]
    assert all(pair[0] == pair[1] for pair in slopes)
    assert [pair[0] for pair in slopes] == sorted(pair[0] for pair in slopes)
    assert slopes[-1] == (math.tan(math.radians(45)),) * 2
    assert level_from_difficulty(1) == 21
    assert slope_range_for_difficulty(1) == slopes[-1]


def test_conservative_migration_for_every_old_slope_row_and_type():
    names = ["slope_up", "slope_down", "rough_slope"]
    levels = list(range(10)) * 3
    columns = [index // 10 for index in range(30)]
    mapped, receipt = migrate_slope_levels(levels, columns, names, "original", SLOPE_PROFILE)
    assert columns == [index // 10 for index in range(30)]
    for old_row in range(10):
        old_min_angle = math.degrees(math.atan(0.1 + 0.468 * old_row / 10))
        new_angle = angle_for_level(mapped[old_row])
        assert new_angle <= old_min_angle
        assert all(angle > old_min_angle for angle in SLOPE_ANGLES_DEG if angle > new_angle)
        assert mapped[old_row] == mapped[old_row + 10] == mapped[old_row + 20]
    assert receipt["slope_envs"] == 30
    assert receipt["counts_before_by_type"]["slope_up"] == dict.fromkeys(range(10), 1)
    assert sum(receipt["counts_after_by_type"]["rough_slope"].values()) == 10
    assert receipt["new_nominal_angle_deg"]["max"] <= receipt["old_lower_angle_deg"]["max"]


def test_pit_and_other_terrain_rows_and_environment_order_are_unchanged():
    names = ["pit", "wave", "slope_up", "stairs_down"]
    levels = [0, 21, 18, 9, 3, 7]
    columns = [0, 0, 1, 2, 3, 1]
    mapped, receipt = migrate_slope_levels(levels, columns, names, "original", SLOPE_PROFILE)
    assert mapped[:3] == levels[:3]
    assert mapped[4:] == levels[4:]
    assert columns == [0, 0, 1, 2, 3, 1]
    assert receipt["pit_height_m"]["min"] == pytest.approx(0.15)
    assert receipt["pit_height_m"]["max"] == pytest.approx(1.2)
    assert receipt["slope_envs"] == 1
    assert len(mapped) == len(levels)
    assert migrate_slope_levels(mapped, columns, names, SLOPE_PROFILE, SLOPE_PROFILE)[0] == mapped


@pytest.mark.parametrize("difficulty", [-0.001, 1.001, math.nan, math.inf, -math.inf])
def test_invalid_difficulties_are_rejected(difficulty):
    with pytest.raises(ValueError):
        level_from_difficulty(difficulty)


@pytest.mark.parametrize("level", [-1, 22, 1.5, math.nan, True])
def test_invalid_levels_are_rejected(level):
    with pytest.raises(ValueError):
        angle_for_level(level)


def test_atlas_row_boundaries_and_bad_migration_inputs():
    assert level_from_difficulty(0) == 0
    assert level_from_difficulty(1 / 22 - 1e-9) == 0
    assert level_from_difficulty(1 / 22) == 1
    assert level_from_difficulty(21 / 22) == 21
    for args in (
        ([0], [0], ["slope_up"], "unknown", SLOPE_PROFILE),
        ([0], [0], ["slope_up"], SLOPE_PROFILE, "original"),
        ([10], [0], ["slope_up"], "original", SLOPE_PROFILE),
        ([0], [], ["slope_up"], "original", SLOPE_PROFILE),
        ([0], [1], ["slope_up"], "original", SLOPE_PROFILE),
    ):
        with pytest.raises(ValueError):
            migrate_slope_levels(*args)
