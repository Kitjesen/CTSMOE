"""Offline migration contract; intentionally does not start Isaac or a GPU job."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from composite_atlas import (
    NEW_COLUMNS,
    PROFILE,
    configure_atlas,
    migrate_allocation_state,
    migrate_assignments,
)
from composite_route import COMPACT_9X9, composite_terrain_9x9, profile_stations, spec_for_level
from stair_transition_coverage import EXPECTED_COLUMNS, column_names, coverage_counts, validate_config
from terrain_allocation import COMPOSITE_BALANCED, EARLY_BALANCED, EARLY_BALANCED_PROFILE
from training_profiles import allocate_counts


def fake_mixed_cfg():
    counts = Counter(EXPECTED_COLUMNS[:20])
    sub = {name: SimpleNamespace(proportion=count / 40, step_height_range=(.05, .257),
                                 step_width=.31, platform_width=3.)
           for name, count in counts.items()}
    sub["pit"] = SimpleNamespace(proportion=.5)
    terrain = SimpleNamespace(size=(9., 9.), num_rows=22, num_cols=40,
                              sub_terrains=sub, curriculum=True,
                              sub_terrain_border_width=.5)
    return SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(terrain_generator=terrain)))


def test_configure_atlas_retains_original_columns_and_coverage():
    cfg = fake_mixed_cfg()
    old = column_names(cfg.scene.terrain.terrain_generator)
    assert old == EXPECTED_COLUMNS
    receipt = configure_atlas(cfg, subterrain_cls=SimpleNamespace)
    terrain = cfg.scene.terrain.terrain_generator
    assert terrain.num_cols == 42
    assert column_names(terrain) == NEW_COLUMNS
    assert column_names(terrain)[:20] == old[:20]
    assert column_names(terrain)[20:22] == ("composite", "composite")
    assert column_names(terrain)[22:] == ("pit",) * 20
    assert terrain.sub_terrains["composite"].function.__name__ == "composite_terrain_9x9"
    assert receipt["old_pit_columns"] == list(range(20, 40))
    contract = validate_config(terrain)
    assert contract["columns_by_kind"] == {"stairs_up": [8], "stairs_down": [11]}
    assert contract["expected_column_names"] == list(NEW_COLUMNS)


def mock_v23_assignments():
    levels, columns, groups = [], [], {}
    representatives = {name: EXPECTED_COLUMNS.index(name) for name in dict.fromkeys(EXPECTED_COLUMNS)}
    for group, size in (("teacher", 3072), ("student", 1024)):
        offset = len(levels)
        targets = allocate_counts(size, EARLY_BALANCED)
        ids = []
        for name, count in targets.items():
            for j in range(count):
                ids.append(len(levels))
                levels.append(j % (22 if name == "pit" else 10))
                columns.append(representatives[name])
        assert len(ids) == size
        groups[group] = list(range(offset, len(levels)))
    return levels, columns, groups


def test_migration_preserves_pit_and_all_old_stair_samples():
    levels, columns, groups = mock_v23_assignments()
    before = Counter(EXPECTED_COLUMNS[c] for c in columns)
    assert before["pit"] == 409
    new_levels, new_cols, receipt = migrate_assignments(levels, columns, groups)
    after = Counter(NEW_COLUMNS[c] for c in new_cols)
    assert receipt["selected_count"] == 205
    assert after["composite"] == 205
    assert after["pit"] == 409
    for name in ("stairs_up", "stairs_down", "slope_up", "slope_down"):
        assert after[name] == before[name]
    for i, old_col in enumerate(columns):
        if EXPECTED_COLUMNS[old_col] == "pit":
            assert new_cols[i] == old_col + 2
            assert new_levels[i] == levels[i]
        elif i not in receipt["selected_from_flat"]:
            assert new_cols[i] == old_col
            assert new_levels[i] == levels[i]
    assert {new_levels[i] for i in receipt["selected_from_flat"]} == {0}


def test_coverage_counts_under_42_columns_separates_new_route_from_pit():
    levels = np.asarray((0, 9, 21))
    columns = np.asarray((20, 8, 22))
    learners = np.asarray((1, 1, 0))
    receipt = coverage_counts(levels, columns, learners, NEW_COLUMNS)
    assert receipt["by_learner"]["student"]["by_type"]["composite"]["total"] == 1
    assert receipt["by_learner"]["student"]["by_type"]["stairs_up"]["coverage"] == 1
    assert receipt["by_learner"]["teacher"]["by_type"]["pit"]["total"] == 1
    with pytest.raises(ValueError, match="Original terrain"):
        coverage_counts(np.asarray((20,)), np.asarray((20,)), np.asarray((1,)), NEW_COLUMNS)


def test_allocation_migration_preserves_nine_strata_exactly_and_adds_tenth():
    old_stats = torch.arange(9 * 22 * 6 * 3 * 3, dtype=torch.float64).reshape(9, 22, 6, 3, 3)
    saved = dict(
        profile="balanced_adaptive_v1",
        course=dict(profile=EARLY_BALANCED_PROFILE, weights=dict(EARLY_BALANCED),
                    reference={name: {} for name in EARLY_BALANCED},
                    bad_streak={name: 0 for name in EARLY_BALANCED},
                    last_check=33000, decisions=[]),
        stats=old_stats, pending={}, moves_applied=3,
    )
    result = migrate_allocation_state(saved)
    assert result["course"]["profile"] == PROFILE
    assert result["course"]["weights"] == COMPOSITE_BALANCED
    assert result["stats"].shape == (10, 22, 6, 3, 3)
    torch.testing.assert_close(result["stats"][:9], old_stats, rtol=0, atol=0)
    assert not result["stats"][9].any()
    assert saved["stats"].shape[0] == 9
    assert saved["course"]["weights"] == EARLY_BALANCED
    with pytest.raises(ValueError, match="settled"):
        migrate_allocation_state({**saved, "pending": {10: (2, 17)}})


def test_all_ten_route_levels_keep_exact_stations_and_continuous_risers():
    cfg = SimpleNamespace(size=(9., 9.))
    heights = []
    for level in range(10):
        spec = spec_for_level(level)
        meshes, origin = composite_terrain_9x9((level + .1) / 22, cfg)
        profile = profile_stations(spec)
        assert len(meshes) == 1
        assert tuple(origin) == (COMPACT_9X9.spawn_x_m, 4.5, 0.)
        assert spec.slope_angle_deg == pytest.approx(10 + level * 10 / 9)
        assert spec.stair_height_m == pytest.approx(.08 + level * .04 / 9)
        assert spec.final_usable_length_m >= 1.5
        assert spec.approach_end_x_m == COMPACT_9X9.approach_end_x_m
        assert spec.final_start_x_m == COMPACT_9X9.final_start_x_m
        assert np.all(np.diff(profile[:, 0]) >= 0)
        assert np.count_nonzero(np.diff(profile[:, 0]) == 0) == 4
        for step in range(4):
            riser = 4 + 2 * step
            assert profile[riser, 0] == profile[riser - 1, 0]
            assert profile[riser, 1] - profile[riser - 1, 1] == pytest.approx(
                spec.stair_height_m)
        assert profile[-1, 1] == pytest.approx(spec.final_height_m)
        assert meshes[0].vertices[:, 2].max() == pytest.approx(spec.final_height_m)
        heights.append(spec.final_height_m)
    assert heights == sorted(heights)
    assert heights[-1] < 1.2
    with pytest.raises(ValueError, match="0..9"):
        spec_for_level(10)
