from collections import Counter

from mixed_logic import mixed_assignments


def test_migration_retains_all_types_and_exact_old_levels_per_learner():
    names = ["wave", "slope_up", "slope_up", "slope_down", "slope_down", "rough_slope",
             "stairs_up", "stairs_up", "stairs_up", "stairs_up", "stairs_up", "stairs_down",
             "stairs_down", "obstacles", "obstacles", "obstacles", "obstacles", "flat", "flat", "flat"]
    columns = [i % 20 for i in range(256)]
    levels = [i % 10 for i in range(256)]
    groups = {"teacher": list(range(192)), "student": list(range(192, 256))}
    new_levels, new_columns, report = mixed_assignments(levels, columns, names, groups)
    for group, ids in groups.items():
        counts = Counter("pit" if new_columns[i] >= 20 else names[new_columns[i]] for i in ids)
        assert counts == report[group]
        assert counts["pit"] == len(ids)//2
        assert len(counts) == 9
    for i in range(256):
        if new_columns[i] < 20:
            assert (new_levels[i], new_columns[i]) == (levels[i], columns[i])
        else:
            assert new_levels[i] == 2
            assert 20 <= new_columns[i] < 40
    assert mixed_assignments(levels, columns, names, groups) == (new_levels, new_columns, report)


def test_different_terrain_level_scales_are_reported_separately():
    from training_profiles import summarize_terrain
    report = summarize_terrain([6, 2, 21], [0, 1, 1], ["stairs_up", "pit"])
    assert report["stairs_up"]["curriculum_fraction"] == 6/9
    assert len(report["pit"]["counts_L0_to_L21"]) == 22
    assert "counts_L0_to_L9" not in report["pit"]
    assert report["pit"]["mean_height_m"] == .15+.05*11.5
