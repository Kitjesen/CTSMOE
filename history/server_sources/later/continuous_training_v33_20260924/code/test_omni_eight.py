"""No high-platform assignment and measured hip cost remain explicit."""

from collections import Counter

import torch

from terrain_allocation import AllocationCourse, BASE, NO_PIT, NO_PIT_PROFILE, reassign


def test_no_pit_migration_preserves_retained_levels_and_same_learner_donors():
    names = list(BASE)
    columns = [i % 9 for i in range(4096)]
    levels = [i % (22 if names[columns[i]] == "pit" else 10) for i in range(4096)]
    groups = {"teacher": list(range(3072)), "student": list(range(3072, 4096))}
    new_levels, new_columns, receipt = reassign(
        levels, columns, names, groups, NO_PIT, seed=34823, profile=NO_PIT_PROFILE)
    assert not any(names[column] == "pit" for column in new_columns)
    for index, donor in receipt["moves"]:
        assert (index < 3072) == (donor < 3072)
        assert (new_levels[index], new_columns[index]) == (levels[donor], columns[donor])
    for group, ids in groups.items():
        counts = Counter(names[new_columns[index]] for index in ids)
        assert counts["pit"] == 0
        assert counts["flat"] == receipt["group_counts"][group]["flat"]
    course = AllocationCourse(profile=NO_PIT_PROFILE, weights=dict(NO_PIT))
    assert course.evaluate({}, 35323)["move"] is None
    assert course.weights == NO_PIT


def test_measured_hip_margin_starts_before_soft_limit():
    from hip_margin_all import margin_cost

    inward = torch.tensor([[0.12], [0.34], [0.39], [0.41]])
    cost = margin_cost(inward, torch.ones(1))
    torch.testing.assert_close(cost, torch.tensor([0., 0., (.05 / .07) ** 2, 1.]), atol=1e-6, rtol=0)
