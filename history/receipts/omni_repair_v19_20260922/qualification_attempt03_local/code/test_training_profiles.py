"""Check height-course disabling and difficulty-preserving terrain migration."""
from types import SimpleNamespace

import pytest

from training_profiles import configure_base_height_penalty, rebalance_terrain_assignments, summarize_terrain


COLUMNS = ['wave', 'slope_up', 'slope_up', 'slope_down', 'slope_down', 'rough_slope',
           'stairs_up', 'stairs_up', 'stairs_up', 'stairs_up', 'stairs_up',
           'stairs_down', 'stairs_down', 'obstacles', 'obstacles', 'obstacles', 'obstacles',
           'flat', 'flat', 'flat']


def test_height_off_disables_both_weight_and_course():
    cfg = SimpleNamespace(rewards=SimpleNamespace(base_height_l2=SimpleNamespace(weight=-1)),
                          curriculum=SimpleNamespace(base_height_l2=object(), terrain_levels=object()))
    terrain_course = cfg.curriculum.terrain_levels
    configure_base_height_penalty(cfg, 'off')
    assert cfg.rewards.base_height_l2.weight == 0
    assert cfg.curriculum.base_height_l2 is None
    assert cfg.curriculum.terrain_levels is terrain_course


def test_formal_migration_preserves_existing_states_and_uses_matching_donors():
    n = 4096
    columns = [min(int(i/(n/20)), 19) for i in range(n)]
    levels = [i % 5 if COLUMNS[col] == 'stairs_up' else 9 for i, col in enumerate(columns)]
    groups = {'teacher': [i for i in range(n) if i % 4], 'student': list(range(0, n, 4))}
    new_levels, new_cols, receipt = rebalance_terrain_assignments(levels, columns, COLUMNS, groups, 'stairs80_v1')
    summary = summarize_terrain(new_levels, new_cols, COLUMNS)
    assert summary['stairs_up']['count'] == 2457
    assert summary['stairs_down']['count'] == 819
    assert max(new_levels[i] for i, col in enumerate(new_cols) if COLUMNS[col] == 'stairs_up') <= 4
    moved_ids = set()
    for move in receipt['moves']:
        i, donor = move['env_id'], move['donor_env_id']
        moved_ids.add(i)
        assert COLUMNS[columns[donor]] == move['to']
        assert i in groups[move['learner']] and donor in groups[move['learner']]
        assert (new_levels[i], new_cols[i]) == (levels[donor], columns[donor])
    for i in set(range(n))-moved_ids:
        assert (new_levels[i], new_cols[i]) == (levels[i], columns[i])
    assert levels == [i % 5 if COLUMNS[col] == 'stairs_up' else 9 for i, col in enumerate(columns)]
    assert receipt['group_counts']['student']['stairs_up'] == 614
    assert receipt['group_counts']['teacher']['stairs_up'] == 1843
    assert rebalance_terrain_assignments(levels, columns, COLUMNS, groups, 'stairs80_v1') == (new_levels, new_cols, receipt)


def test_malformed_learner_partition_rejected():
    with pytest.raises(ValueError, match='partition'):
        rebalance_terrain_assignments([1, 2], [0, 1], COLUMNS, {'teacher': [0, 0]}, 'stairs80_v1')


def test_missing_donor_fails_instead_of_using_flat_difficulty():
    columns = [col if COLUMNS[col] != 'stairs_up' else 19 for col in range(20)] * 4
    with pytest.raises(ValueError, match='No saved stairs_up donor'):
        rebalance_terrain_assignments([9]*80, columns, COLUMNS,
                                      {'teacher': list(range(80))}, 'stairs80_v1')
