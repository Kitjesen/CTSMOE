import copy
from collections import Counter

import pytest

from terrain_allocation import AllocationCourse, BASE, FLOOR, CEILING, compare, reassign


def samples(fail=0, error=.1, n=160, level='6:2:0'):
    return {level: [n, n*fail, n*error]}


def test_migration_keeps_exact_retained_levels_and_same_learner_donors():
    names = list(BASE)
    columns = [i % 9 for i in range(4096)]
    levels = [i % (22 if names[columns[i]] == 'pit' else 10) for i in range(4096)]
    groups = dict(teacher=list(range(3072)), student=list(range(3072, 4096)))
    new_levels, new_columns, receipt = reassign(levels, columns, names, groups, BASE)
    moved = dict(receipt['moves'])
    for i in range(4096):
        if i in moved:
            donor = moved[i]
            assert (i < 3072) == (donor < 3072)
            assert (new_levels[i], new_columns[i]) == (levels[donor], columns[donor])
        else:
            assert (new_levels[i], new_columns[i]) == (levels[i], columns[i])
    # Largest-remainder rounding is performed independently per learner.
    assert Counter(names[c] for c in new_columns)['pit'] == 1639
    assert sum(receipt['group_counts']['student'].values()) == 1024


def test_no_adjustment_on_new_difficulty_or_too_few_samples():
    assert not compare(samples(), samples(level='7:2:0'))['ready']
    assert not compare(samples(), samples(n=127))['ready']
    course = AllocationCourse()
    for iteration in (500, 1000, 1500):
        assert course.evaluate({'pit': samples(level=f'{iteration}:2:0')}, iteration)['move'] is None


def test_two_confirmations_required_and_checkpoint_restores_decision():
    course = AllocationCourse()
    baseline = {name: samples() for name in BASE}
    assert course.evaluate(baseline, 500)['move'] is None
    worse = dict(baseline, wave=samples(fail=.1))
    assert course.evaluate(worse, 1000)['move'] is None
    restored = AllocationCourse(**copy.deepcopy(course.state_dict()))
    decision = restored.evaluate(worse, 1500)
    assert decision['move'] == dict(donor='pit', receiver='wave', fraction=.02)
    assert restored.weights['pit'] == pytest.approx(.38)
    assert restored.weights['wave'] == pytest.approx(.08)
    for i in range(20):
        restored.evaluate(worse, 2000+500*i)
    assert sum(restored.weights.values()) == pytest.approx(1)
    assert all(FLOOR[k]-1e-8 <= v <= CEILING[k]+1e-8 for k,v in restored.weights.items())


def test_changing_stratum_mix_does_not_fake_regression():
    reference = {'6:2:0': [160, 16, 16], '7:2:0': [160, 32, 32]}
    current = {'6:2:0': [320, 32, 32], '7:2:0': [1600, 320, 320]}
    result = compare(reference, current)
    assert result['ready'] and not result['bad']
    assert result['failure'] == result['reference_failure']


def test_unknown_weights_fail_closed():
    with pytest.raises(ValueError):
        reassign([0]*9, list(range(9)), list(BASE), {'student':list(range(9))}, {'pit':1})


def test_flat_focus_preserves_donors_and_cannot_be_undone_by_adaptive_sampling():
    from terrain_allocation import FLAT_FOCUS, FLAT_FOCUS_PROFILE
    names = list(BASE)
    columns = [i % 9 for i in range(4096)]
    levels = [i % (22 if names[columns[i]] == 'pit' else 10) for i in range(4096)]
    groups = {'teacher':[i for i in range(4096) if i%4],
              'student':[i for i in range(4096) if not i%4]}
    new_levels, new_columns, receipt = reassign(levels,columns,names,groups,FLAT_FOCUS,
                                                profile=FLAT_FOCUS_PROFILE)
    moved = dict(receipt['moves'])
    for i in range(4096):
        if i in moved:
            donor = moved[i]
            assert bool(i%4) == bool(donor%4)
            assert (new_levels[i],new_columns[i]) == (levels[donor],columns[donor])
        else:
            assert (new_levels[i],new_columns[i]) == (levels[i],columns[i])
    for counts in receipt['group_counts'].values():
        assert abs(counts['flat']/sum(counts.values())-.4) < .002
        assert all(counts[name]>0 for name in BASE)
    course = AllocationCourse(profile=FLAT_FOCUS_PROFILE,weights=dict(FLAT_FOCUS))
    for iteration in (500,1000,1500):
        assert course.evaluate({name:samples(fail=.1 if name=='pit' else 0) for name in BASE},iteration)['move'] is None
    assert course.weights == FLAT_FOCUS


def test_pending_moves_only_apply_to_reset_ids_and_clear_pit_rewards():
    from types import SimpleNamespace as NS
    import torch
    from terrain_allocation_runtime import AdaptiveAllocation

    n = 18
    names = list(BASE)
    terrain = NS(terrain_levels=torch.zeros(n, dtype=torch.long),
                 terrain_types=torch.arange(n) % 9,
                 terrain_origins=torch.arange(22*9*3).reshape(22, 9, 3).float(),
                 env_origins=torch.zeros(n, 3))
    pit = NS(**{name: torch.ones(n) for name in (
        'hold', 'success', 'climb_hold', 'climb_seen', 'climb_event', 'platform_clear',
        'settle_steps', 'stop_on_platform', 'front_seen', 'all_seen', 'progress',
        'physical', 'outside', 'body_contact')}, supported=torch.ones(n, 4), last_refresh=9, individual=None)
    command = NS(pit=pit, terrain_types=names+['stepping_stones'],
                 terrain_type2idx={name:i for i,name in enumerate(names+['stepping_stones'])})
    def refresh():
        pit.active = terrain.terrain_types == names.index('pit')
    command.refresh_membership = refresh
    env = NS(device='cpu', num_envs=n, command_manager=NS(get_term=lambda _: command),
             scene=NS(terrain=terrain, env_origins=torch.zeros(n,3)))
    runtime = AdaptiveAllocation(env, names, {'student':list(range(n))}, 1000)
    runtime.pending = {0:(7, names.index('wave')), 9:(5, names.index('slope_up'))}
    before = terrain.terrain_levels.clone()
    runtime.after_curriculum(torch.tensor([1,2]))
    assert torch.equal(before, terrain.terrain_levels)
    runtime.after_curriculum(torch.tensor([0,2]))
    assert terrain.terrain_levels[0] == 7 and not pit.active[0]
    assert pit.stop_on_platform[0] == 0 and pit.climb_event[0] == 0
    assert pit.supported[0].sum() == 0
    assert runtime.pending == {9:(5, names.index('slope_up'))}
    torch.testing.assert_close(env.scene.env_origins[0], terrain.terrain_origins[7, names.index('wave')])
    restored = AdaptiveAllocation(env, names, {'student':list(range(n))}, 1000,
                                  copy.deepcopy(runtime.state_dict()))
    assert restored.pending == runtime.pending
