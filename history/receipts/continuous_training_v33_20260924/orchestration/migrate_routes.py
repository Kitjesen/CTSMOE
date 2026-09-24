"""Explicit old-atlas to six named route columns; no optimizer changes."""
import copy
from collections import Counter
import random
import torch
from terrain_allocation import BASE, NO_PIT, NO_PIT_PROFILE, CONTINUOUS_WEIGHTS
from training_profiles import allocate_counts


def migrate(saved):
    if len(saved['terrain_levels']) != 8192 or saved['terrain_allocation']['pending']:
        raise ValueError('Expected 8192 source without pending allocation')
    if saved.get('continuous_routes_profile'):
        raise ValueError('Use direct restore for already migrated checkpoints')
    names = saved['terrain_column_names']
    if len(names) != 40 or set(names[20:]) != {'pit'}:
        raise ValueError('Unexpected original atlas')
    out = copy.deepcopy(saved)
    new_names = names[:20] + list(CONTINUOUS_WEIGHTS) + ['pit']*14
    rng = random.Random(330924)
    moved, counts = [], {}
    for learner in (0, 1):
        ids = [i for i in range(8192) if int(i % 4 == 0) == learner]
        targets = allocate_counts(len(ids), NO_PIT)
        free = []
        for name in dict.fromkeys(names[:20]):
            pool = [i for i in ids if names[int(saved['terrain_types'][i])] == name]
            if len(pool) < targets[name]:
                raise ValueError('Old terrain pool cannot satisfy reduced quota')
            rng.shuffle(pool)
            free.extend(pool[targets[name]:])
        rng.shuffle(free)
        cursor = 0
        for column, name in enumerate(CONTINUOUS_WEIGHTS, 20):
            chosen = free[cursor:cursor+targets[name]]
            cursor += targets[name]
            out['terrain_types'][chosen] = column
            # New task difficulty is independent of the donor's old terrain.
            out['terrain_levels'][chosen] = 0
            moved.extend(chosen)
        if cursor != len(free):
            raise ValueError('Route quotas did not consume the exact free pool')
        observed = Counter(new_names[int(out['terrain_types'][i])] for i in ids)
        assert all(observed[name] == targets[name] for name in targets)
        counts[str(learner)] = dict(observed)
    out['terrain_column_names'] = new_names
    allocation = out['terrain_allocation']
    oldstats = saved['terrain_allocation']['stats']
    assert oldstats.shape == (9, 22, 6, 3, 3)
    allocation['stats'] = torch.cat((oldstats, torch.zeros((6, *oldstats.shape[1:]), dtype=oldstats.dtype)), 0)
    course = allocation['course']
    course['profile'], course['weights'] = NO_PIT_PROFILE, dict(NO_PIT)
    course['bad_streak'] = {name: course['bad_streak'].get(name, 0) for name in BASE}
    out['continuous_routes_profile'] = 'continuous_routes_v33'
    out['continuous_routes_initial_migration'] = True
    retained = torch.ones(8192, dtype=torch.bool)
    retained[moved] = False
    assert torch.equal(out['terrain_levels'][retained], saved['terrain_levels'][retained])
    assert torch.equal(out['terrain_types'][retained], saved['terrain_types'][retained])
    for key in ('model_state_dict', 'optimizer_state_dict', 'optimizer_stu_enc_state_dict'):
        # Fingerprint includes nested tensors and scalar optimizer parameters.
        from state_guard import state_sha256
        assert state_sha256(out[key], torch) == state_sha256(saved[key], torch)
    return out, dict(moved=len(moved), retained=int(retained.sum()), by_learner=counts,
                     original_difficulty_retained=True, new_routes_start_level=0,
                     model_dual_optimizers_unchanged=True)
