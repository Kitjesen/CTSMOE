"""Bounded terrain sampling; compare matched training strata, never raw levels."""
from collections import Counter
from dataclasses import dataclass, field, asdict
import random

from training_profiles import allocate_counts

PROFILE = 'balanced_adaptive_v1'
STARTUP_PROFILE = 'startup_flat8_v1'
BASE = dict(pit=.40, stairs_up=.20, stairs_down=.10, wave=.06,
            slope_up=.06, slope_down=.05, rough_slope=.05, obstacles=.05, flat=.03)
FLOOR = dict(pit=.35, stairs_up=.15, stairs_down=.08, wave=.04,
             slope_up=.04, slope_down=.03, rough_slope=.04, obstacles=.03, flat=.03)
CEILING = dict(pit=.45, stairs_up=.25, stairs_down=.15, wave=.12,
               slope_up=.12, slope_down=.10, rough_slope=.12, obstacles=.10, flat=.06)


def bounds(profile=PROFILE):
    if profile == PROFILE:
        return FLOOR, CEILING
    if profile == STARTUP_PROFILE:
        return {**FLOOR, 'flat': .08}, {**CEILING, 'flat': .12}
    raise ValueError('Unknown allocation bounds profile')


def validate_weights(weights, profile=PROFILE):
    floor, ceiling = bounds(profile)
    if set(weights) != set(BASE) or abs(sum(weights.values())-1) > 1e-8:
        raise ValueError('Terrain weights must cover nine types and sum to one')
    if any(not floor[k]-1e-8 <= v <= ceiling[k]+1e-8 for k, v in weights.items()):
        raise ValueError('Terrain weight outside bounded allocation')


def reassign(levels, columns, names, groups, weights, seed=73, profile=PROFILE):
    """Keep retained states; newcomers copy a same-terrain/same-learner donor."""
    validate_weights(weights, profile)
    if sorted(i for ids in groups.values() for i in ids) != list(range(len(levels))):
        raise ValueError('Learners must partition environments')
    rng = random.Random(seed)
    new_levels, new_columns = list(levels), list(columns)
    moves, counts = [], {}
    for group, ids in groups.items():
        pools = {name: [i for i in ids if names[columns[i]] == name] for name in BASE}
        if any(not pool for pool in pools.values()):
            raise ValueError('Every learner needs donors for all terrain types')
        targets = allocate_counts(len(ids), weights)
        free = []
        for name, pool in pools.items():
            shuffled = rng.sample(pool, len(pool))
            free.extend(shuffled[targets[name]:])
        rng.shuffle(free)
        cursor = 0
        for name, pool in pools.items():
            for _ in range(max(0, targets[name]-len(pool))):
                i, donor = free[cursor], rng.choice(pool)
                cursor += 1
                new_levels[i], new_columns[i] = levels[donor], columns[donor]
                moves.append([i, donor])
        assert cursor == len(free)
        counts[group] = dict(Counter(names[new_columns[i]] for i in ids))
        assert counts[group] == targets
    return new_levels, new_columns, dict(profile=PROFILE, target_weights=weights,
                                         group_counts=counts, moves=moves,
                                         retained_environments=len(levels)-len(moves),
                                         difficulty_source='same terrain and learner donors')


def compare(reference, current):
    """Equal-stratum comparison prevents a changing level mix dominating signals.

    Each stratum is physical terrain row, episode-mean XY speed and yaw bin.
    This is observational stochastic training evidence, not held-out evaluation.
    """
    keys = [k for k in reference.keys() & current.keys()
            if min(reference[k][0], current[k][0]) >= 16]
    n = sum(current[k][0] for k in keys)
    if n < 128 or sum(reference[k][0] for k in keys) < 128:
        return dict(ready=False, matched_episodes=n)
    def average(data, index):
        return sum(data[k][index]/data[k][0] for k in keys)/len(keys)
    fail0, fail = average(reference, 1), average(current, 1)
    err0, err = average(reference, 2), average(current, 2)
    bad = fail-fail0 >= .03 or err-err0 >= max(.04, .2*err0)
    return dict(ready=True, matched_episodes=n, strata=len(keys),
                failure=fail, reference_failure=fail0, error=err, reference_error=err0,
                bad=bad, severity=max((fail-fail0)/.03, (err-err0)/max(.04, .2*err0)))


@dataclass
class AllocationCourse:
    profile: str = PROFILE
    weights: dict = field(default_factory=lambda: dict(BASE))
    reference: dict = field(default_factory=dict)
    bad_streak: dict = field(default_factory=lambda: dict.fromkeys(BASE, 0))
    last_check: int = 0
    decisions: list = field(default_factory=list)

    def evaluate(self, current, iteration):
        validate_weights(self.weights, self.profile)
        floor, ceiling = bounds(self.profile)
        scores = {}
        for name in BASE:
            now = current.get(name, {})
            before = self.reference.setdefault(name, {})
            scores[name] = compare(before, now)
            score = scores[name]
            self.bad_streak[name] = self.bad_streak[name]+1 if score.get('bad') else 0
            # Fill unseen or under-sampled strata, but do not ratchet an established
            # baseline downward when a skill deteriorates.
            for key, values in now.items():
                if values[0] >= 16 and before.get(key, [0])[0] < 16:
                    before[key] = list(values)
        receivers = [k for k in BASE if self.bad_streak[k] >= 2
                     and self.weights[k] < ceiling[k]-1e-8]
        donors = [k for k in BASE if scores[k].get('ready') and not scores[k]['bad']
                  and self.weights[k] > floor[k]+1e-8]
        move = None
        if receivers and donors:
            receiver = max(receivers, key=lambda k: scores[k]['severity'])
            donor = max(donors, key=lambda k: self.weights[k]-floor[k])
            amount = min(.02, self.weights[donor]-floor[donor], ceiling[receiver]-self.weights[receiver])
            self.weights[donor] -= amount
            self.weights[receiver] += amount
            self.bad_streak[receiver] = 0
            move = dict(donor=donor, receiver=receiver, fraction=amount)
        validate_weights(self.weights, self.profile)
        self.last_check = iteration
        decision = dict(iteration=iteration, scores=scores, move=move, weights=dict(self.weights))
        self.decisions.append(decision)
        return decision

    def state_dict(self):
        return asdict(self)
