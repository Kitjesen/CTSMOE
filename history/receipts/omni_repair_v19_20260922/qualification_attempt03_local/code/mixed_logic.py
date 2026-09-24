"""Deterministic migration from saved eight-terrain learner distributions."""
from collections import Counter
import random

from training_profiles import allocate_counts


def mixed_assignments(levels, columns, names, groups, seed=73):
    if len(names) != 20 or len(levels) != len(columns):
        raise ValueError("Expected saved 20-column original task")
    if sorted(i for ids in groups.values() for i in ids) != list(range(len(levels))):
        raise ValueError("Learner groups must partition all saved environments")
    new_levels, new_columns = list(levels), list(columns)
    rng = random.Random(seed)
    receipt = {}
    for group, ids in groups.items():
        counts = Counter(names[columns[i]] for i in ids)
        if len(counts) != 8:
            raise ValueError("Each learner must retain all eight original terrain types")
        weights = {name: count/len(ids)*.5 for name, count in counts.items()}
        weights["pit"] = .5
        targets = allocate_counts(len(ids), weights)
        freed = []
        for name in counts:
            members = [i for i in ids if names[columns[i]] == name]
            rng.shuffle(members)
            freed.extend(members[targets[name]:])
        if len(freed) != targets["pit"]:
            raise ValueError("Pit allocation mismatch")
        for j, i in enumerate(freed):
            new_columns[i], new_levels[i] = 20+j % 20, 2
        receipt[group] = targets
    return new_levels, new_columns, receipt
