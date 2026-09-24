"""Explicit Thunder training profiles; leave the pinned terrain geometry intact."""
from collections import Counter
import math
import random


def configure_base_height_penalty(cfg, mode):
    if mode == "original":
        return
    if mode != "off":
        raise ValueError(f"Unknown base height penalty mode: {mode}")
    cfg.rewards.base_height_l2.weight = 0.0
    cfg.curriculum.base_height_l2 = None


def terrain_profile_weights(column_names, profile):
    counts = Counter(column_names)
    if profile == "original":
        return {name: count / len(column_names) for name, count in counts.items()}
    if profile != "stairs80_v1":
        raise ValueError(f"Unknown terrain training profile: {profile}")
    if not {"stairs_up", "stairs_down"} <= counts.keys():
        raise ValueError("The stairs profile requires both original stair terrain types")
    others = sum(count for name, count in counts.items() if name not in ("stairs_up", "stairs_down"))
    if not others:
        raise ValueError("The stairs profile requires non-stair review terrain")
    return {name: (0.6 if name == "stairs_up" else 0.2 if name == "stairs_down" else 0.2*count/others)
            for name, count in counts.items()}


def allocate_counts(total, weights):
    """Largest remainders keep every learner group close to the same profile."""
    if not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("Terrain weights must sum to one")
    raw = {name: total*weight for name, weight in weights.items()}
    result = {name: math.floor(value) for name, value in raw.items()}
    order = sorted(raw, key=lambda name: raw[name]-result[name], reverse=True)
    for name in order[:total-sum(result.values())]:
        result[name] += 1
    return result


def rebalance_terrain_assignments(levels, columns, column_names, learner_groups, profile, seed=42):
    """Keep matching assignments; additions copy same-terrain/same-learner donors.

    Terrain rows on different terrain types are not interchangeable. A new stairs
    environment copies a saved stairs donor's column and level, never its own old
    slope/flat level. This preserves the existing skill difficulty distribution.
    """
    if len(levels) != len(columns):
        raise ValueError("Terrain state dimensions differ")
    indices = [i for ids in learner_groups.values() for i in ids]
    if sorted(indices) != list(range(len(levels))):
        raise ValueError("Learner groups must partition every environment exactly once")
    if any(not 0 <= col < len(column_names) for col in columns):
        raise ValueError("Invalid terrain column")
    weights = terrain_profile_weights(column_names, profile)
    rng = random.Random(seed)
    new_levels, new_columns = list(levels), list(columns)
    moves, group_counts = [], {}
    for group, ids in learner_groups.items():
        targets = allocate_counts(len(ids), weights)
        donors = {name: [i for i in ids if column_names[columns[i]] == name] for name in weights}
        retained, free = {}, []
        for name, available in donors.items():
            shuffled = list(available)
            rng.shuffle(shuffled)
            retained[name] = min(len(shuffled), targets[name])
            free.extend(shuffled[targets[name]:])
        rng.shuffle(free)
        cursor = 0
        for name, target in targets.items():
            missing = target-retained[name]
            available = list(donors[name])
            if missing and not available:
                raise ValueError(f"No saved {name} donor in {group} environments")
            rng.shuffle(available)
            for index in range(missing):
                destination, donor = free[cursor], available[index % len(available)]
                cursor += 1
                new_levels[destination], new_columns[destination] = levels[donor], columns[donor]
                moves.append({"env_id": destination, "donor_env_id": donor, "learner": group,
                              "from": column_names[columns[destination]], "to": name,
                              "old_level": levels[destination], "new_level": levels[donor]})
        assert cursor == len(free)
        actual = Counter(column_names[new_columns[i]] for i in ids)
        assert all(actual[name] == targets[name] for name in targets)
        group_counts[group] = {name: actual[name] for name in targets}
    return new_levels, new_columns, {
        "profile": profile, "target_weights": weights, "group_counts": group_counts,
        "retained_environments": len(levels)-len(moves), "reassigned_environments": len(moves),
        "difficulty_source": "saved same-terrain, same-learner donors", "moves": moves,
    }


def summarize_terrain(levels, columns, column_names, slope_profile="original"):
    result = {}
    for name in dict.fromkeys(column_names):
        subset = [level for level, col in zip(levels, columns) if column_names[col] == name]
        if subset:
            result[name] = {"count": len(subset), "mean": sum(subset)/len(subset),
                            "counts_L0_to_L9": [subset.count(level) for level in range(10)]}
            if name == "pit":
                result[name].pop("counts_L0_to_L9")
                result[name].update(counts_L0_to_L21=[subset.count(level) for level in range(22)],
                                    mean_height_m=.15+.05*result[name]["mean"],
                                    curriculum_fraction=result[name]["mean"]/21)
            elif name == "composite":
                from composite_route import spec_for_level

                if any(not 0 <= level < 10 for level in subset):
                    raise ValueError("Composite route difficulty must be in 0..9")
                result[name].update(
                    geometry="fixed_stations_bounded_angle_and_riser_v25",
                    curriculum_fraction=result[name]["mean"] / 9,
                    nominal_angle_deg_mean=sum(spec_for_level(level).slope_angle_deg
                                               for level in subset) / len(subset),
                    riser_height_m_mean=sum(spec_for_level(level).stair_height_m
                                            for level in subset) / len(subset),
                    curriculum_max_angle_deg=20., curriculum_max_riser_height_m=.12)
            else:
                result[name]["curriculum_fraction"] = result[name]["mean"]/9
            if slope_profile == "slope45_v1" and name in ("slope_up", "slope_down", "rough_slope"):
                from slope_curriculum import angle_for_level
                angles = [angle_for_level(level) for level in subset]
                result[name].update(nominal_angle_deg_mean=sum(angles)/len(angles),
                                    nominal_angle_deg_max=max(angles), curriculum_max_angle_deg=45.)
    return result
