"""Opt-in 42-column atlas with protected V23 terrain-state migration.

The old 20 locomotion columns retain their indices. Two compact composite
columns follow; the 20 pit columns move from 20..39 to 22..41. No policy or
optimizer tensor is changed. Five percent of flat donors move to the new
composite category; all old pit and stair assignments retain their levels.
"""

from __future__ import annotations

import copy
import random
from collections import Counter

import torch

from composite_route import composite_terrain_9x9
from stair_transition_coverage import EXPECTED_COLUMNS as OLD_COLUMNS
from stair_transition_coverage import column_names

PROFILE = "composite_flat35_v25"
OLD_PIT_FIRST = 20
NEW_COMPOSITE_COLUMNS = (20, 21)
NEW_PIT_FIRST = 22
NEW_COLUMNS = (*OLD_COLUMNS[:20], "composite", "composite", *("pit",) * 20)
assert len(OLD_COLUMNS) == 40 and len(NEW_COLUMNS) == 42


def configure_atlas(cfg, subterrain_cls=None):
    """Call after configure_mixed and before coverage and environment creation."""
    if subterrain_cls is None:
        from isaaclab.terrains import SubTerrainBaseCfg

        subterrain_cls = SubTerrainBaseCfg

    terrain = cfg.scene.terrain.terrain_generator
    if (tuple(terrain.size) != (9.0, 9.0) or terrain.num_rows != 22
            or terrain.num_cols != 40 or column_names(terrain) != OLD_COLUMNS):
        raise ValueError("Expected exact V23 22x40 mixed atlas before composite insertion")
    if "composite" in terrain.sub_terrains or list(terrain.sub_terrains)[-1] != "pit":
        raise ValueError("Unexpected terrain definition order")
    original = list(terrain.sub_terrains.items())
    pit = terrain.sub_terrains["pit"]
    for name, sub in original:
        if name != "pit":
            sub.proportion *= 40.0 / 42.0
    pit.proportion = 20.0 / 42.0
    terrain.sub_terrains.clear()
    terrain.sub_terrains.update((name, sub) for name, sub in original if name != "pit")
    terrain.sub_terrains["composite"] = subterrain_cls(
        function=composite_terrain_9x9, proportion=2.0 / 42.0)
    terrain.sub_terrains["pit"] = pit
    terrain.num_cols = 42
    if column_names(terrain) != NEW_COLUMNS:
        raise ValueError("New atlas columns did not preserve old terrain ordering")
    return dict(profile=PROFILE, old_columns=list(OLD_COLUMNS), new_columns=list(NEW_COLUMNS),
                old_pit_columns=list(range(20, 40)), new_pit_columns=list(range(22, 42)),
                composite_columns=list(NEW_COMPOSITE_COLUMNS), geometry_size_m=[9, 9])


def migrate_assignments(levels, columns, learner_groups, seed=250923):
    """Take 5% from flat in each learner; keep every pit/stair level and sample."""
    if len(levels) != len(columns):
        raise ValueError("Unequal checkpoint terrain arrays")
    if any(not 0 <= col < len(OLD_COLUMNS) for col in columns):
        raise ValueError("Expected old 40-column terrain state")
    if sorted(i for ids in learner_groups.values() for i in ids) != list(range(len(levels))):
        raise ValueError("Learner groups must partition every environment")
    before = Counter(OLD_COLUMNS[col] for col in columns)
    new_levels = list(levels)
    new_columns = [col + 2 if col >= OLD_PIT_FIRST else col for col in columns]
    selected = []
    for group, ids in learner_groups.items():
        flat = [i for i in ids if OLD_COLUMNS[columns[i]] == "flat"]
        quota = int(len(ids) * 0.05 + 0.5)
        if len(flat) < quota:
            raise ValueError(f"Insufficient flat donors in {group}")
        group_rng = random.Random(seed + (0 if group == "teacher" else 1))
        chosen = sorted(group_rng.sample(flat, quota))
        for j, i in enumerate(chosen):
            new_columns[i] = NEW_COMPOSITE_COLUMNS[j % 2]
            new_levels[i] = 0  # fixed first-stage geometry is not level 0..9 difficulty
        selected.extend(chosen)
    after = Counter(NEW_COLUMNS[col] for col in new_columns)
    if after["pit"] != before["pit"] or any(
        after[name] != before[name] for name in ("stairs_up", "stairs_down", "slope_up", "slope_down")):
        raise AssertionError("Migration changed a protected terrain quota")
    if any((new_columns[i] != columns[i] + 2 or new_levels[i] != levels[i])
           for i in range(len(levels)) if OLD_COLUMNS[columns[i]] == "pit"):
        raise AssertionError("Pit level or order was lost")
    return new_levels, new_columns, dict(
        profile=PROFILE, seed=seed, old_type_counts=dict(before), new_type_counts=dict(after),
        selected_from_flat=selected, selected_count=len(selected),
        preserved_pit_count=before["pit"],
        retained_terrain_levels=sum(a == b for a, b in zip(levels, new_levels)),
        migration_kind="only category/atlas origin for selected flat; old pit column offset +2",
    )


def migrate_allocation_state(saved):
    """Expand a fixed V19 nine-class course/strata to ten classes exactly once."""
    from terrain_allocation import EARLY_BALANCED, EARLY_BALANCED_PROFILE

    if saved.get("profile") != "balanced_adaptive_v1" or saved.get("pending"):
        raise ValueError("Only settled V23 adaptive allocation may migrate")
    course = saved["course"]
    if course.get("profile") != EARLY_BALANCED_PROFILE or course.get("weights") != EARLY_BALANCED:
        raise ValueError("Expected frozen V23 fixed 40% flat course")
    if saved["stats"].shape != (9, 22, 6, 3, 3):
        raise ValueError("Old adaptive strata shape changed")
    result = copy.deepcopy(saved)
    result["stats"] = torch.cat((saved["stats"], torch.zeros_like(saved["stats"][:1])), dim=0)
    migrated = result["course"]
    migrated["profile"] = PROFILE
    migrated["weights"] = {**EARLY_BALANCED, "flat": 0.35, "composite": 0.05}
    migrated["reference"]["composite"] = {}
    migrated["bad_streak"]["composite"] = 0
    result["migration"] = PROFILE
    return result
