"""CPU-only slope angles and conservative migration for the mixed terrain atlas."""

from collections import Counter
import math


SLOPE_PROFILE = "slope45_v1"
SLOPE_ANGLES_DEG = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 42.5, 45.0)
SLOPE_NAMES = frozenset(("slope_up", "slope_down", "rough_slope"))
ATLAS_ROWS = 22
OLD_ROWS = 10


def level_from_difficulty(difficulty):
    """Resolve the shared 22-row atlas, including difficulty exactly equal to one."""
    if not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("Difficulty must be finite and in [0, 1]")
    return min(math.floor(ATLAS_ROWS * difficulty), ATLAS_ROWS - 1)


def angle_for_level(level):
    if isinstance(level, bool) or int(level) != level or not 0 <= level < ATLAS_ROWS:
        raise ValueError("Atlas level must be an integer in [0, 21]")
    return SLOPE_ANGLES_DEG[min(int(level), len(SLOPE_ANGLES_DEG) - 1)]


def slope_for_level(level):
    return math.tan(math.radians(angle_for_level(level)))


def slope_range_for_difficulty(difficulty):
    """Use a fixed nominal slope for each row of a height-field subterrain."""
    slope = slope_for_level(level_from_difficulty(difficulty))
    return slope, slope


def _old_slope_lower_angle(level):
    # The old row spans [level / 10, (level + 1) / 10). Its minimum slope
    # defines the largest safe discrete replacement angle.
    return math.degrees(math.atan(0.1 + 0.468 * level / OLD_ROWS))


def _conservative_level(level):
    lower_angle = _old_slope_lower_angle(level)
    return max(index for index, angle in enumerate(SLOPE_ANGLES_DEG) if angle <= lower_angle)


def migrate_slope_levels(levels, columns, names, from_profile, to_profile):
    """Preserve environment/column identity while replacing old slope rows.

    ``original`` denotes the original ten-row slope geometry. A slope45
    checkpoint is returned unchanged. Other terrain types, including
    pit, retain both row and column. The receipt includes exact row histograms
    and nominal angle/pit-height ranges for auditing.
    """
    if from_profile not in ("original", SLOPE_PROFILE):
        raise ValueError(f"Unknown slope profile: {from_profile!r}")
    if to_profile != SLOPE_PROFILE:
        raise ValueError(f"Unsupported target slope profile: {to_profile!r}")
    if len(levels) != len(columns):
        raise ValueError("Levels and columns must have the same length")
    result = list(levels)
    before = {name: Counter() for name in SLOPE_NAMES}
    after = {name: Counter() for name in SLOPE_NAMES}
    old_lower_angles = []
    new_angles = []
    pit_heights = []
    changed = 0
    for index, (level, column) in enumerate(zip(levels, columns, strict=True)):
        angle_for_level(level)
        if isinstance(column, bool) or int(column) != column or not 0 <= column < len(names):
            raise ValueError(f"Invalid terrain column: {column!r}")
        name = names[int(column)]
        if name == "pit":
            pit_heights.append(0.15 + 0.05 * int(level))
        if name not in SLOPE_NAMES:
            continue
        before[name][int(level)] += 1
        if from_profile == SLOPE_PROFILE:
            mapped = int(level)
        else:
            if level >= OLD_ROWS:
                raise ValueError(f"Old slope level outside [0, 9]: {level!r}")
            old_lower_angles.append(_old_slope_lower_angle(int(level)))
            mapped = _conservative_level(int(level))
        result[index] = mapped
        after[name][mapped] += 1
        new_angles.append(angle_for_level(mapped))
        changed += mapped != level

    def stats(values):
        return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)} if values else None

    receipt = {
        "from_profile": from_profile,
        "to_profile": to_profile,
        "slope_envs": len(new_angles),
        "changed_envs": changed,
        "counts_before_by_type": {name: dict(sorted(counts.items())) for name, counts in sorted(before.items())},
        "counts_after_by_type": {name: dict(sorted(counts.items())) for name, counts in sorted(after.items())},
        "old_lower_angle_deg": stats(old_lower_angles),
        "new_nominal_angle_deg": stats(new_angles),
        "pit_height_m": stats(pit_heights),
    }
    return result, receipt
