"""Shared invariants used by the corrected evaluator and CPU regression tests."""
from __future__ import annotations


def assign_columns(cases, ranges):
    """Use verified upstream column ranges, including disabled zero-width types."""
    assigned = []
    for case in cases:
        bounds = ranges.get(case["terrain"])
        if bounds is None:
            raise ValueError(f"No atlas range for {case['terrain']}")
        first, last = bounds
        col = case.get("terrain_column", first)
        if not first <= col < last:
            raise ValueError(f"Invalid or empty atlas range: {case['terrain']}: {bounds}")
        assigned.append(col)
    return assigned


def update_geometric_climb(hold, seen, eligible, dt):
    """Tensor/array-compatible milestone; strict parking remains separate."""
    new_hold = (hold + dt) * eligible
    return new_hold, seen | (new_hold >= 0.2 - 1e-7)
