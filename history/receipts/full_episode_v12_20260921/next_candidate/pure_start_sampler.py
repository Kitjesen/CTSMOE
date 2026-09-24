"""Pure tensor sampler for the prepared flat normal-reset candidate.

This module deliberately has no Isaac/production imports.  It returns explicit
startup goals only; the caller is responsible for mixing them with the existing
MixedCommand path and for preserving all non-selected commands unchanged.
"""
from __future__ import annotations

import torch


KIND_NAMES = (
    "forward", "backward", "left", "right", "yaw_left", "yaw_right",
    "diagonal_a", "diagonal_b", "longitudinal_turn", "lateral_turn",
)


def sample_explicit_goals(
    count: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(goals, kinds)`` for ten balanced explicit primitives.

    Kind ids are sampled uniformly. Magnitudes stay inside the existing V6
    anchor domain: vx in [.2, 1.2], vy in [.2, .5], yaw in [.3, .8].  Every
    combination component is signed symmetrically. Cardinal kinds 0..5 have
    fixed signs, paired across kinds; absent components are exactly zero.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    kinds = torch.randint(len(KIND_NAMES), (count,), device=device, generator=generator)
    vx = 0.2 + torch.rand(count, device=device, generator=generator)
    vy = 0.2 + 0.3 * torch.rand(count, device=device, generator=generator)
    yaw = 0.3 + 0.5 * torch.rand(count, device=device, generator=generator)
    signs = torch.where(torch.rand(count, 3, device=device, generator=generator) < 0.5, -1.0, 1.0)
    goals = torch.zeros(count, 3, device=device)
    goals[:, 0] = torch.where(kinds == 0, vx, torch.where(kinds == 1, -vx, 0.0))
    goals[:, 1] = torch.where(kinds == 2, vy, torch.where(kinds == 3, -vy, 0.0))
    goals[:, 2] = torch.where(kinds == 4, yaw, torch.where(kinds == 5, -yaw, 0.0))
    goals[:, 0] += torch.where((kinds == 6) | (kinds == 8), vx, torch.where(kinds == 7, -vx, 0.0))
    goals[:, 1] += torch.where((kinds == 6) | (kinds == 9), vy, torch.where(kinds == 7, -vy, 0.0))
    goals[:, 2] += torch.where((kinds == 8) | (kinds == 9), yaw, 0.0)
    # Kinds 0..5 have fixed semantic signs. Combination kinds only use random signs.
    combo = kinds >= 6
    goals = torch.where(combo[:, None], goals.abs() * signs, goals)
    return goals, kinds


def explicit_selection(cohort_size: int, *, device: torch.device | str = "cpu", generator: torch.Generator | None = None) -> torch.Tensor:
    """Select half of one complete fixed cohort once; reuse its global mask on resets."""
    if cohort_size < 0:
        raise ValueError("cohort_size must be non-negative")
    selected = torch.zeros(cohort_size, dtype=torch.bool, device=device)
    if cohort_size:
        order = torch.randperm(cohort_size, device=device, generator=generator)
        selected[order[: cohort_size // 2]] = True
    return selected


def reset_selection(reset_ids: torch.Tensor, global_mask: torch.Tensor) -> torch.Tensor:
    """Return selected reset ids from a precomputed global cohort mask."""
    ids = reset_ids.to(device=global_mask.device, dtype=torch.long)
    if ids.numel() and (ids.min() < 0 or ids.max() >= global_mask.numel()):
        raise IndexError("reset id outside global mask")
    return ids[global_mask[ids]]
