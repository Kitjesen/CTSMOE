"""Finite post-climb traversal contract for a continuously commanded pit exit."""

import torch


PROFILE = "platform_traverse_v1"
TRAVERSE_DISTANCE_M = 0.6
MIN_TRAVERSE_S = 0.5
MIN_TRACKED_S = 0.5
MAX_SUCCESS_ROOT_X_M = 3.7  # The existing out-of-lane guard is at x=4.0.
MIN_TRACKING_TOLERANCE_MPS = 0.08
RELATIVE_TRACKING_TOLERANCE = 0.25


def command_speed(held_speed, stop_on_platform, profile):
    """Keep the selected forward command after entry for the traverse task."""
    if profile == PROFILE:
        return held_speed
    return torch.where(stop_on_platform, torch.zeros_like(held_speed), held_speed)


def tracking_tolerance(command_vx):
    return torch.maximum(
        torch.full_like(command_vx, MIN_TRACKING_TOLERANCE_MPS),
        RELATIVE_TRACKING_TOLERANCE * command_vx,
    )


def finish(
    entered, entry_x, root_x, elapsed_s, tracked_s, error_integral,
    command_vx, supported, body_clear, upright, physical, outside, started_in_pit,
):
    """Require a bounded platform traverse and real speed tracking, not parking.

    The four wheels need not all carry force on every gait step. At the finish,
    one front and one rear wheel must support the robot on the platform.
    """
    tracking_mae = error_integral / tracked_s.clamp_min(1e-6)
    balanced_support = supported[:, :2].any(-1) & supported[:, 2:].any(-1)
    return (
        entered
        & ((root_x - entry_x) >= TRAVERSE_DISTANCE_M)
        & (elapsed_s >= MIN_TRAVERSE_S)
        & (tracked_s >= MIN_TRACKED_S)
        & (root_x <= MAX_SUCCESS_ROOT_X_M)
        & (tracking_mae <= tracking_tolerance(command_vx))
        & balanced_support
        & body_clear
        & upright
        & ~physical
        & ~outside
        & started_in_pit
    )


def platform_tracking_reward(entered, body_clear, command_vx, actual_vx, supported):
    """Reward forward tracking on the top while retaining contact evidence."""
    error = actual_vx - command_vx
    front_and_rear = supported[:, :2].any(-1) & supported[:, 2:].any(-1)
    return (entered & body_clear & front_and_rear).float() * torch.exp(-error.square() / 0.04)
