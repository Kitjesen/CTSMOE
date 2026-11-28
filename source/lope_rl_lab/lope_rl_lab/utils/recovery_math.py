"""Fallen-pose sampling and upright progress math reserved for ``tasks/recovery/``."""
import math

FALLEN_MODES = ("left_side", "right_side", "back", "belly")


def fallen_orientation(mode, roll_jitter, pitch_jitter, yaw):
    """One fallen root orientation.

    ``roll_jitter`` and ``pitch_jitter`` are in [-1, 1]. Side and back poses
    use large tilt angles; belly is a chest-down pitch short of full inversion.
    """
    if mode not in FALLEN_MODES:
        raise ValueError("Unknown fallen mode: " + str(mode))
    if not -1.0 <= roll_jitter <= 1.0 or not -1.0 <= pitch_jitter <= 1.0:
        raise ValueError("Orientation jitter must be in [-1, 1]")
    if mode == "left_side":
        roll = 1.55 + 0.35 * roll_jitter
        pitch = 0.35 * pitch_jitter
    elif mode == "right_side":
        roll = -1.55 + 0.35 * roll_jitter
        pitch = 0.35 * pitch_jitter
    elif mode == "back":
        roll = math.pi + 0.4 * roll_jitter
        if roll > math.pi:
            roll -= 2.0 * math.pi
        pitch = 0.4 * pitch_jitter
    else:
        roll = 0.25 * roll_jitter
        pitch = 1.45 + 0.35 * pitch_jitter
    return roll, pitch, yaw


def upright_score(gravity_z):
    """Map projected gravity z from back (+1) through upright (-1) onto [0, 1]."""
    return max(0.0, min(1.0, (1.0 - gravity_z) * 0.5))


def upright_deficit(gravity_z, saturation):
    """Continuous non-negative upright deficit: 0 once upright_score >= saturation."""
    return max(0.0, saturation - upright_score(gravity_z))


def recovering(gravity_z, saturation):
    """True while the base is not yet counted as standing."""
    return upright_score(gravity_z) < saturation
