"""12-DoF sagittal mirror for Lope. No wheels and no Torch dependency.

Go2's mapper flips hip signs only. That map does not fix Lope's standing pose:
left thigh/calf URDF axes point along -Y while the right axes point along +Y,
and the standing angles are odd under that reflection. Every revolute joint
therefore uses a minus after the leg permutation.
"""
from lope_lab3.spec import FOOT_PERM, FRAME_DIM, JOINT_NAMES, JOINT_PERM, JOINT_SIGN, STANDING_POSE


def permute(values, perm):
    if len(values) != len(perm):
        raise ValueError(f"Expected {len(perm)} values")
    return [values[index] for index in perm]


def mirror_joints(values):
    """Swap left/right legs and apply Lope's URDF sign mirror."""
    if len(values) != 12:
        raise ValueError("Lope symmetry acts on 12 revolute joints")
    swapped = permute(values, JOINT_PERM)
    return [sign * value for sign, value in zip(JOINT_SIGN, swapped)]


def standing_vector():
    return [STANDING_POSE[name] for name in JOINT_NAMES]


def _scale(values, signs):
    if len(values) != len(signs):
        raise ValueError("Vector width does not match its mirror signs")
    return [sign * value for sign, value in zip(signs, values)]


def mirror_policy_frame(frame):
    """45 values: ang, gravity, command, q, dq, previous action."""
    if len(frame) != FRAME_DIM:
        raise ValueError("Lope actor frame must contain 45 values")
    joints = mirror_joints(frame[9:21])
    velocity = mirror_joints(frame[21:33])
    actions = mirror_joints(frame[33:45])
    return (
        _scale(frame[0:3], (-1, 1, -1))
        + _scale(frame[3:6], (1, -1, 1))
        + _scale(frame[6:9], (1, -1, -1))
        + joints
        + velocity
        + actions
    )


def mirror_critic(frame, height_perm):
    """263 values. Foot-force magnitudes are permuted with no extra sign."""
    if len(frame) != 76 + len(height_perm):
        raise ValueError("Lope critic width does not match its height scan")
    height = permute(frame[76:], height_perm)
    feet = permute(frame[72:76], FOOT_PERM)
    return (
        _scale(frame[0:3], (1, -1, 1))
        + _scale(frame[3:6], (-1, 1, -1))
        + _scale(frame[6:9], (1, -1, 1))
        + _scale(frame[9:12], (1, -1, -1))
        + mirror_joints(frame[12:24])
        + mirror_joints(frame[24:36])
        + mirror_joints(frame[36:48])
        + mirror_joints(frame[48:60])
        + mirror_joints(frame[60:72])
        + feet
        + height
    )


def reflected_ray_indices(starts, tolerance=1e-6):
    """Match each yaw-aligned ray to its sagittal reflection. Must be an involution."""
    reflected = [(x, -y, z) for x, y, z in starts]
    indices = []
    for point in reflected:
        best = None
        best_error = None
        for index, candidate in enumerate(starts):
            error = sum((left - right) ** 2 for left, right in zip(point, candidate)) ** 0.5
            if best_error is None or error < best_error:
                best = index
                best_error = error
        if best_error > tolerance:
            raise ValueError("Height scan is not sagittally symmetric")
        indices.append(best)
    if len(set(indices)) != len(starts):
        raise ValueError("Height scan reflection is not bijective")
    if permute(indices, indices) != list(range(len(starts))):
        raise ValueError("Height scan reflection is not an involution")
    return indices
