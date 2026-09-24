"""Reward candidate for inward hip requests that cannot change the limited target.

This does not change action processing or make the new 0.12-rad default a limit.
The existing positive inward soft limit is 0.369 rad on each mirrored hip.
"""

import torch

from hip_contract import INWARD_SIGNS, SOFT_LIMIT


PROFILE = "inward_unreachable_request_v1"
# V21 fixed traces: mean summed invalid inward request was 2.49 rad on flat
# and 3.77-4.80 rad on selected stairs, so this starts near 0.05-0.10 raw
# reward units per control step, before the reward manager's step-time scaling.
WEIGHT = -0.02


def excess_per_environment(requested, indices, signs, inward_limit=SOFT_LIMIT):
    """Sum only inward request exceeding the already-enforced soft target bound."""
    if requested.ndim != 2 or len(indices) != 4 or len(signs) != 4:
        raise ValueError("Expected batched actions and exactly four hip joints")
    return (requested[:, indices] * signs - inward_limit).clamp_min(0).sum(dim=1)


def reward(env):
    term = env.action_manager.get_term("joint_pos")
    robot = env.scene["robot"]
    joint_indices = list(term.joint_indices)
    indices = [joint_indices.index(robot.joint_names.index(name)) for name in INWARD_SIGNS]
    signs = torch.tensor(list(INWARD_SIGNS.values()), device=env.device,
                         dtype=term.requested_actions.dtype)
    return excess_per_environment(term.requested_actions, indices, signs)


def configure(cfg, profile):
    if profile == "off":
        cfg.hip_request_profile = profile
        return
    if profile != PROFILE:
        raise ValueError(f"Unknown hip request profile: {profile}")
    from isaaclab.managers import RewardTermCfg

    cfg.rewards.hip_inward_unreachable_request = RewardTermCfg(func=reward, weight=WEIGHT)
    cfg.hip_request_profile = profile


def snapshot(env):
    profile = getattr(env.cfg, "hip_request_profile", "off")
    if profile == "off":
        return {"profile": "off", "weight": 0.0}
    actual = env.reward_manager.get_term_cfg("hip_inward_unreachable_request").weight
    if profile != PROFILE or actual != WEIGHT:
        raise ValueError("Hip request reward differs from frozen candidate")
    return {"profile": profile, "weight": actual,
            "soft_inward_limit_rad": SOFT_LIMIT,
            "default_is_not_a_limit": True,
            "meaning": "cost of raw inward request above existing reachable target bound"}
