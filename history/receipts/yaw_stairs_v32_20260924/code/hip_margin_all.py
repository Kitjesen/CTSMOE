"""Continuous pre-hard-limit cost on measured inward hip angles, all eight terrains."""

import torch

from hip_contract import INWARD_SIGNS

PROFILE = "actual_prehard_v1"
START_INWARD_RAD = 0.34
HARD_MARGIN_RAD = 0.07
WEIGHT = -2.0


def margin_cost(actual_hip_radians, inward_signs):
    inward = actual_hip_radians * inward_signs
    excess = ((inward - START_INWARD_RAD) / HARD_MARGIN_RAD).clamp_min(0.)
    return excess.square().sum(-1)


def reward(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_all_hip_margin_ids"):
        env._all_hip_margin_ids = [robot.joint_names.index(name) for name in INWARD_SIGNS]
        env._all_hip_margin_signs = torch.tensor(list(INWARD_SIGNS.values()), device=env.device)
    actual = robot.data.joint_pos[:, env._all_hip_margin_ids]
    return margin_cost(actual, env._all_hip_margin_signs)


def configure(cfg, profile):
    if profile == "off":
        return
    if profile != PROFILE:
        raise ValueError("Unknown all-terrain hip margin profile")
    from isaaclab.managers import RewardTermCfg

    cfg.rewards.all_hip_margin = RewardTermCfg(func=reward, weight=WEIGHT)


def validate(env, profile):
    if profile == "off":
        return {"profile": profile}
    term = env.reward_manager.get_term_cfg("all_hip_margin")
    if term.weight != WEIGHT or term.func is not reward:
        raise ValueError("All-terrain hip margin cost changed")
    values = reward(env)
    if values.shape != (env.num_envs,) or not bool(torch.isfinite(values).all()):
        raise ValueError("Invalid measured hip margin cost")
    return dict(profile=profile, weight=WEIGHT, onset_rad=START_INWARD_RAD,
                physical_hard_limit_rad=0.41, measured_actual_joint_angle=True,
                current_positive_envs=int((values > 0).sum()))
