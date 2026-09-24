"""V10 flat posture: retain continuous L1 and charge concentrated excursions.

The matched V9 startup trace concentrates 63% of squared pose error in three
legs' joints. Penalize the largest three normalized deviations symmetrically,
without identifying a preferred leg, adding a dead band, or clamping actions.
This is a bounded reward experiment, not a guarantee on physical joint peaks.
"""
import torch

TOP_K = 3
TAIL_WEIGHT = .5


def components(q, nominal, scales):
    normalized = (q-nominal).abs()/scales
    return normalized.mean(-1), normalized.topk(TOP_K, dim=-1).values.square().mean(-1)


def posture_cost(q, nominal, scales):
    mean, tail = components(q, nominal, scales)
    return mean+TAIL_WEIGHT*tail


def replace_flat_cost(previous, flat, q, nominal, scales):
    return torch.where(flat, posture_cost(q, nominal, scales), previous)


def reward(env, kind):
    from flat_posture import flat_mask
    from phase_transition import joint_reward
    if kind != 'surface_joint':
        raise ValueError('Flat outlier reward only replaces actual joint pose')
    previous = joint_reward(env, kind)
    data = env.scene['robot'].data
    ids = env._surface_posture_ids
    return replace_flat_cost(previous, flat_mask(env), data.joint_pos[:, ids],
                             data.default_joint_pos[:, ids], env._surface_posture_tolerance)


def configure(cfg):
    cfg.rewards.surface_joint_posture.func = reward
    cfg.rewards.surface_joint_posture.params = {'kind': 'surface_joint'}
    cfg.rewards.surface_joint_posture.weight = -2.


def snapshot(env):
    from flat_continuous import snapshot as baseline_snapshot
    from flat_posture import flat_mask
    result = baseline_snapshot(env)
    result.update(profile='flat_outlier_v1', top_k=TOP_K, tail_weight=TAIL_WEIGHT)
    data = env.scene['robot'].data
    ids = env._surface_posture_ids
    q, nominal = data.joint_pos[:, ids], data.default_joint_pos[:, ids]
    mean, tail = components(q, nominal, env._surface_posture_tolerance)
    mask = flat_mask(env)
    learner = env.command_manager.get_term('base_velocity').learner
    for name, group in [('teacher', 0), ('student', 1)]:
        selected = mask & (learner == group)
        if bool(selected.any()):
            result[name].update(cost=float((mean+TAIL_WEIGHT*tail)[selected].mean()),
                                mean_l1=float(mean[selected].mean()),
                                tail_square=float(tail[selected].mean()),
                                max_abs_joint_rad=float((q-nominal)[selected].abs().max()))
    return result
