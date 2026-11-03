"""V9: replace flat actual-pose dead bands with continuous normalized L1.

The 43,501-step matched probe shows a persistent left-rear thigh displacement
and crouch despite fully active height shaping. The former tolerances are now
scales, not free ranges. This is a reward change, never a target/state clamp.
"""
import torch


def continuous_cost(q, nominal, scales):
    return ((q-nominal).abs()/scales).mean(-1)


def replace_flat_cost(previous, flat, q, nominal, scales):
    return torch.where(flat, continuous_cost(q, nominal, scales), previous)


def reward(env, kind):
    from flat_posture import flat_mask
    from phase_transition import joint_reward
    previous = joint_reward(env, kind)
    if kind != 'surface_joint':
        raise ValueError('Flat continuous repair only replaces actual joint pose')
    robot = env.scene['robot'].data
    ids = env._surface_posture_ids
    # No command-speed or scanner-relief gate on an explicitly flat terrain.
    # Every non-flat reward, including stair landing and pit phases, is exact V8.
    return replace_flat_cost(previous, flat_mask(env), robot.joint_pos[:, ids],
                             robot.default_joint_pos[:, ids], env._surface_posture_tolerance)


def configure(cfg):
    cfg.rewards.surface_joint_posture.func = reward
    cfg.rewards.surface_joint_posture.params = {'kind': 'surface_joint'}
    cfg.rewards.surface_joint_posture.weight = -2.


def snapshot(env):
    from flat_posture import flat_mask
    from surface_posture import measure
    state = measure(env)
    data = env.scene['robot'].data
    ids = env._surface_posture_ids
    cost = continuous_cost(data.joint_pos[:, ids], data.default_joint_pos[:, ids],
                           env._surface_posture_tolerance)
    mask = flat_mask(env)
    command = env.command_manager.get_term('base_velocity')
    result = dict(profile='flat_continuous_v1', weight=-2., deadband_rad=0.,
                  command_gating=False, replaces='surface_joint_posture on flat only')
    for name, group in [('teacher', 0), ('student', 1)]:
        selected = mask & (command.learner == group)
        result[name] = dict(count=int(selected.sum()))
        if bool(selected.any()):
            result[name].update(cost=float(cost[selected].mean()),
                                rms_rad=float(state['rms_rad'][selected].mean()),
                                height_m=float(state['height_m'][selected].mean()))
    return result
