"""Stronger support-relative height and zero-command nominal stance objectives."""
import torch

PROFILE = 'strong_stand_v1'


def zero_command(commands):
    # Translation zero with nonzero yaw is a turn, not a request to stand still.
    return (commands[:, :2].norm(dim=-1) <= .05) & (commands[:, 2].abs() <= .05)


def stationary_cost(q, default_q, tolerance, commands):
    # Unlike the locomotion dead band, every deviation costs something at stop.
    # Normalizing per joint family avoids treating calf and hip ranges alike.
    return ((q-default_q)/tolerance).square().mean(-1)*zero_command(commands)


def measure(env):
    from surface_posture import measure as surface_measure
    from pit_posture import measure as pit_measure
    command = env.command_manager.get_term('base_velocity')
    command.pit.refresh()
    surface = surface_measure(env)
    pit = pit_measure(command.pit)
    gate = torch.where(command.pit.active, pit['weight'], surface['height_weight'])
    if getattr(env.cfg, 'startup_repair_profile', 'off') in ('omni_start_v8', 'omni_start_v9', 'omni_start_v10'):
        from phase_transition import measure as phase_measure
        phase = phase_measure(env)
        gate = torch.where(phase['relevant'], phase['weight'], gate)
    robot = env.scene['robot']
    ids = env._surface_posture_ids
    cost = stationary_cost(robot.data.joint_pos[:, ids], robot.data.default_joint_pos[:, ids],
                           env._surface_posture_tolerance, command.command)
    return gate, cost, zero_command(command.command)


def reward(env):
    gate, cost, _ = measure(env)
    return gate*cost


def configure(cfg, profile):
    if profile not in ('off', PROFILE):
        raise ValueError('Unknown stance strength profile')
    cfg.stance_strength_profile = profile
    if profile == 'off':
        return
    if (getattr(cfg, 'surface_posture_profile', 'off') != 'level_surface_v1'
            or getattr(cfg, 'pit_posture_profile', 'off') != 'phase_stance_v1'):
        raise ValueError('Strong stance requires both validated surface and pit phase objectives')
    from isaaclab.managers import RewardTermCfg
    cfg.rewards.surface_stance_height.weight = -2.
    cfg.rewards.pit_stance_height.weight = -2.
    cfg.rewards.zero_command_nominal_stance = RewardTermCfg(func=reward, weight=-2.)


def snapshot(env):
    profile = getattr(env.cfg, 'stance_strength_profile', 'off')
    if profile == 'off':
        return None
    gate, cost, zero = measure(env)
    command = env.command_manager.get_term('base_velocity')
    active = (gate > .5) & zero
    result = dict(profile=profile,
                  height_weight=env.reward_manager.get_term_cfg('surface_stance_height').weight,
                  zero_command_weight=env.reward_manager.get_term_cfg('zero_command_nominal_stance').weight,
                  zero_xy_threshold_mps=.05, zero_yaw_threshold_rad_s=.05,
                  nominal_stance_deadband_rad=0., height_command_gating=False)
    for name, mask in [('all', active), ('pit', active & command.pit.active),
                       ('other', active & ~command.pit.active)]:
        result[name] = dict(count=int(mask.sum()))
        if bool(mask.any()):
            result[name]['mean_nominal_stance_cost'] = float(cost[mask].mean())
    return result
