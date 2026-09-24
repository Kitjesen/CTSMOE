"""Normal stance on the pit floor and after climbing, with an unconstrained climb phase."""
import torch

PROFILE = 'phase_stance_v1'
JOINT_TOLERANCE = {'hip': .25, 'thigh': .45, 'calf': .55}
TARGET_HEIGHT = .46
HEIGHT_TOLERANCE = .10


def phase_weight(front_gap, on_platform, active):
    # Fade the floor-stance cost over 0.75 -> 0.25 m wheel-to-wall clearance.
    # Close to the wall it is zero, allowing the reach and pull-up posture.
    approach = ((front_gap-.25)/.5).clamp(0., 1.)
    return torch.where(on_platform, torch.ones_like(approach), approach)*active


def stance_values(q, default_q, tolerance, height):
    error = q-default_q
    # Dead bands allow ordinary gait deviations. No measured state is clamped.
    joint_cost = ((error.abs()/tolerance-1.).clamp(min=0.)).square().mean(-1)
    height_error = height-TARGET_HEIGHT
    height_cost = (height_error/.10).square()
    acceptable = (error.abs() <= tolerance).all(-1) & (height_error.abs() <= HEIGHT_TOLERANCE)
    return joint_cost, height_cost, acceptable, error.square().mean(-1).sqrt()


def measure(command):
    env = command._env
    if getattr(env.cfg, 'pit_posture_profile', 'off') != PROFILE:
        return None
    if getattr(command, '_posture_step', -1) == env.common_step_counter:
        return command.posture_state
    data = command.robot.data
    if not hasattr(command, '_posture_joint_ids'):
        names = command.robot.joint_names
        ids = [i for i,n in enumerate(names) if any(n.endswith('_'+kind+'_joint') for kind in JOINT_TOLERANCE)]
        if len(ids) != 12:
            raise ValueError('Pit stance requires exactly twelve leg joints')
        command._posture_joint_ids = ids
        command._posture_tolerance = torch.tensor(
            [next(v for kind,v in JOINT_TOLERANCE.items() if names[i].endswith('_'+kind+'_joint')) for i in ids],
            device=env.device)
    ids = command._posture_joint_ids
    origins = env.scene.env_origins
    front_x = data.body_pos_w[:,command.wheel_ids,:][:,:,0].amax(-1)+.093
    gap = origins[:,0]+1.5-front_x
    phase = phase_weight(gap, command.stop_on_platform, command.active)
    surface_z = origins[:,2]+torch.where(command.stop_on_platform,
        .15+.05*env.scene.terrain.terrain_levels, 0.)
    height = data.root_pos_w[:,2]-surface_z
    joint_cost, height_cost, acceptable, rms = stance_values(
        data.joint_pos[:,ids], data.default_joint_pos[:,ids], command._posture_tolerance, height)
    command.posture_state = dict(weight=phase, joint_cost=joint_cost, height_cost=height_cost,
        acceptable=acceptable, rms_rad=rms, height_m=height, front_gap_m=gap)
    command._posture_step = env.common_step_counter
    return command.posture_state


def reward(env, kind):
    command = env.command_manager.get_term('base_velocity').pit
    command.refresh()
    state = measure(command)
    return state['weight']*state[kind+'_cost']


def configure(cfg, profile):
    if profile not in ('off', PROFILE):
        raise ValueError('Unknown pit posture profile')
    cfg.pit_posture_profile = profile
    if profile == 'off':
        return
    if getattr(cfg, 'mixed_profile', None) != 'mixed9_v1':
        raise ValueError('Pit posture shaping requires the mixed task')
    from isaaclab.managers import RewardTermCfg
    cfg.rewards.pit_joint_posture = RewardTermCfg(func=reward, weight=-1., params={'kind':'joint'})
    cfg.rewards.pit_stance_height = RewardTermCfg(func=reward, weight=-.5, params={'kind':'height'})


def snapshot(env):
    if getattr(env.cfg, 'pit_posture_profile', 'off') == 'off':
        return None
    c = env.command_manager.get_term('base_velocity').pit
    c.refresh()
    state = measure(c)
    result = dict(profile=PROFILE, target_height_m=TARGET_HEIGHT,
                  height_tolerance_m=HEIGHT_TOLERANCE, joint_tolerance_rad=JOINT_TOLERANCE,
                  joint_weight=-1., height_weight=env.reward_manager.get_term_cfg('pit_stance_height').weight,
                  settle_budget_s=6.)
    for name,mask in [('approach',c.active & ~c.stop_on_platform & (state['weight']>.5)),
                      ('platform',c.active & c.stop_on_platform)]:
        result[name] = dict(count=int(mask.sum()))
        if bool(mask.any()):
            result[name].update({k:float(state[k][mask].float().mean())
                for k in ('acceptable','rms_rad','height_m','joint_cost','height_cost')})
    return result
