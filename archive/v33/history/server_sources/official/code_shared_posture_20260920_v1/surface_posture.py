"""Normal stance on locally level ground; climbing keeps its own objective."""
import torch

from pit_posture import JOINT_TOLERANCE, stance_values

PROFILE = 'level_surface_v1'


def surface_weight(ray_heights, commands, pit_active):
    """Use raw terrain geometry, not body posture or noisy policy observations.

    The existing 1.6 x 1.0 m scanner covers the nominal wheel footprint and
    sees an approaching edge before the front wheels reach it. Full shaping
    requires <=4 cm relief, fading to zero at 10 cm. This is deliberately a
    level-ground objective, not a desired posture on a slope or across steps.
    High-speed/turning gaits retain their existing rewards: the new objective
    fades over 0.8..1.5 m/s translation and 0.5..1 rad/s commanded yaw.
    """
    finite = torch.isfinite(ray_heights).all(-1)
    # Missing ray hits cannot define a support plane. Suppress only this new
    # shaping term; termination, contact and actuator constraints still apply.
    safe = torch.where(finite[:, None], ray_heights, torch.zeros_like(ray_heights))
    relief = safe.amax(-1)-safe.amin(-1)
    level = ((.10-relief)/.06).clamp(0., 1.)
    speed = ((1.5-commands[:, :2].norm(dim=-1))/.7).clamp(0., 1.)
    turning = ((1.-commands[:, 2].abs())/.5).clamp(0., 1.)
    return level*speed*turning*finite*(~pit_active), safe.mean(-1), relief


def measure(env):
    command = env.command_manager.get_term('base_velocity')
    robot = env.scene['robot']
    if not hasattr(env, '_surface_posture_ids'):
        ids = [i for i,n in enumerate(robot.joint_names)
               if any(n.endswith('_'+kind+'_joint') for kind in JOINT_TOLERANCE)]
        if len(ids) != 12:
            raise ValueError('Surface stance requires twelve leg joints')
        env._surface_posture_ids = ids
        env._surface_posture_tolerance = torch.tensor(
            [next(v for kind,v in JOINT_TOLERANCE.items()
                  if robot.joint_names[i].endswith('_'+kind+'_joint')) for i in ids], device=env.device)
    ids = env._surface_posture_ids
    weight, ground, relief = surface_weight(
        env.scene['height_scanner'].data.ray_hits_w[..., 2], command.command, command.pit.active)
    height = robot.data.root_pos_w[:, 2]-ground
    joint, height_cost, acceptable, rms = stance_values(
        robot.data.joint_pos[:, ids], robot.data.default_joint_pos[:, ids],
        env._surface_posture_tolerance, height)
    return dict(weight=weight, joint_cost=joint, height_cost=height_cost,
                height_m=height, rms_rad=rms, acceptable=acceptable, relief_m=relief)


def reward(env, kind):
    state = measure(env)
    return state['weight']*state[kind+'_cost']


def configure(cfg, profile):
    if profile not in ('off', PROFILE):
        raise ValueError('Unknown surface posture profile')
    cfg.surface_posture_profile = profile
    if profile == 'off':
        return
    if getattr(cfg, 'mixed_profile', None) != 'mixed9_v1':
        raise ValueError('Surface stance requires mixed terrain membership')
    from isaaclab.managers import RewardTermCfg
    cfg.rewards.surface_joint_posture = RewardTermCfg(func=reward, weight=-1., params={'kind': 'joint'})
    cfg.rewards.surface_stance_height = RewardTermCfg(func=reward, weight=-.5, params={'kind': 'height'})


def snapshot(env):
    if getattr(env.cfg, 'surface_posture_profile', 'off') == 'off':
        return None
    state = measure(env)
    command = env.command_manager.get_term('base_velocity')
    result = dict(profile=PROFILE, full_relief_m=.04, zero_relief_m=.10,
                  full_speed_mps=.8, zero_speed_mps=1.5, joint_weight=-1., height_weight=-.5,
                  target_height_m=.46, joint_tolerance_rad=JOINT_TOLERANCE)
    for name,index in command.terrain_type2idx.items():
        mask = (command.terrain_idxs == index) & (state['weight'] > .5)
        result[name] = dict(count=int(mask.sum()))
        if bool(mask.any()):
            result[name].update({k: float(state[k][mask].float().mean())
                                 for k in ('acceptable', 'rms_rad', 'height_m', 'joint_cost', 'height_cost')})
    return result
