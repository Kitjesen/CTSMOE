"""V8 reward transitions: geometric release, supported landing, timed recovery.

Privileged geometry shapes training rewards only; it is not added to student
observations and does not directly change actions or motor limits.
"""
import torch

PROFILE = 'omni_start_v8'
RECOVERY_S = .8
SUPPORT_DWELL_S = .12


def approach_gate(gap):
    return ((gap-.5)/.5).clamp(0., 1.)


class Recovery:
    def __init__(self, count, device):
        self.weight = torch.ones(count, device=device)
        self.hold = torch.zeros(count, device=device)
        self.stable = torch.zeros(count, device=device)

    def update(self, desired, supported_level, reset, dt, quiet):
        self.weight[reset] = desired[reset]
        self.hold[reset] = 0.
        self.stable[reset] = 0.
        self.hold = torch.where(supported_level, self.hold+dt, 0.)
        # Release immediately as terrain demands motion. Re-enable only after
        # support confirmation, with a bounded rise; posture never releases it.
        rising = torch.minimum(desired, self.weight+dt/RECOVERY_S)
        self.weight = torch.where(desired < self.weight, desired,
                                  torch.where(self.hold >= SUPPORT_DWELL_S, rising, self.weight))
        self.stable = torch.where(supported_level & quiet, self.stable+dt, 0.)
        return self.weight


def measure(env):
    if getattr(env, '_phase_step', -1) == env.common_step_counter:
        return env.phase_state
    from isaaclab.utils.warp import raycast_mesh
    command = env.command_manager.get_term('base_velocity')
    pit = command.pit
    pit.refresh()
    robot = env.scene['robot']
    data = robot.data
    stairs = command.stair_mask()
    relevant = stairs | pit.active
    ids = relevant.nonzero().flatten()
    if not hasattr(env, '_phase_recovery'):
        env._phase_recovery = Recovery(env.num_envs, env.device)
    scan = env.scene['height_scanner'].data.ray_hits_w[..., 2]
    finite = torch.isfinite(scan).all(-1)
    relief = scan.amax(-1)-scan.amin(-1)
    level = finite & (relief <= .05)
    ground = data.root_pos_w[:, 2]-.46
    supported = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    foot_level = torch.zeros_like(supported)
    wheel_ground = torch.zeros(env.num_envs, 4, device=env.device)
    if len(ids):
        positions = data.body_pos_w[ids][:, pit.wheel_ids].clone()
        starts = positions.clone()
        starts[..., 2] += 10.
        direction = torch.zeros_like(starts)
        direction[..., 2] = -1.
        mesh = env.scene['height_scanner'].meshes[env.scene['height_scanner'].cfg.mesh_prim_paths[0]]
        hits = raycast_mesh(starts, direction, mesh)[0][..., 2]
        valid = torch.isfinite(hits).all(-1)
        wheel_ground[ids] = hits
        force = torch.stack([s.data.force_matrix_w[:, 0, 0, 2] for s in pit.ground_contacts], -1)[ids]
        near = (positions[..., 2]-hits-.093).abs() < .04
        contact = (force > 5.) & near
        foot_level[ids] = valid & ((hits.amax(-1)-hits.amin(-1)) < .05)
        supported[ids] = foot_level[ids] & (contact.sum(-1) >= 3)
        ground[ids] = torch.where(valid, hits.mean(-1), ground[ids])
    gap = env.scene.env_origins[:, 0]+1.5-(data.body_pos_w[:, pit.wheel_ids, 0].amax(-1)+.093)
    desired = torch.ones(env.num_envs, device=env.device)
    desired[stairs] = (level & foot_level)[stairs].float()
    on_top = pit.stop_on_platform & foot_level
    desired[pit.active] = torch.where(on_top, torch.ones_like(gap), approach_gate(gap))[pit.active]
    ready = supported & level
    ready = torch.where(pit.active, supported & ((gap >= 1.) | on_top), ready)
    quiet = (data.root_lin_vel_w[:, 2].abs() < .25) & (data.root_ang_vel_w[:, :2].norm(dim=-1) < .8)
    quiet &= -data.projected_gravity_b[:, 2] > .94
    reset = env.episode_length_buf <= 1
    weight = env._phase_recovery.update(desired, ready, reset, env.step_dt, quiet)
    # Unrelated terrain keeps its V6 behavior; flat never uses this gate.
    weight = torch.where(relevant, weight, torch.ones_like(weight))
    ramping = relevant & ready & (weight < .999)
    state = dict(weight=weight, relevant=relevant, stairs=stairs, pit=pit.active,
                 ground=ground, ready=ready, gap=gap, ramping=ramping,
                 wheel_ground=wheel_ground, stable_s=env._phase_recovery.stable,
                 l1_scale=torch.where(relevant, .2+.8*weight, torch.ones_like(weight)))
    env.phase_state, env._phase_step = state, env.common_step_counter
    return state


def joint_reward(env, kind):
    from pit_posture import measure as pit_measure
    from surface_posture import measure as surface_measure
    state = measure(env)
    surface = surface_measure(env)
    pit = env.command_manager.get_term('base_velocity').pit
    ps = pit_measure(pit)
    if kind.startswith('pit_'):
        cost = ps['joint_cost'] if kind == 'pit_joint' else ((env.scene['robot'].data.root_pos_w[:, 2]-state['ground']-.46)/.1).square()
        return cost*state['weight']*pit.active
    if kind == 'surface_joint':
        return surface['joint_cost']*torch.where(state['stairs'], state['weight'], surface['weight'])
    height_cost = ((env.scene['robot'].data.root_pos_w[:, 2]-state['ground']-.46)/.1).square()
    return torch.where(state['stairs'], height_cost*state['weight'], surface['height_cost']*surface['height_weight'])


def hip_l1(env, command_name, asset_cfg, stand_still_scale, command_threshold):
    from robot_lab.tasks.go2.mdp.rewards import hip_pos_penalty_l1
    return measure(env)['l1_scale']*hip_pos_penalty_l1(env, command_name, asset_cfg, stand_still_scale, command_threshold)


def leg_l1(env, command_name, asset_cfg, stand_still_scale, velocity_threshold, command_threshold):
    from robot_lab.tasks.go2.mdp.rewards import joint_pos_penalty_l1
    return measure(env)['l1_scale']*joint_pos_penalty_l1(env, command_name, asset_cfg, stand_still_scale, velocity_threshold, command_threshold)


def recovery_motion(env):
    state = measure(env)
    data = env.scene['robot'].data
    # Active as soon as landing support is established, not delayed until the
    # pose ramp completes. No orientation target or vertical penalty in climb.
    cost = ((data.root_lin_vel_w[:, 2].abs()-.25).clamp_min(0.)/.5).square()
    cost += .2*((data.root_ang_vel_w[:, :2].norm(dim=-1)-.8).clamp_min(0.)).square()
    return cost*(state['ready'] & state['relevant'])


def configure(cfg):
    from isaaclab.managers import RewardTermCfg
    cfg.rewards.pit_joint_posture.func = joint_reward
    cfg.rewards.pit_joint_posture.params = {'kind': 'pit_joint'}
    cfg.rewards.pit_stance_height.func = joint_reward
    cfg.rewards.pit_stance_height.params = {'kind': 'pit_height'}
    cfg.rewards.surface_joint_posture.func = joint_reward
    cfg.rewards.surface_joint_posture.params = {'kind': 'surface_joint'}
    cfg.rewards.surface_stance_height.func = joint_reward
    cfg.rewards.surface_stance_height.params = {'kind': 'surface_height'}
    cfg.rewards.hip_pos_penalty_l1.func = hip_l1
    cfg.rewards.joint_pos_penalty_l1.func = leg_l1
    cfg.rewards.landing_motion = RewardTermCfg(func=recovery_motion, weight=-.5)
    for name in ('stairs_up', 'stairs_down'):
        cfg.scene.terrain.terrain_generator.sub_terrains[name].phase_landing_kind = name


def snapshot(env):
    state = measure(env)
    return dict(profile=PROFILE, recovery_s=RECOVERY_S, support_dwell_s=SUPPORT_DWELL_S,
                approach_release_m=[1., .5], climbing_l1_scale=.2,
                active_climb=int((state['relevant'] & (state['weight'] < .01)).sum()),
                recovering=int(state['ramping'].sum()),
                supported_level=int((state['relevant'] & state['ready']).sum()),
                mean_weight_by_terrain={name: float(state['weight'][mask].mean()) if bool(mask.any()) else None
                    for name, mask in [('pit', state['pit']), ('stairs', state['stairs'])]})
