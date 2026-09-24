"""V18: one extra actual-angle L1 objective with explicit normal/climb recovery scope.

Privileged phase geometry is used in rewards and diagnostics only. It neither
changes actor observations nor overrides policy actions or actuator limits.
"""
import torch

PROFILE = 'phase_pose_v18'
EXTRA_L1_WEIGHT = -0.20
RELEASE_FAR_M = .85
RELEASE_NEAR_M = .35
RECOVERY_S = 1.5
SUPPORT_DWELL_S = .12


def approach_weight(reference_gap):
    t = ((reference_gap - RELEASE_NEAR_M) / (RELEASE_FAR_M - RELEASE_NEAR_M)).clamp(0., 1.)
    return t.square() * (3. - 2. * t)


def reference_gap(base_x, origin_x, fixed_front_extent):
    # Use a fixed default-pose hull, not current wheel locations: stretching a
    # leg alone must not release its own posture cost. The task's +X exit is fixed.
    from pit_geometry import EDGE_OFFSET
    return origin_x + EDGE_OFFSET - base_x - fixed_front_extent


class Recovery:
    def __init__(self, count, device):
        self.weight = torch.ones(count, device=device)
        self.hold = torch.zeros(count, device=device)

    def update(self, desired, supported, reset, dt):
        self.weight[reset] = desired[reset]
        self.hold[reset] = 0.
        self.hold = torch.where(supported, self.hold + dt, 0.)
        rising = torch.minimum(desired, self.weight + dt / RECOVERY_S)
        self.weight = torch.where(desired < self.weight, desired,
            torch.where(self.hold >= SUPPORT_DWELL_S, rising, self.weight))
        return self.weight


def measure(env):
    if getattr(env, '_pose_v18_step', -1) == env.common_step_counter:
        return env._pose_v18_state
    from phase_transition import measure as support_measure
    phase = support_measure(env)
    command = env.command_manager.get_term('base_velocity')
    pit = command.pit
    data = env.scene['robot'].data
    if not hasattr(env, '_pose_v18_front_extent'):
        pose = torch.zeros(1, 7, device=env.device)
        pose[:, 3] = 1.
        points = pit.geometry.world_points(data.default_joint_pos[:1], pose)
        extent = float(points[..., 0].max())
        if not .1 < extent < 1.:
            raise ValueError(f'Unexpected default collision hull extent: {extent}')
        env._pose_v18_front_extent = extent
        env._pose_v18_recovery = Recovery(env.num_envs, env.device)
    gap = reference_gap(data.root_pos_w[:, 0], env.scene.env_origins[:, 0],
                        env._pose_v18_front_extent)
    flat = command.terrain_idxs == command.terrain_type2idx['flat']
    desired = torch.zeros(env.num_envs, device=env.device)
    desired[flat] = 1.
    # Existing stair support/terrain logic already suppresses the extra cost on
    # steps and confirms a level platform. V18 adds its own bounded recovery rise.
    desired[phase['stairs']] = phase['weight'][phase['stairs']]
    # The platform stage is independent of whether this task requests a stop.
    # Intermittent loss of support pauses recovery, rather than restarting it.
    platform_stage = pit.stop_on_platform | pit.platform_entered
    pit_desired = torch.where(platform_stage, torch.ones_like(gap), approach_weight(gap))
    desired[pit.active] = pit_desired[pit.active]
    supported = phase['ready'].clone()
    supported[pit.active] = ((pit.supported.all(-1) & pit.platform_clear)
                             | ((gap >= RELEASE_FAR_M) & phase['ready']))[pit.active]
    supported[flat] = True
    weight = env._pose_v18_recovery.update(desired, supported, env.episode_length_buf <= 1, env.step_dt)
    # Flat always gets continuous pose feedback, including pure yaw and high speed.
    weight = torch.where(flat, torch.ones_like(weight), weight)
    ids = [i for i, n in enumerate(env.scene['robot'].joint_names)
           if n.endswith(('_hip_joint', '_thigh_joint', '_calf_joint'))]
    if len(ids) != 12:
        raise ValueError('V18 requires exactly twelve leg joints')
    if not bool(torch.isfinite(torch.cat((weight[:, None], data.joint_pos[:, ids]), dim=1)).all()):
        raise FloatingPointError('Nonfinite V18 pose state')
    state = dict(weight=weight, reference_gap=gap, flat=flat, pit=pit.active,
                 stairs=phase['stairs'], supported=supported,
                 recovering=supported & (weight > 0.) & (weight < .999),
                 fixed_front_extent_m=env._pose_v18_front_extent)
    env._pose_v18_step, env._pose_v18_state = env.common_step_counter, state
    return state


def reward(env, kind):
    names = env.scene['robot'].joint_names
    suffixes = ('_hip_joint',) if kind == 'hip' else ('_thigh_joint', '_calf_joint')
    if kind not in ('hip', 'thigh_calf'):
        raise ValueError(kind)
    ids = [i for i, name in enumerate(names) if name.endswith(suffixes)]
    data = env.scene['robot'].data
    return (data.joint_pos[:, ids] - data.default_joint_pos[:, ids]).abs().sum(-1) * measure(env)['weight']


def configure(cfg):
    if cfg.posture_stop_profile != 'posture_stop_v17':
        raise ValueError('V18 requires the verified V17 baseline')
    # Reuse the two disabled actual-angle terms; do not add request/target proxies.
    for name, kind in [('early_actual_hip_l1', 'hip'),
                       ('early_actual_thigh_calf_l1', 'thigh_calf')]:
        term = getattr(cfg.rewards, name)
        if term.weight != 0.:
            raise ValueError('Do not stack an already-active early posture repair')
        term.func, term.params, term.weight = reward, {'kind': kind}, EXTRA_L1_WEIGHT
    cfg.phase_pose_profile = PROFILE


def snapshot(env):
    state = measure(env)
    names = env.scene['robot'].joint_names
    ids = [i for i, name in enumerate(names) if name.endswith(('_hip_joint', '_thigh_joint', '_calf_joint'))]
    data = env.scene['robot'].data
    error = (data.joint_pos[:, ids] - data.default_joint_pos[:, ids]).abs()
    stats = {}
    for label, mask in [('flat', state['flat']), ('pit', state['pit']), ('stairs', state['stairs'])]:
        if bool(mask.any()):
            stats[label] = dict(count=int(mask.sum()), gate_mean=float(state['weight'][mask].mean()),
                gate_min=float(state['weight'][mask].min()), gate_max=float(state['weight'][mask].max()),
                actual_abs_mean_per_joint=error[mask].mean(0).tolist(),
                actual_abs_max_per_joint=error[mask].amax(0).tolist())
    return dict(profile=PROFILE, extra_l1_weight=EXTRA_L1_WEIGHT, recovery_s=RECOVERY_S,
        release_reference_gap_m=[RELEASE_FAR_M, RELEASE_NEAR_M],
        fixed_front_extent_m=state['fixed_front_extent_m'],
        joint_names=[names[i] for i in ids], by_terrain=stats,
        normal_effective_hip_weight=-.25, normal_effective_thigh_calf_weight=-.21,
        climb_baseline_hip_weight=-.05, climb_baseline_thigh_calf_weight=-.01,
        diagnostic_scope='current control step across training environments, not held-out performance')
