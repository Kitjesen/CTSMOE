"""V17: restore height/soft-limit costs without forcing a flat pose mid-climb."""
import torch

PROFILE = 'posture_stop_v17'
SOURCE_COMPLETED = 31831
HEIGHT_COURSE_UPDATES = 5000


def height_weight(completed):
    progress = min(max((completed - SOURCE_COMPLETED) / HEIGHT_COURSE_UPDATES, 0.), 1.)
    return -1. - 9. * progress


def height_curriculum(env, env_ids):
    completed = env.common_step_counter // 24
    term = env.reward_manager.get_term_cfg('base_height_l2')
    term.weight = height_weight(completed)
    env.reward_manager.set_term_cfg('base_height_l2', term)
    return term.weight


def height_cost(env, target_height, asset_cfg, sensor_cfg=None):
    from robot_lab.tasks.go2.mdp.rewards import base_height_l2
    from phase_transition import measure
    phase = measure(env)
    original = base_height_l2(env, target_height, asset_cfg, sensor_cfg)
    relative = (env.scene['robot'].data.root_pos_w[:, 2] - phase['ground'] - target_height).square()
    # Stair/platform support rays determine local height; flat and other terrain
    # retain the official reference. Weight is geometric, never based on q error.
    result = torch.where(phase['relevant'], relative * phase['weight'], original)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Nonfinite V17 support-height reward')
    return result


def configure(cfg):
    from isaaclab.managers import CurriculumTermCfg
    if cfg.early_restart_profile != 'early_phase_v14_no_reward':
        raise ValueError('V17 starts from the audited V15 baseline; do not stack old repairs')
    cfg.posture_stop_profile = PROFILE
    cfg.commands.base_velocity.limit_ang_vel_at_zero_command_prob = 0.0
    cfg.rewards.base_height_l2.func = height_cost
    cfg.rewards.base_height_l2.weight = -1.0
    cfg.curriculum.base_height_l2 = CurriculumTermCfg(func=height_curriculum)
    cfg.rewards.joint_pos_limits.weight = -2.0


def snapshot(env):
    from phase_transition import measure
    phase = measure(env)
    command = env.command_manager.get_term('base_velocity')
    flat = command.terrain_idxs == command.terrain_type2idx['flat']
    expected = height_weight(env.common_step_counter // 24)
    actual = env.reward_manager.get_term_cfg('base_height_l2').weight
    # Curriculum updates on episode boundaries, so one rollout's lag is allowed.
    if abs(actual - expected) > 9. / HEIGHT_COURSE_UPDATES * 2:
        raise RuntimeError(f'Height curriculum mismatch: {actual} vs {expected}')
    soft = env.reward_manager.get_term_cfg('joint_pos_limits').weight
    if soft != -2.:
        raise RuntimeError('Soft joint limit reward was overwritten')
    return dict(profile=PROFILE, height_weight=actual, soft_limit_weight=soft,
                height_course_source_completed=SOURCE_COMPLETED,
                height_course_updates=HEIGHT_COURSE_UPDATES,
                full_zero_flat_envs=int((flat & (command.command == 0).all(-1)).sum()),
                rotation_only_flat_envs=int((flat & (command.command[:, :2] == 0).all(-1)
                                             & (command.command[:, 2] != 0)).sum()),
                relevant_phase_weight_min=float(phase['weight'][phase['relevant']].min()),
                relevant_phase_weight_max=float(phase['weight'][phase['relevant']].max()),
                height_target_m=0.46,
                extra_joint_posture_weights_unchanged=True)
