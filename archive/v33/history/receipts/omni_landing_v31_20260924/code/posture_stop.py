"""V17: restore height/soft-limit costs without forcing a flat pose mid-climb."""
import math

import torch

PROFILE = 'posture_stop_v17'
SOURCE_COMPLETED = 31831
HEIGHT_COURSE_UPDATES = 5000
APPLIED_STEP_ATTR = '_posture_stop_height_applied_step'


def height_weight(completed):
    progress = min(max((completed - SOURCE_COMPLETED) / HEIGHT_COURSE_UPDATES, 0.), 1.)
    return -1. - 9. * progress


def validate_height_curriculum_state(actual_weight, applied_step, current_step, max_lag_steps):
    """Validate the global weight against when the reset-driven course applied it."""
    if applied_step is None:
        raise RuntimeError('Height curriculum has no recorded application step')
    applied_step = int(applied_step)
    current_step = int(current_step)
    max_lag_steps = int(max_lag_steps)
    if applied_step > current_step:
        raise RuntimeError(
            f'Height curriculum application step is in the future: {applied_step} vs {current_step}'
        )
    if max_lag_steps < 0:
        raise ValueError(f'Negative height curriculum lag budget: {max_lag_steps}')
    expected_at_apply = height_weight(applied_step // 24)
    actual_weight = float(actual_weight)
    if not math.isclose(actual_weight, expected_at_apply, rel_tol=0., abs_tol=1e-12):
        raise RuntimeError(
            'Height curriculum weight mismatch at applied step: '
            f'{actual_weight} vs {expected_at_apply} (step {applied_step})'
        )
    lag_steps = current_step - applied_step
    if lag_steps > max_lag_steps:
        raise RuntimeError(
            'Height curriculum is stale: '
            f'{lag_steps} control steps since application, maximum {max_lag_steps}'
        )
    return dict(
        applied_step=applied_step,
        current_step=current_step,
        lag_steps=lag_steps,
        lag_updates=lag_steps / 24.,
        max_lag_steps=max_lag_steps,
        expected_at_apply=expected_at_apply,
    )


def height_curriculum(env, env_ids):
    applied_step = int(env.common_step_counter)
    completed = applied_step // 24
    term = env.reward_manager.get_term_cfg('base_height_l2')
    term.weight = height_weight(completed)
    env.reward_manager.set_term_cfg('base_height_l2', term)
    setattr(env, APPLIED_STEP_ATTR, applied_step)
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
    actual = env.reward_manager.get_term_cfg('base_height_l2').weight
    course_state = validate_height_curriculum_state(
        actual,
        getattr(env, APPLIED_STEP_ATTR, None),
        env.common_step_counter,
        env.max_episode_length,
    )
    soft = env.reward_manager.get_term_cfg('joint_pos_limits').weight
    if soft != -2.:
        raise RuntimeError('Soft joint limit reward was overwritten')
    return dict(profile=PROFILE, height_weight=actual, soft_limit_weight=soft,
                height_course_source_completed=SOURCE_COMPLETED,
                height_course_updates=HEIGHT_COURSE_UPDATES,
                height_course_applied_step=course_state['applied_step'],
                height_course_current_step=course_state['current_step'],
                height_course_lag_steps=course_state['lag_steps'],
                height_course_lag_updates=course_state['lag_updates'],
                height_course_max_lag_steps=course_state['max_lag_steps'],
                full_zero_flat_envs=int((flat & (command.command == 0).all(-1)).sum()),
                rotation_only_flat_envs=int((flat & (command.command[:, :2] == 0).all(-1)
                                             & (command.command[:, 2] != 0)).sum()),
                relevant_phase_weight_min=float(phase['weight'][phase['relevant']].min()),
                relevant_phase_weight_max=float(phase['weight'][phase['relevant']].max()),
                height_target_m=0.46,
                extra_joint_posture_weights_unchanged=True)
