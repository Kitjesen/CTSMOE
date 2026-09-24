"""Independent V20 source/restore and V21 zero-update/geometry qualification."""
import json
from pathlib import Path
import re
import sys

from runtime_contract import (
    COMMON_STEP, COMPLETED, IDENTITY_NAME, NUM_ENVS, STEPS, load_identity,
    validate_coverage, verify_prepared,
)

HERE = Path(__file__).resolve().parent
IDENTITY = load_identity(HERE / IDENTITY_NAME)
SOURCE_SHA = IDENTITY['sha256']
SOURCE_PATH = IDENTITY['path']
RESTORE_CHECKS = (
    'model', 'ppo_optimizer', 'student_optimizer', 'terrain_assignment',
    'command_terrain_membership', 'terrain_restored_exactly', 'terrain_allocation',
    'stair_episode_totals',
)


def validate(final, run, log, saved):
    if (final.get('state') != 'complete' or final.get('steps') != STEPS
            or final.get('num_envs') != NUM_ENVS or final.get('optimizer_updates') != 0
            or final.get('checkpoint_sha256') != SOURCE_SHA or final.get('source_completed') != COMPLETED):
        raise ValueError('Invalid zero-update completion or checkpoint identity')
    if (run.get('resume') != SOURCE_PATH or run.get('resume_sha256') != SOURCE_SHA
            or run.get('starting_completed_updates') != COMPLETED or run.get('additional_updates') != 0
            or run.get('target_completed_updates') != COMPLETED):
        raise ValueError('Wrong source or preflight budget')
    if re.findall(r'^RESUME_OK (\d+)\s*$', log, re.M) != [str(COMPLETED)]:
        raise ValueError('Actual restored runner iteration is not source34823')
    if saved.get('completed_updates') != COMPLETED or saved.get('common_step_counter') != COMMON_STEP:
        raise ValueError('Source update/physics-step metadata mismatch')
    if not all(run.get('resume_checks', {}).get(k) is True for k in RESTORE_CHECKS):
        raise ValueError('Model, dual optimizers, or exact terrain/course restoration not verified')
    if (final.get('common_step_before') != COMMON_STEP or final.get('common_step_after') != COMMON_STEP + STEPS
            or final.get('runner_iteration_before') != COMPLETED
            or final.get('runner_iteration_after') != COMPLETED
            or final.get('mutation_guard_restored') is not True
            or final.get('forbidden_calls') != [] or final.get('checkpoint_artifacts_written') != []):
        raise ValueError('Preflight mutation guard, age, or no-save evidence missing')
    hashes = final.get('state_hashes_before', {})
    if (set(hashes) != {'model', 'ppo_optimizer', 'student_optimizer'}
            or hashes != final.get('state_hashes_after')
            or not all(re.fullmatch('[0-9a-f]{64}', h) for h in hashes.values())):
        raise ValueError('Model/optimizer fingerprints changed or missing')
    sys.path.insert(0, str(HERE / "code"))
    from hip_contract import contract
    if final["hip_contract"] != contract() or run["hip_contract"] != contract():
        raise ValueError("Wrong default or hip operating envelope")
    if max(final["hip_evidence"]["peak_inward_rad"]["target"]) > .120001:
        raise ValueError("Hip target guard failed")
    initial = run['stair_transition_coverage']
    if initial != final['initial_coverage']:
        raise ValueError('Restored coverage snapshot mismatch')
    validate_coverage(initial)
    observation = final['coverage_observation']
    if observation['cache_missing_control_steps'] != 0:
        raise ValueError('Existing phase/pose caches were unavailable for exposure observation')
    for label, values in observation['groups'].items():
        learner, kind = label.split('/')
        if values['initial_env_count'] != initial['assignments']['by_learner'][learner]['by_type'][kind]['coverage']:
            raise ValueError('Exposure observation did not retain the actual initial coverage cohort')
    if set(observation['groups']) != {'teacher/stairs_up', 'teacher/stairs_down', 'student/stairs_up', 'student/stairs_down'}:
        raise ValueError('Incomplete exposure observation groups')
    for learner, values in initial['assignments']['by_learner'].items():
        for kind, counts in values['by_type'].items():
            if counts['total'] != run['terrain_by_learner'][learner][kind]['count']:
                raise ValueError('Coverage counts disagree with exact restored terrain summaries')
    s = final['settings']
    validate_coverage(s['stair_transition_coverage'])
    if s['stair_transition_coverage']['mesh_receipts'] != initial['mesh_receipts']:
        raise ValueError('Runtime geometry differs from restored snapshot')
    expected_profiles = dict(
        early_restart_profile='early_phase_v14_no_reward', startup_repair_profile='off',
        pit_posture_profile='off', surface_posture_profile='off', stance_strength_profile='off',
        actuator_profile='rs0402_transient_v4', allocation_profile='balanced_adaptive_v1',
        slope_profile='slope45_v1', command_curriculum_start='intermediate',
        base_height_penalty='original', terrain_training_profile='stairs80_v1',
        stair_command_profile='straight_speed_v1', wave_reset_profile='wave_surface_v1',
        pit_training_profile='strict_v1', pit_course_profile='global_v1', mixed_profile='mixed9_v1')
    for key, expected in expected_profiles.items():
        if run.get(key) != expected or saved.get(key) != expected:
            raise ValueError(f'Inherited profile changed: {key}')
    if run.get('episode_length_s') != 25. or run.get('initial_episode_lengths_randomized') is not False:
        raise ValueError('Original episode/reset protocol changed')
    if run.get('terrain_migration') is not None or run.get('slope_migration') is not None:
        raise ValueError('Unexpected terrain difficulty or allocation migration')
    if final['phase_coverage']['flat'] != [1., 1.] or s['phase_pose']['profile'] != 'phase_pose_v18':
        raise ValueError('Inherited pose gate changed')
    if (s['omni']['profile'] != 'omni_full_episode_v20'
            or s['allocation']['bounds_profile'] != 'omni_repair_v19'
            or s['allocation']['target_weights']['flat'] != .40 or s['omni']['boundary']['crossings'] != 0):
        raise ValueError('Command or allocation profile changed')
    if (s['allocation']['pending_moves'] != 0
            or s['allocation']['target_weights'] != run['terrain_allocation']['target_weights']
            or any(abs(s['allocation']['actual_counts'][k] - NUM_ENVS * w) > 3
                   for k, w in s['allocation']['target_weights'].items())):
        raise ValueError('Actual allocation differs from preserved target proportions')
    for kind, actual in s['allocation']['actual_counts'].items():
        if sum(group['by_type'][kind]['total'] for group in
               s['stair_transition_coverage']['assignments']['by_learner'].values()) != actual:
            raise ValueError('Final coverage counts disagree with actual allocation')
    observed = final['observed_flat_command_abs_max']
    if len(observed) != 3 or not all(lo <= actual <= hi for lo, actual, hi in zip(
            [1.9, .95, 1.4], observed, [2.00001, 1.00001, 1.50001])):
        raise ValueError('Missing actual V20 full-speed command coverage')
    if (s['lift']['landing_events_since_env_init'] <= 0
            or s['lift']['weights'] != {'lift_event': .5, 'landing_event': 1.}
            or not all(v['tracking_gate_env_steps'] > 0 for k, v in s['lift']['cumulative_by_command'].items()
                       if k != 'other')
            or not all(min(row) > 0 for row in s['omni']['kind_counts_by_learner'])
            or not all(min(row) > 0 for row in s['omni']['stop_restart_by_learner'])):
        raise ValueError('Inherited lift or learner sampling qualification missing')
    if s['soft_limit_weight'] != -2. or not -10. <= s['height_weight'] <= -1.:
        raise ValueError('Unexpected effective restored reward/course age')
    applied_step, current_step = s['height_course_applied_step'], s['height_course_current_step']
    if current_step != COMMON_STEP + STEPS or not 0 <= current_step - applied_step <= 1250:
        raise ValueError('Unexpected actual height-course clock or reset lag')
    expected_weight = -1. - 9. * min(max((applied_step // 24 - 31831) / 5000., 0.), 1.)
    if abs(s['height_weight'] - expected_weight) > 1e-12:
        raise ValueError('Height weight disagrees with its recorded application step')
    expected_rewards = {
        'landing_motion': (False, None), 'lin_vel_z_l2': (True, 0.),
        'normal_vertical_velocity': (False, None), 'early_vertical_velocity': (True, 0.),
        'early_actual_hip_l1': (True, -.2), 'early_actual_thigh_calf_l1': (True, -.2),
        'base_height_l2': (True, s['height_weight']), 'joint_pos_limits': (True, -2.),
    }
    for key, (registered, weight) in expected_rewards.items():
        if s['effective_reward_audit'].get(key) != dict(registered=registered, weight=weight):
            raise ValueError(f'Inherited effective reward changed: {key}')
    if min(final[k] for k in ('full_zero_flat_env_steps', 'rotation_flat_env_steps',
                             'within_episode_stop_transitions', 'within_episode_restart_transitions')) <= 0:
        raise ValueError('Missing command transition evidence')
    if any(not (final['phase_coverage'][name][0] < .1 and final['phase_coverage'][name][1] > .9)
           for name in ('pit', 'stairs')):
        raise ValueError('Inherited phase qualification missing')


def main():
    import torch

    identity, _ = verify_prepared(HERE)
    output = HERE / 'preflight_3000'
    final = json.loads((output / 'FINALIZED.json').read_text())
    run = json.loads((output / 'run.json').read_text())
    geometry = json.loads((output / 'COVERAGE_GEOMETRY.json').read_text())
    if geometry != run['stair_transition_coverage']:
        raise ValueError('Standalone geometry receipt differs from restored run record')
    if list(output.rglob('*.pt')):
        raise ValueError('Preflight unexpectedly wrote checkpoint artifacts')
    saved = torch.load(identity['path'], map_location='cpu', weights_only=False)
    validate(final, run, (HERE / 'preflight.log').read_text(), saved)
    # Tie the start-of-preflight hash directly to the preserved checkpoint contents.
    from preflight_support import state_sha256
    for label, key in (('model', 'model_state_dict'), ('ppo_optimizer', 'optimizer_state_dict'),
                       ('student_optimizer', 'optimizer_stu_enc_state_dict')):
        if state_sha256(saved[key], torch) != final['state_hashes_before'][label]:
            raise ValueError(f'Preflight started from different {label} state')
    verify_prepared(HERE)
    with (HERE / 'VERIFIED_READY.json').open('x') as stream:
        json.dump(dict(state='verified_not_training', source_completed=COMPLETED,
                       source_common_step_counter=COMMON_STEP, source_sha256=SOURCE_SHA,
                       actual_runner_resume_verified=True, optimizer_updates=0,
                       preflight=final), stream, indent=2)
    print('VERIFIED_READY source34823, zero optimizer updates; no training started')


if __name__ == '__main__':
    if sys.flags.optimize:
        raise RuntimeError('Qualification requires assertions enabled')
    main()
