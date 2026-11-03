"""Independent source/iteration evidence check; authority is VERIFIED_READY.json."""
import hashlib
import json
from pathlib import Path
import re

SOURCE_SHA = 'f40891182ce1b0f9d80adac2c962b74c6ad8226e183fc44eb62cf9cba4ddd32a'
SOURCE_PATH = '/root/autodl-tmp/cts-omni-repair-v19-20260922/training_31831_to_32831/model_32830.pt'


def validate(final, run, log, saved):
    if (final.get('state') != 'complete' or final.get('steps') != 3000
            or final.get('num_envs') != 4096 or final.get('optimizer_updates') != 0
            or final.get('checkpoint_sha256') != SOURCE_SHA):
        raise ValueError('Invalid zero-update completion or checkpoint hash')
    if (run.get('resume') != SOURCE_PATH or run.get('resume_sha256') != SOURCE_SHA
            or run.get('starting_completed_updates') != 32831 or run.get('additional_updates') != 0):
        raise ValueError('Wrong source or preflight budget')
    if re.findall(r'^RESUME_OK (\d+)\s*$', log, re.M) != ['32831']:
        raise ValueError('Actual restored runner iteration is not 32831')
    if saved.get('completed_updates') != 32831 or saved.get('common_step_counter') != 787944:
        raise ValueError('Source update/physics-step metadata mismatch')
    if not all(run.get('resume_checks',{}).get(k) is True for k in ('model','ppo_optimizer','student_optimizer')):
        raise ValueError('Source model/optimizer restore not verified')
    assert final['phase_coverage']['flat'] == [1., 1.]
    assert final['settings']['phase_pose']['profile'] == 'phase_pose_v18'
    s = final['settings']
    assert s['omni']['profile'] == 'omni_full_episode_v19'
    assert s['allocation']['bounds_profile'] == 'omni_repair_v19'
    assert s['allocation']['target_weights']['flat'] == .40
    assert s['omni']['boundary']['crossings'] == 0
    assert s['lift']['landing_events_since_env_init'] > 0
    assert s['lift']['weights'] == {'lift_event': .5, 'landing_event': 1.}
    assert all(min(row) > 0 for row in s['omni']['kind_counts_by_learner'])
    if s['soft_limit_weight'] != -2. or not -3.025000001 <= s['height_weight'] <= -2.931399999:
        raise ValueError('Unexpected effective restored reward/course age')
    applied_step = s['height_course_applied_step']
    current_step = s['height_course_current_step']
    if current_step != 787944 + 3000 or not 0 <= current_step-applied_step <= 1250:
        raise ValueError('Unexpected actual height-course clock or reset lag')
    expected_weight = -1. - 9. * min(max((applied_step//24-31831)/5000., 0.), 1.)
    if abs(s['height_weight']-expected_weight) > 1e-12:
        raise ValueError('Height weight disagrees with its recorded application step')
    if min(final[k] for k in ('full_zero_flat_env_steps','rotation_flat_env_steps',
            'within_episode_stop_transitions','within_episode_restart_transitions')) <= 0:
        raise ValueError('Missing command transition evidence')


def main():
    import torch
    root = Path(__file__).resolve().parent
    source = Path(SOURCE_PATH)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Preserved source hash changed')
    prepared = json.loads((root/'PREPARED.json').read_text())
    for name, expected in prepared['files'].items():
        if hashlib.sha256((root/'code'/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Prepared runtime changed: {name}')
    final = json.loads((root/'preflight_3000/FINALIZED.json').read_text())
    run = json.loads((root/'preflight_3000/run.json').read_text())
    log = (root/'preflight.log').read_text()
    saved = torch.load(source, map_location='cpu', weights_only=False)
    validate(final,run,log,saved)
    with (root/'VERIFIED_READY.json').open('x') as f:
        json.dump(dict(state='verified_not_training', source_completed=32831,
                       source_common_step_counter=787944, source_sha256=SOURCE_SHA,
                       actual_runner_resume_verified=True, preflight=final),f,indent=2)
    print('VERIFIED_READY source32831, zero optimizer updates; no training started')


if __name__ == '__main__':
    main()
