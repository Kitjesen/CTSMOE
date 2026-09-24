"""Independent source/iteration evidence check; authority is VERIFIED_READY.json."""
import hashlib
import json
from pathlib import Path
import re

SOURCE_SHA = '05c34f02dcfb0bec462640c361258a896a3435e0684f3c3b8ae15a6175b0ef83'
SOURCE_PATH = '/root/autodl-tmp/cts-omni-repair-v19-20260922/recovery01/training_32831_to_34823/model_33173.pt'


def validate(final, run, log, saved):
    if (final.get('state') != 'complete' or final.get('steps') != 3000
            or final.get('num_envs') != 4096 or final.get('optimizer_updates') != 0
            or final.get('checkpoint_sha256') != SOURCE_SHA):
        raise ValueError('Invalid zero-update completion or checkpoint hash')
    if (run.get('resume') != SOURCE_PATH or run.get('resume_sha256') != SOURCE_SHA
            or run.get('starting_completed_updates') != 33174 or run.get('additional_updates') != 0):
        raise ValueError('Wrong source or preflight budget')
    if re.findall(r'^RESUME_OK (\d+)\s*$', log, re.M) != ['33174']:
        raise ValueError('Actual restored runner iteration is not 33174')
    if saved.get('completed_updates') != 33174 or saved.get('common_step_counter') != 796176:
        raise ValueError('Source update/physics-step metadata mismatch')
    if not all(run.get('resume_checks',{}).get(k) is True for k in ('model','ppo_optimizer','student_optimizer')):
        raise ValueError('Source model/optimizer restore not verified')
    assert final['phase_coverage']['flat'] == [1., 1.]
    assert final['settings']['phase_pose']['profile'] == 'phase_pose_v18'
    s = final['settings']
    assert s['omni']['profile'] == 'omni_full_episode_v20'
    assert s['allocation']['bounds_profile'] == 'omni_repair_v19'
    assert s['allocation']['target_weights']['flat'] == .40
    assert s['omni']['boundary']['crossings'] == 0
    assert final.get('source_completed') == 33174
    observed = final['observed_flat_command_abs_max']
    assert all(lo <= actual <= hi for lo, actual, hi in zip([1.9, .95, 1.4], observed, [2.00001, 1.00001, 1.50001]))
    assert s['lift']['landing_events_since_env_init'] > 0
    assert s['lift']['weights'] == {'lift_event': .5, 'landing_event': 1.}
    assert all(min(row) > 0 for row in s['omni']['kind_counts_by_learner'])
    if s['soft_limit_weight'] != -2. or not -10. <= s['height_weight'] <= -1.:
        raise ValueError('Unexpected effective restored reward/course age')
    applied_step = s['height_course_applied_step']
    current_step = s['height_course_current_step']
    if current_step != 796176 + 3000 or not 0 <= current_step-applied_step <= 1250:
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
        json.dump(dict(state='verified_not_training', source_completed=33174,
                       source_common_step_counter=796176, source_sha256=SOURCE_SHA,
                       actual_runner_resume_verified=True, preflight=final),f,indent=2)
    print('VERIFIED_READY source33174, zero optimizer updates; no training started')


if __name__ == '__main__':
    main()
