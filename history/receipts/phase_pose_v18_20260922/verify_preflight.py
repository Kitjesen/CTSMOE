"""Independent source/iteration evidence check; authority is VERIFIED_READY.json."""
import hashlib
import json
from pathlib import Path
import re

SOURCE_SHA = '04fcb792c8f18d3abe913d57eecea1d38b54a4e8a36a9899324bd39540b9876d'
SOURCE_PATH = '/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/mixed9_4096_60000_relax45_20260919/model_31830.pt'


def validate(final, run, log, saved):
    if (final.get('state') != 'complete' or final.get('steps') != 3000
            or final.get('num_envs') != 4096 or final.get('optimizer_updates') != 0
            or final.get('checkpoint_sha256') != SOURCE_SHA):
        raise ValueError('Invalid zero-update completion or checkpoint hash')
    if (run.get('resume') != SOURCE_PATH or run.get('resume_sha256') != SOURCE_SHA
            or run.get('starting_completed_updates') != 31831 or run.get('additional_updates') != 0):
        raise ValueError('Wrong source or preflight budget')
    if re.findall(r'^RESUME_OK (\d+)\s*$', log, re.M) != ['31831']:
        raise ValueError('Actual restored runner iteration is not 31831')
    if saved.get('completed_updates') != 31831 or saved.get('common_step_counter') != 763944:
        raise ValueError('Source update/physics-step metadata mismatch')
    if not all(run.get('resume_checks',{}).get(k) is True for k in ('model','ppo_optimizer','student_optimizer')):
        raise ValueError('Source model/optimizer restore not verified')
    assert final['phase_coverage']['flat'] == [1., 1.]
    assert final['settings']['phase_pose']['profile'] == 'phase_pose_v18'
    s = final['settings']
    if s['soft_limit_weight'] != -2. or not -1.229 <= s['height_weight'] <= -1.221:
        raise ValueError('Unexpected effective restored reward/course age')
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
        json.dump(dict(state='verified_not_training', source_completed=31831,
                       source_common_step_counter=763944, source_sha256=SOURCE_SHA,
                       actual_runner_resume_verified=True, preflight=final),f,indent=2)
    print('VERIFIED_READY source31831, zero optimizer updates; no training started')


if __name__ == '__main__':
    main()
