"""Check wave reset profile inheritance, both optimizers, save, and policy export."""
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root/'recording/wave_reset_20260919'
code = root/'code_wave_reset_20260919'
source = root/'recording/wave_diagnosis_20260919/resume64/model_95.pt'
jobs = [('smoke64', source, 4, ['--wave-reset-profile', 'wave_surface_v1']),
        ('resume64', folder/'smoke64/model_99.pt', 2, [])]
checks = {}
for name, checkpoint, updates, extra in jobs:
    args = [sys.executable, str(code/'train.py'), '--root', str(root), '--output', str(folder/name),
            '--num-envs', '64', '--updates', str(updates), '--resume', str(checkpoint)] + extra
    with (folder/f'{name}.log').open('x') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    run = json.loads((folder/name/'run.json').read_text())
    assert all(run['resume_checks'].values())
    assert run['wave_reset_profile'] == 'wave_surface_v1'
    assert run['wave_reset']['count'] > 0
    assert run['actuator_profile'] == 'rs0402_slew12_v3'
    assert run['leg_target_max_delta_rad'] == .24
    complete = run['target_completed_updates']
    saved = torch.load(folder/name/f'model_{complete-1}.pt', map_location='cpu', weights_only=False)
    assert saved['wave_reset_profile'] == 'wave_surface_v1'
    extra_files = {'training_context.json': ''}
    torch.jit.load(str(folder/name/f'student_{complete}.pt'), map_location='cpu', _extra_files=extra_files)
    assert json.loads(extra_files['training_context.json']) == {'wave_reset_profile': 'wave_surface_v1'}
    rows = [json.loads(line) for line in (folder/name/'metrics.jsonl').read_text().splitlines()]
    assert all(row['wave_reset_profile'] == 'wave_surface_v1' for row in rows)
    assert all(math.isfinite(row[key]) for row in rows for key in ('value', 'surrogate', 'entropy', 'mean_latent_loss'))
    checks[name] = dict(completed_updates=complete, resume_checks=run['resume_checks'], reset=run['wave_reset'])
(folder/'training_validation.json').write_text(json.dumps(dict(passed=True, checks=checks), indent=2))
print('TRAINING_VALIDATION_OK', flush=True)
