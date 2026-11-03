"""Smoke train, resume, and replay with the complete v3 contract."""

import json
import math
from pathlib import Path
import subprocess
import sys

import torch


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root / 'recording/wave_diagnosis_20260919'
code = root / 'code_slew12_20260919'
sys.path.insert(0, str(code))
from policy_contract import checkpoint_profile, export_contract, load_student_export  # noqa: E402

profile = 'rs0402_slew12_v3'
source = root / 'recording/actuator_peak_20260918/resume64/model_89.pt'
jobs = [
    ('smoke64', [str(code / 'train.py'), '--root', str(root), '--output', str(folder / 'smoke64'),
                 '--num-envs', '64', '--updates', '4', '--resume', str(source), '--actuator-profile', profile]),
    ('resume64', [str(code / 'train.py'), '--root', str(root), '--output', str(folder / 'resume64'),
                  '--num-envs', '64', '--updates', '2', '--resume', str(folder / 'smoke64/model_93.pt')]),
    ('playback', [str(code / 'record_student.py'), '--root', str(root), '--output', str(folder / 'playback'),
                  '--policy', str(folder / 'resume64/student_96.pt'), '--smoke', '--trace-only']),
]
for name, args in jobs:
    with (folder / f'{name}.log').open('x') as log:
        subprocess.run([sys.executable] + args, cwd=code, stdout=log, stderr=subprocess.STDOUT, check=True)
contract = export_contract(profile)
checks = {}
for name, completed in [('smoke64', 94), ('resume64', 96)]:
    output = folder / name
    run = json.loads((output / 'run.json').read_text())
    assert all(run['resume_checks'].values())
    assert run['actuator_profile'] == profile
    assert run['leg_target_max_delta_rad'] == 0.24
    saved = torch.load(output / f'model_{completed-1}.pt', map_location='cpu', weights_only=False)
    assert checkpoint_profile(saved) == profile
    assert saved['actuator_contract'] == contract
    _, actual = load_student_export(output / f'student_{completed}.pt')
    assert actual == contract
    rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    for row in rows:
        assert all(math.isfinite(row[key]) for key in ('value', 'surrogate', 'entropy', 'mean_latent_loss'))
        assert row['actuator_guard']['limits'] == contract['limits']
        for joint in row['actuator_guard']['joints'].values():
            assert joint['torque_violation_samples'] == 0
            assert joint['max_target_position_excess'] < 1e-6
    checks[name] = dict(completed_updates=completed, resume_checks=run['resume_checks'])
assert json.loads((folder / 'playback/manifest.json').read_text())['actuator_contract'] == contract
result = dict(passed=True, actual_target_delta_rad=0.24, checks=checks, contract=contract,
              scope='64-env migration, optimizer/curriculum resume, export/replay; not motion acceptance')
(folder / 'validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
