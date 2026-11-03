"""Save at an iteration boundary and resume with the explicit slew12 v3 profile."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root / 'recording/wave_diagnosis_20260919'
old = root / 'runs/official4096_30000_rs0402peak_20260918'
old_pid = 83126
parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=('stop', 'start'))
args = parser.parse_args()
assert json.loads((folder / 'validation.json').read_text())['passed']
if args.phase == 'stop':
    cmdline = Path(f'/proc/{old_pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode()
    assert str(old) in cmdline and 'code_peak_20260918/train.py' in cmdline
    status = json.loads((old / 'status.json').read_text())
    assert status['state'] == 'training'
    (folder / 'before_switch_status.json').write_text(json.dumps(status, indent=2))
    (old / 'STOP').touch(exist_ok=False)
    print(json.dumps(dict(state='save_then_stop_requested', completed_updates=status['completed_updates'])))
else:
    assert json.loads((old / 'status.json').read_text())['state'] == 'stopped'
    assert not Path(f'/proc/{old_pid}').exists(), 'Prior training still exists'
    source = max(old.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    completed = checkpoint['completed_updates']
    assert completed == int(source.stem.split('_')[1]) + 1
    assert checkpoint['actuator_profile'] == 'rs0402_peak_v2'
    assert checkpoint['terrain_levels'].shape == (4096,)
    assert checkpoint['terrain_training_profile'] == 'stairs80_v1'
    assert checkpoint['stair_command_profile'] == 'straight_speed_v1'
    assert 0 < 30000 - completed < 30000
    code = root / 'code_slew12_20260919'
    output = root / 'runs/official4096_30000_slew12_20260919'
    assert not output.exists()
    command = [sys.executable, str(code / 'train.py'), '--root', str(root), '--output', str(output),
               '--num-envs', '4096', '--updates', str(30000-completed), '--resume', str(source),
               '--actuator-profile', 'rs0402_slew12_v3']
    with output.with_suffix('.log').open('x') as log:
        process = subprocess.Popen(command, cwd=code, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    receipt = dict(pid=process.pid, command=command, output=str(output), starting_completed_updates=completed,
                   source_checkpoint=str(source), checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   only_limit_change='leg target rate 6 to 12 rad/s; per 20 ms target delta 0.12 to 0.24 rad',
                   target_completed_updates=30000,
                   code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')})
    (folder / 'formal_launch.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))
