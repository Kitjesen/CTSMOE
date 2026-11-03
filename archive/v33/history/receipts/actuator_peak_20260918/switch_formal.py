"""Checkpoint-boundary transition from historical v1 to peak-corrected v2."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/actuator_peak_20260918'
old = root / 'runs/official4096_30000_rs0402guard_20260918'
old_pid = 80467
parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=('stop', 'start'))
args = parser.parse_args()
assert json.loads((audit / 'validation.json').read_text())['passed']
if args.phase == 'stop':
    cmdline = Path(f'/proc/{old_pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode()
    assert str(old) in cmdline and 'code_guard_20260918/train_v2.py' in cmdline
    status = json.loads((old / 'status.json').read_text())
    assert status['state'] == 'training'
    (audit / 'before_switch_status.json').write_text(json.dumps(status, indent=2))
    (old / 'STOP').touch(exist_ok=False)
    print(json.dumps({'state': 'stop_requested', 'completed_updates_before_stop': status['completed_updates']}))
else:
    assert json.loads((old / 'status.json').read_text())['state'] == 'stopped'
    assert not Path(f'/proc/{old_pid}').exists(), 'Old training process still exists'
    checkpoint = max(old.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    completed = saved['completed_updates']
    assert completed == int(checkpoint.stem.split('_')[1]) + 1
    assert saved['actuator_profile'] == 'rs0402_guard_v1'
    assert saved['terrain_levels'].shape == (4096,)
    assert saved['terrain_training_profile'] == 'stairs80_v1'
    assert saved['stair_command_profile'] == 'straight_speed_v1'
    assert 0 < 30000-completed < 30000
    output = root / 'runs/official4096_30000_rs0402peak_20260918'
    assert not output.exists()
    code = root / 'code_peak_20260918'
    argv = [sys.executable, str(code / 'train.py'), '--root', str(root), '--output', str(output),
            '--num-envs', '4096', '--updates', str(30000-completed), '--resume', str(checkpoint),
            '--actuator-profile', 'rs0402_peak_v2']
    with output.with_suffix('.log').open('x') as log:
        process = subprocess.Popen(argv, cwd=code, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    receipt = {'pid': process.pid, 'argv': argv, 'output': str(output), 'starting_completed_updates': completed,
               'source_checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')},
               'purpose': 'restore supported torque peaks, retain soft load and speed constraints',
               'previous_checkpoint_and_run_preserved': True}
    (audit / 'formal_launch.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))
