"""Checkpoint-boundary stop and separately invoked guarded continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/actuator_guard_20260918'
old = root / 'runs/official4096_30000_straightspeed_20260918'
parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=('stop', 'start'))
args = parser.parse_args()
validation = json.loads((audit / 'validation.json').read_text())
assert validation['implementation_checks_passed']
if args.phase == 'stop':
    cmdline = Path('/proc/70061/cmdline').read_bytes().replace(b'\x00', b' ').decode()
    assert str(old) in cmdline and 'train_straight_speed_20260918.py' in cmdline
    status = json.loads((old / 'status.json').read_text())
    assert status['state'] == 'training'
    (audit / 'before_switch_status.json').write_text(json.dumps(status, indent=2))
    (old / 'STOP').touch(exist_ok=False)
    print(json.dumps({'state': 'stop_requested', 'completed_updates_before_stop': status['completed_updates']}))
else:
    assert json.loads((old / 'status.json').read_text())['state'] == 'stopped'
    assert not Path('/proc/70061').exists(), 'Old training process still exists'
    assert json.loads((audit / 'export_check.json').read_text())['passed']
    checkpoint = max(old.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    completed = saved['completed_updates']
    assert completed == int(checkpoint.stem.split('_')[1]) + 1
    assert saved['terrain_levels'].shape == (4096,)
    assert saved['terrain_training_profile'] == 'stairs80_v1'
    assert saved['stair_command_profile'] == 'straight_speed_v1'
    assert 0 < 30000-completed < 30000
    output = root / 'runs/official4096_30000_rs0402guard_20260918'
    assert not output.exists()
    code = root / 'code_guard_20260918'
    argv = [sys.executable, str(code / 'train_v2.py'), '--root', str(root), '--output', str(output),
            '--num-envs', '4096', '--updates', str(30000-completed), '--resume', str(checkpoint),
            '--actuator-profile', 'rs0402_guard_v1']
    with output.with_suffix('.log').open('x') as log:
        process = subprocess.Popen(argv, cwd=code, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    source = {'pid': process.pid, 'argv': argv, 'output': str(output), 'starting_completed_updates': completed,
              'source_checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')},
              'purpose': 'guarded adaptation training; not accepted for deployment',
              'original_checkpoint_and_run_preserved': True}
    (audit / 'formal_launch.json').write_text(json.dumps(source, indent=2))
    print(json.dumps(source))
