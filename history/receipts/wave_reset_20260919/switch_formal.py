"""Save the live slew12 trainer and resume with validated wave-only placement."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root/'recording/wave_reset_20260919'
old = root/'runs/official4096_30000_slew12_20260919'
old_pid = 103171
parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=('stop', 'start'))
args = parser.parse_args()
assert json.loads((folder/'geometry_validation.json').read_text())['passed']
assert json.loads((folder/'training_validation.json').read_text())['passed']
comparison = json.loads((folder/'motion_comparison.json').read_text())
assert comparison['controls_matched'] and comparison['actual_slew_verified']
baseline, fixed = comparison['groups']['original_rate12'], comparison['groups']['surface_rate12']
assert all(fixed[name]['early_failures'] < baseline[name]['early_failures'] for name in ('teacher', 'student'))
assert all(fixed[name]['passes'] >= baseline[name]['passes'] for name in ('teacher', 'student'))
assert json.loads((folder/'random_comparison.json').read_text())['not_worse']
if args.phase == 'stop':
    cmdline = Path(f'/proc/{old_pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode()
    assert str(old) in cmdline and 'code_slew12_20260919/train.py' in cmdline
    status = json.loads((old/'status.json').read_text())
    assert status['state'] == 'training'
    (folder/'before_switch_status.json').write_text(json.dumps(status, indent=2))
    (old/'STOP').touch(exist_ok=False)
    print('SAVE_THEN_STOP_REQUESTED', status['completed_updates'])
else:
    assert json.loads((old/'status.json').read_text())['state'] == 'stopped'
    assert not Path(f'/proc/{old_pid}').exists()
    checkpoint = max(old.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    completed = saved['completed_updates']
    assert completed == int(checkpoint.stem.split('_')[1])+1
    assert saved['actuator_profile'] == 'rs0402_slew12_v3'
    assert saved.get('wave_reset_profile', 'original') == 'original'
    assert saved['terrain_levels'].shape == (4096,)
    assert 0 < 30000-completed < 30000
    output, code = root/'runs/official4096_30000_wave_surface_20260919', root/'code_wave_reset_20260919'
    assert not output.exists()
    command = [sys.executable, str(code/'train.py'), '--root', str(root), '--output', str(output),
               '--num-envs', '4096', '--updates', str(30000-completed), '--resume', str(checkpoint),
               '--wave-reset-profile', 'wave_surface_v1']
    with output.with_suffix('.log').open('x') as log:
        process = subprocess.Popen(command, cwd=code, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    receipt = dict(pid=process.pid, command=command, output=str(output), starting_completed_updates=completed,
                   checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   target_completed_updates=30000, wave_reset_profile='wave_surface_v1',
                   actuator_profile='rs0402_slew12_v3', code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')})
    (folder/'formal_launch.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))
