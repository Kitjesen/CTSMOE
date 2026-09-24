"""Start one checked baseline or formal continuation from the same saved checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/stair_speed_20260918'
old = root / 'runs/official4096_30000_stairs80_noheight_20260918'
parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=('baseline', 'formal'))
args = parser.parse_args()
assert json.loads((audit / 'smoke_verified.json').read_text())['checks_passed']
assert json.loads((old / 'status.json').read_text()) == {'state': 'stopped', 'error': 'STOP requested after saving'}
if args.phase == 'baseline':
    checkpoint = max(old.glob('model_*.pt'), key=lambda path: int(path.stem.split('_')[1]))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    completed = saved['completed_updates']
    assert completed == int(checkpoint.stem.split('_')[1]) + 1
    assert saved['terrain_levels'].shape == (4096,)
    assert saved['terrain_training_profile'] == 'stairs80_v1'
    assert saved['base_height_penalty'] == 'off'
    assert (old / f'student_{completed}.pt').exists()
    source = {'checkpoint': str(checkpoint), 'completed_updates': completed,
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'terrain_mean_before_resume': saved['terrain_levels'].float().mean().item(),
              'learning_rate': saved['learning_rate'], 'common_step_counter': saved['common_step_counter'],
              'code_sha256': {name: hashlib.sha256((root / 'code' / name).read_bytes()).hexdigest()
                              for name in ('train_straight_speed_20260918.py', 'stair_speed.py',
                                           'stair_speed_logic.py', 'evaluate_stair_speeds.py')}}
    with (audit / 'switch_source.json').open('x') as stream:
        json.dump(source, stream, indent=2)
    output = audit / 'baseline'
    argv = [sys.executable, str(root / 'code/evaluate_stair_speeds.py'), '--root', str(root),
            '--checkpoint', str(checkpoint), '--output', str(output)]
else:
    source = json.loads((audit / 'switch_source.json').read_text())
    baseline = json.loads((audit / 'baseline/results.json').read_text())
    assert json.loads((audit / 'baseline/status.json').read_text())['state'] == 'complete'
    assert baseline['checkpoint_sha256'] == source['checkpoint_sha256']
    assert len(baseline['cases']) == 192
    for terrain in ('stairs_up', 'stairs_down'):
        assert any(case['reference_exit_pass'] for case in baseline['cases']
                   if case['learner'] == 'student' and case['terrain'] == terrain and case['level'] == 3)
    for name, digest in source['code_sha256'].items():
        assert hashlib.sha256((root / 'code' / name).read_bytes()).hexdigest() == digest, name
    output = root / 'runs/official4096_30000_straightspeed_20260918'
    remaining = 30000 - source['completed_updates']
    assert 0 < remaining < 30000
    argv = [sys.executable, str(root / 'code/train_straight_speed_20260918.py'), '--root', str(root),
            '--output', str(output), '--num-envs', '4096', '--updates', str(remaining),
            '--resume', source['checkpoint'], '--stair-command-profile', 'straight_speed_v1']
assert not output.exists(), output
log = audit / f'{args.phase}.log' if args.phase == 'baseline' else Path(str(output) + '.log')
with log.open('x') as stream:
    child = subprocess.Popen(argv, cwd=root / 'code', stdin=subprocess.DEVNULL, stdout=stream,
                             stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
receipt = {'pid': child.pid, 'argv': argv, 'log': str(log), 'output': str(output), 'source': source}
with (audit / f'{args.phase}_launch.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
