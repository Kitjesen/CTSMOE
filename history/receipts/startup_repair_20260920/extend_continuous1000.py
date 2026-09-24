"""Continue the saved V7 experiment to 1000 total new updates, then evaluate."""
import json
from pathlib import Path
import subprocess
import sys

import torch

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
previous = root / 'runs/startup_repair_continuous96_20260920'
output = root / 'runs/startup_repair_continuous1000_20260920'
code = root / 'code_startup_repair_20260920_v7'
target = 43106 + 1000
status = json.loads((previous / 'status.json').read_text())
assert status['state'] == 'stopped', status
assert status['error'] == 'STOP requested after saving', status
assert not output.exists(), output
resume = max(previous.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
saved = torch.load(resume, map_location='cpu', weights_only=False)
completed = saved['completed_updates']
assert completed == int(resume.stem.split('_')[1]) + 1
assert saved['startup_repair_profile'] == 'omni_start_v7'
assert 43106 < completed < target, completed
remaining = target - completed
del saved
selection_path = root / 'startup_continuous1000_selection.json'
selection = [
    dict(path='runs/startup_repair_repeat96_20260920/model_43105.pt',
         label='43106_before_continuous1000', teacher=False),
    dict(path=f'{output.relative_to(root)}/model_{target-1}.pt',
         label=f'{target}_continuous1000', teacher=False),
    dict(path=f'{output.relative_to(root)}/model_{target-1}.pt',
         label=f'{target}_continuous1000_teacher', teacher=True),
]
with selection_path.open('x') as stream:
    json.dump(selection, stream, indent=2)
receipt = dict(resume=str(resume), resumed_completed_updates=completed,
               initial_completed_updates=43106, experiment_total_updates=1000,
               remaining_updates=remaining, target_completed_updates=target,
               code=str(code), output=str(output), selection=str(selection_path))
with (root / 'startup_continuous1000_extension.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt), flush=True)
subprocess.run([sys.executable, 'train.py', '--root', str(root), '--output', str(output),
                '--num-envs', '4096', '--updates', str(remaining), '--mixed-pit',
                '--startup-repair-profile', 'omni_start_v7', '--resume', str(resume)],
               cwd=code, check=True)
success = json.loads((output / 'SUCCESS.json').read_text())
assert success['completed_updates'] == target, success
for script, destination in (
    ('startup_repair_audit.py', 'startup_continuous1000_eval_20260920'),
    ('startup_repair_regression.py', 'startup_continuous1000_regression_20260920'),
):
    subprocess.run([sys.executable, str(root / script), '--root', str(root),
                    '--selection', str(selection_path), '--output',
                    str(root / 'recording' / destination)], cwd=code, check=True)
print('CONTINUOUS1000_PIPELINE_COMPLETE', flush=True)
