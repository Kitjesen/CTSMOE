"""Resume the saved formal run with the authorized stair and height settings."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/stair_focus_20260918'
old = root / 'runs/official4096_30000_speed1_20260918'
output = root / 'runs/official4096_30000_stairs80_noheight_20260918'
status = json.loads((old / 'status.json').read_text())
assert status == {'state': 'stopped', 'error': 'STOP requested after saving'}, status
assert not output.exists(), output
checkpoint = max(old.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
required = {'model_state_dict', 'optimizer_state_dict', 'optimizer_stu_enc_state_dict',
            'completed_updates', 'common_step_counter', 'terrain_levels', 'terrain_types',
            'learning_rate', 'command_curriculum_start'}
assert required <= saved.keys(), required - saved.keys()
assert saved['command_curriculum_start'] == 'intermediate'
completed = saved['completed_updates']
assert completed == int(checkpoint.stem.split('_')[1]) + 1
assert saved['terrain_levels'].shape == (4096,)
assert (old / f'student_{completed}.pt').exists()
remaining = 30000-completed
assert 0 < remaining < 30000
script = root / 'code/train_stair_focus_20260918.py'
helper = root / 'code/training_profiles.py'
args = [sys.executable, str(script), '--root', str(root), '--output', str(output),
        '--num-envs', '4096', '--updates', str(remaining), '--resume', str(checkpoint),
        '--base-height-penalty', 'off', '--terrain-training-profile', 'stairs80_v1']
receipt = {'checkpoint': str(checkpoint),
           'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
           'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
           'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
           'completed_updates': completed, 'remaining_updates': remaining,
           'total_target': 30000, 'previous_status': status,
           'terrain_mean_before_resume': saved['terrain_levels'].float().mean().item(),
           'common_step_counter': saved['common_step_counter'],
           'learning_rate': saved['learning_rate'], 'argv': args}
with (audit / 'formal_switch.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print('FORMAL_SWITCH', json.dumps(receipt), flush=True)
del saved
os.execv(sys.executable, args)
