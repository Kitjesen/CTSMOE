"""Save the active training run, then resume it with verified flat speed code."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
PYTHON = '/root/autodl-tmp/conda/envs/isaaclab/bin/python'
old = ROOT/'runs/mixed9_4096_60000_individualpit_20260919'
new = ROOT/'runs/mixed9_4096_60000_flat_speed_20260920'
assert not new.exists()
for name in ('flat_speed_smoke_20260920', 'flat_speed_inherit_20260920'):
    run = ROOT/'runs'/name
    assert (run/'SUCCESS.json').exists(), name
    receipt = json.loads((run/'run.json').read_text())
    assert all(receipt['resume_checks'].values()), receipt['resume_checks']
    assert receipt['flat_speed']['scheduled_limit_mps'] >= 1.5
old_pid = int((ROOT/'individual_formal.pid').read_text())
assert str(old).encode() in Path(f'/proc/{old_pid}/cmdline').read_bytes()
with (old/'STOP').open('x'):
    pass
deadline = time.monotonic() + 120
while Path(f'/proc/{old_pid}').exists():
    if time.monotonic() > deadline:
        raise TimeoutError('Old training did not stop; new training was not launched')
    time.sleep(1)
state = json.loads((old/'status.json').read_text())
assert state['state'] == 'stopped' and 'STOP requested' in state['error'], state
checkpoint = max(old.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
completed = int(checkpoint.stem.split('_')[1]) + 1
assert completed < 60000
args = [PYTHON, 'train.py', '--root', str(ROOT), '--output', str(new),
        '--num-envs', '4096', '--updates', str(60000-completed), '--mixed-pit',
        '--expand-flat-speed', '--resume', str(checkpoint)]
with (ROOT/'flat_speed_formal.log').open('xb') as log:
    proc = subprocess.Popen(args, cwd=ROOT/'code_flat_speed_20260920_v1',
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env={**os.environ, 'PYTHONUNBUFFERED': '1'}, start_new_session=True)
(ROOT/'flat_speed_formal.pid').write_text(str(proc.pid))
receipt = dict(old_run=str(old), old_pid=old_pid, checkpoint=str(checkpoint),
               completed_updates=completed, new_run=str(new), new_pid=proc.pid, args=args)
(ROOT/'flat_speed_switch.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
