"""Checkpoint the active run and continue with the verified uphill speed range."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
PYTHON = '/root/autodl-tmp/conda/envs/isaaclab/bin/python'
old = ROOT/'runs/mixed9_4096_60000_flat_speed_20260920'
new = ROOT/'runs/mixed9_4096_60000_uphill_speed_20260920'
assert not new.exists()
receipts = []
for name in ('uphill_speed_smoke_20260920', 'uphill_speed_inherit_20260920'):
    run = ROOT/'runs'/name
    assert (run/'SUCCESS.json').exists(), name
    receipt = json.loads((run/'run.json').read_text())
    assert all(receipt['resume_checks'].values()), receipt['resume_checks']
    assert receipt['uphill_speed_limit'] == 1.2
    assert receipt['uphill_speed']['effective_x_ranges'][0][1] > 1.19
    assert receipt['flat_speed']['scheduled_limit_mps'] >= 1.5
    receipts.append(receipt)
assert receipts[0]['flat_speed_course'] == receipts[1]['flat_speed_course']
old_pid = int((ROOT/'flat_speed_formal.pid').read_text())
assert str(old).encode() in Path(f'/proc/{old_pid}/cmdline').read_bytes()
with (old/'STOP').open('x'):
    pass
deadline = time.monotonic() + 120
while Path(f'/proc/{old_pid}').exists():
    if time.monotonic() > deadline:
        raise TimeoutError('Old run did not stop; new training was not launched')
    time.sleep(1)
state = json.loads((old/'status.json').read_text())
assert state['state'] == 'stopped' and 'STOP requested' in state['error'], state
checkpoint = max(old.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
completed = int(checkpoint.stem.split('_')[1]) + 1
assert completed < 60000
args = [PYTHON, 'train.py', '--root', str(ROOT), '--output', str(new),
        '--num-envs', '4096', '--updates', str(60000-completed), '--mixed-pit',
        '--uphill-speed-limit', '1.2', '--resume', str(checkpoint)]
with (ROOT/'uphill_speed_formal.log').open('xb') as log:
    proc = subprocess.Popen(args, cwd=ROOT/'code_uphill_speed_20260920_v1',
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env={**os.environ, 'PYTHONUNBUFFERED': '1'}, start_new_session=True)
(ROOT/'uphill_speed_formal.pid').write_text(str(proc.pid))
result = dict(old_run=str(old), old_pid=old_pid, checkpoint=str(checkpoint),
              completed_updates=completed, new_run=str(new), new_pid=proc.pid, args=args)
(ROOT/'uphill_speed_switch.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
