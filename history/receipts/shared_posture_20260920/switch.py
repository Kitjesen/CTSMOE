"""Preserve the live optimizer/course state when enabling shared surface posture."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
python='/root/autodl-tmp/conda/envs/isaaclab/bin/python'
old=root/'runs/mixed9_4096_60000_pit_posture_20260920'
new=root/'runs/mixed9_4096_60000_shared_posture_20260920'
assert not new.exists()
for name in ('shared_posture_smoke_20260920','shared_posture_inherit_20260920'):
    run=root/'runs'/name
    assert (run/'SUCCESS.json').exists()
    receipt=json.loads((run/'run.json').read_text())
    # Completed runs replace status.json with a compact success receipt. The
    # final measured training row remains in metrics.jsonl.
    state=json.loads((run/'metrics.jsonl').read_text().splitlines()[-1])
    assert all(receipt['resume_checks'].values())
    assert receipt['surface_posture_profile']=='level_surface_v1'
    assert receipt['pit_posture_profile']=='phase_stance_v1'
    assert state['surface_posture']['pit']['count']==0
    assert state['surface_posture']['flat']['count']>0
    assert receipt['uphill_speed_limit']==1.2
    assert receipt['flat_speed']['scheduled_limit_mps']==2.
    assert state['base_height_penalty_weight']==0.
pid=int((root/'pit_posture_formal.pid').read_text())
assert str(old).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
with (old/'STOP').open('x'):
    pass
deadline=time.monotonic()+120
while Path(f'/proc/{pid}').exists():
    if time.monotonic()>deadline:
        raise TimeoutError('Previous run has not exited; new training not started')
    time.sleep(1)
state=json.loads((old/'status.json').read_text())
assert state['state']=='stopped' and 'STOP requested' in state['error']
checkpoint=max(old.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
completed=int(checkpoint.stem.split('_')[1])+1
assert completed<60000
args=[python,'train.py','--root',str(root),'--output',str(new),'--num-envs','4096',
      '--updates',str(60000-completed),'--mixed-pit','--surface-posture-profile','level_surface_v1',
      '--resume',str(checkpoint)]
with (root/'shared_posture_formal.log').open('xb') as log:
    proc=subprocess.Popen(args,cwd=root/'code_shared_posture_20260920_v1',stdout=log,
        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
        env={**os.environ,'PYTHONUNBUFFERED':'1'})
(root/'shared_posture_formal.pid').write_text(str(proc.pid))
receipt=dict(old_run=str(old),new_run=str(new),checkpoint=str(checkpoint),
             completed_updates=completed,new_pid=proc.pid,args=args)
(root/'shared_posture_switch.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
