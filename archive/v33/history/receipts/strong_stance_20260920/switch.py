"""Switch the live run only after measured short training and resume checks pass."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
python='/root/autodl-tmp/conda/envs/isaaclab/bin/python'
old=root/'runs/mixed9_4096_60000_shared_posture_20260920'
new=root/'runs/mixed9_4096_60000_strong_stance_20260920'
assert not new.exists()
for name in ('strong_stance_smoke_20260920','strong_stance_inherit_20260920'):
    run=root/'runs'/name
    assert (run/'SUCCESS.json').exists()
    receipt=json.loads((run/'run.json').read_text())
    state=json.loads((run/'metrics.jsonl').read_text().splitlines()[-1])
    assert all(receipt['resume_checks'].values())
    assert receipt['stance_strength_profile']=='strong_stand_v1'
    assert state['surface_posture']['height_weight']==-2.
    assert not state['surface_posture']['height_command_gating']
    assert state['pit_posture']['height_weight']==-2.
    assert state['stance_strength']['zero_command_weight']==-2.
    assert state['stance_strength']['all']['count']>0
    assert state['surface_posture']['pit']['height_active_count']==0
    assert receipt['uphill_speed_limit']==1.2
    assert receipt['flat_speed']['scheduled_limit_mps']==2.
    assert state['base_height_penalty_weight']==0.
pid=int((root/'shared_posture_formal.pid').read_text())
assert str(old).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
with (old/'STOP').open('x'):
    pass
deadline=time.monotonic()+120
while Path(f'/proc/{pid}').exists():
    if time.monotonic()>deadline:
        raise TimeoutError('Previous training has not stopped; new run not launched')
    time.sleep(1)
state=json.loads((old/'status.json').read_text())
assert state['state']=='stopped' and 'STOP requested' in state['error']
checkpoint=max(old.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
completed=int(checkpoint.stem.split('_')[1])+1
assert completed<60000
args=[python,'train.py','--root',str(root),'--output',str(new),'--num-envs','4096',
      '--updates',str(60000-completed),'--mixed-pit','--stance-strength-profile','strong_stand_v1',
      '--resume',str(checkpoint)]
with (root/'strong_stance_formal.log').open('xb') as log:
    proc=subprocess.Popen(args,cwd=root/'code_strong_stance_20260920_v1',stdout=log,
        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
        env={**os.environ,'PYTHONUNBUFFERED':'1'})
(root/'strong_stance_formal.pid').write_text(str(proc.pid))
receipt=dict(old_run=str(old),new_run=str(new),checkpoint=str(checkpoint),
             completed_updates=completed,new_pid=proc.pid,args=args)
(root/'strong_stance_switch.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
