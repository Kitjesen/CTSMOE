"""Checkpoint active training, then resume with the validated posture objective."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
python='/root/autodl-tmp/conda/envs/isaaclab/bin/python'
old=root/'runs/mixed9_4096_60000_uphill_speed_20260920'
new=root/'runs/mixed9_4096_60000_pit_posture_20260920'
assert not new.exists()
receipts=[]
for name in ('pit_posture_smoke_20260920','pit_posture_inherit_20260920'):
    run=root/'runs'/name
    assert (run/'SUCCESS.json').exists()
    r=json.loads((run/'run.json').read_text())
    assert all(r['resume_checks'].values())
    assert r['pit_posture_profile']=='phase_stance_v1'
    assert r['pit']['post_climb_observation_s']==6.
    assert r['uphill_speed_limit']==1.2 and r['flat_speed']['scheduled_limit_mps']==2.
    receipts.append(r)
assert receipts[0]['flat_speed_course']==receipts[1]['flat_speed_course']
probe=json.loads((root/'recording/pit_posture_frozen_20260920/manifest.json').read_text())
assert all(c['climb'] and not c['stable'] for c in probe['cases']), probe['cases']
pid=int((root/'uphill_speed_formal.pid').read_text())
assert str(old).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
with (old/'STOP').open('x'):
    pass
deadline=time.monotonic()+120
while Path(f'/proc/{pid}').exists():
    if time.monotonic()>deadline:
        raise TimeoutError('Old training did not stop; new run was not launched')
    time.sleep(1)
state=json.loads((old/'status.json').read_text())
assert state['state']=='stopped' and 'STOP requested' in state['error']
checkpoint=max(old.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
completed=int(checkpoint.stem.split('_')[1])+1
assert completed<60000
args=[python,'train.py','--root',str(root),'--output',str(new),'--num-envs','4096',
      '--updates',str(60000-completed),'--mixed-pit','--pit-posture-profile','phase_stance_v1',
      '--resume',str(checkpoint)]
with (root/'pit_posture_formal.log').open('xb') as log:
    proc=subprocess.Popen(args,cwd=root/'code_pit_posture_20260920_v1',stdout=log,
        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
        env={**os.environ,'PYTHONUNBUFFERED':'1'})
(root/'pit_posture_formal.pid').write_text(str(proc.pid))
receipt=dict(old_run=str(old),new_run=str(new),checkpoint=str(checkpoint),
             completed_updates=completed,new_pid=proc.pid,args=args)
(root/'pit_posture_switch.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
