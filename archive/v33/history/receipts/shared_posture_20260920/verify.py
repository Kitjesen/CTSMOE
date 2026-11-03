import json
import os
from pathlib import Path

root=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
run=root/'runs/mixed9_4096_60000_shared_posture_20260920'
switch=json.loads((root/'shared_posture_switch.json').read_text())
receipt=json.loads((run/'run.json').read_text())
state=json.loads((run/'status.json').read_text())
os.kill(switch['new_pid'],0)
assert all(receipt['resume_checks'].values())
assert state['state']=='training' and state['completed_updates']>switch['completed_updates']
assert receipt['starting_completed_updates']==switch['completed_updates']
assert receipt['target_completed_updates']==60000
assert state['surface_posture_profile']=='level_surface_v1'
assert state['pit_posture_profile']=='phase_stance_v1'
assert state['surface_posture']['pit']['count']==0
assert state['surface_posture']['flat']['count']>0
assert state['base_height_penalty_weight']==0.
assert not any(j['torque_violation_samples'] for j in state['actuator_guard']['joints'].values())
result=dict(completed_updates=state['completed_updates'],seconds=state['seconds'],
            starting_completed_updates=switch['completed_updates'],pid=switch['new_pid'],
            resume_checks=receipt['resume_checks'],surface_posture=state['surface_posture'],
            pit_posture=state['pit_posture'],counts=state['terrain_allocation']['actual_counts'],
            losses={k:state[k] for k in ('value','surrogate','mean_latent_loss')})
(root/'shared_posture_verified.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
