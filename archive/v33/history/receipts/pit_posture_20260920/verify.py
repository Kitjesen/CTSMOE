import json
import os
from pathlib import Path

root=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
run=root/'runs/mixed9_4096_60000_pit_posture_20260920'
handoff=json.loads((root/'pit_posture_switch.json').read_text())
r=json.loads((run/'run.json').read_text())
row=json.loads((run/'status.json').read_text())
os.kill(handoff['new_pid'],0)
assert all(r['resume_checks'].values())
assert row['state']=='training' and row['completed_updates']>handoff['completed_updates']
assert r['starting_completed_updates']==handoff['completed_updates']
assert r['target_completed_updates']==60000
assert row['pit_posture_profile']=='phase_stance_v1'
assert row['pit_posture']['platform']['count']>0
assert row['pit_posture']['approach']['count']>0
assert row['pit']['post_climb_observation_s']==6.
assert row['uphill_speed']['requested_limit_mps']==1.2
assert row['flat_speed']['scheduled_limit_mps']==2.
assert not any(j['torque_violation_samples'] for j in row['actuator_guard']['joints'].values())
result=dict(completed_updates=row['completed_updates'],seconds=row['seconds'],
            resume_checks=r['resume_checks'],starting_completed_updates=handoff['completed_updates'],
            posture=row['pit_posture'],counts=row['terrain_allocation']['actual_counts'],
            losses={k:row[k] for k in ('value','surrogate','mean_latent_loss')})
(run/'posture_verified.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
