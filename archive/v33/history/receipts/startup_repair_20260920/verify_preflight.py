"""Require real sim and checkpoint inheritance before starting bounded repair."""
import json
from pathlib import Path

r=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
records=[]
for name,count in [('startup_repair_smoke_20260920',42922),('startup_repair_inherit_20260920',42924)]:
    p=r/'runs'/name
    success=json.loads((p/'SUCCESS.json').read_text())
    receipt=json.loads((p/'run.json').read_text())
    assert (p/f'model_{count-1}.pt').is_file()
    assert success['state']=='complete'
    assert all(receipt['resume_checks'].values())
    assert receipt['startup_repair_profile']=='omni_start_v1'
    assert receipt['startup_repair']['target_weight']==-2.
    assert receipt['startup_repair']['vertical_weight']==-2.
    assert receipt['terrain_allocation']['bounds_profile']=='startup_flat8_v1'
    assert receipt['terrain_allocation']['target_weights']['flat']==.08
    last=json.loads((p/'metrics.jsonl').read_text().splitlines()[-1])
    assert last['completed_updates']==count
    assert not any(x['torque_violation_samples'] for x in last['actuator_guard']['joints'].values())
    records.append(dict(run=name,completed=count,resume_checks=receipt['resume_checks'],
                        repair=last['startup_repair'],allocation=last['terrain_allocation']))
(r/'startup_repair_preflight_verified.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
