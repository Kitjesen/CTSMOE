"""Verify real 4096-env continuation before starting the longer repair run."""
import hashlib
import json
from pathlib import Path
root=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
p=root/'runs/flat_v9_smoke_20260921'
run=json.loads((p/'run.json').read_text())
assert json.loads((p/'SUCCESS.json').read_text())['completed_updates']==43727
rows=[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()]
assert len(rows)==16
assert run['startup_repair_profile']=='omni_start_v9'
for key in ('model','ppo_optimizer','student_optimizer','terrain_restored_exactly','terrain_assignment','pit_individual','terrain_allocation'):
    assert run['resume_checks'][key],key
assert not (p/'phase_foundation_migration.json').exists()
for row in rows:
    state=row['flat_continuous']
    assert state['teacher']['count']+state['student']['count']==1639
    assert state['deadband_rad']==0 and state['weight']==-2
    assert row['phase_transition'] is not None
receipt=dict(completed_updates=43727,additional_smoke_updates=16,target_completed_updates=44711,
    resume_checks=run['resume_checks'],terrain_allocation=rows[-1]['terrain_allocation']['actual_counts'],
    flat_continuous=rows[-1]['flat_continuous'],phase_transition=rows[-1]['phase_transition'],
    last_seconds=rows[-1]['seconds'],
    code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'code_flat_v9_20260921').glob('*.py')})
(root/'flat_v9_deployment_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='code_hashes'},indent=2))
