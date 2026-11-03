import json
from pathlib import Path

r=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
p=r/'runs/startup_repair_request_smoke_20260920'
run=json.loads((p/'run.json').read_text())
success=json.loads((p/'SUCCESS.json').read_text())
assert success['state']=='complete'
assert all(run['resume_checks'].values())
assert run['startup_repair_profile']=='omni_start_v3'
metrics=[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()]
assert metrics[-1]['completed_updates']==43114
assert all(m['startup_repair']['request_weight']==-1. for m in metrics)
assert not any(j['torque_violation_samples'] for m in metrics for j in m['actuator_guard']['joints'].values())
print('REQUEST_SMOKE_PASS',flush=True)
