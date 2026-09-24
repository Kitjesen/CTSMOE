import json
from pathlib import Path
import torch

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
report = {}
for name in ('allocation_smoke_v2_20260919', 'allocation_inherit_v2_20260919', 'allocation_reset_probe_v2_20260919'):
    p = root/'runs'/name
    assert json.loads((p/'SUCCESS.json').read_text())['state'] == 'complete'
    run = json.loads((p/'run.json').read_text())
    checks = run['resume_checks']
    assert all(v for k,v in checks.items() if k != 'terrain_restored_exactly')
    assert checks['terrain_restored_exactly'] == ('smoke' not in name)
    assert run['allocation_profile'] == 'balanced_adaptive_v1'
    assert run['mixed_checks']['pit_rewards_zero_on_old']
    assert run['actuator_profile'] == 'rs0402_transient_v4'
    assert run['pit_training_profile'] == 'climb_first_v2'
    assert run['slope_profile'] == 'slope45_v1'
    rows = [json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
    assert all(sum(v['count'] for v in r['terrain_by_type'].values()) == 4096 for r in rows)
    assert all(sum(j['torque_violation_samples'] for j in r['actuator_guard']['joints'].values()) == 0 for r in rows)
    report[name] = dict(start=run['starting_completed_updates'], end=rows[-1]['completed_updates'],
                        checks=checks, counts=rows[-1]['terrain_allocation']['actual_counts'],
                        mean_seconds=sum(r['seconds'] for r in rows)/len(rows))
probe = root/'runs/allocation_reset_probe_v2_20260919/RESET_BOUNDARY_VERIFIED.json'
report['live_reset_probe'] = json.loads(probe.read_text())
assert report['live_reset_probe']['moves_applied'] > 0
source = root/'runs/allocation_inherit_v2_20260919/model_33366.pt'
saved = torch.load(source, weights_only=False, map_location='cpu')
assert saved['completed_updates'] == 33367
assert saved['terrain_allocation']['course']['weights']['pit'] == .4
assert not saved['terrain_allocation']['pending']
assert all(torch.isfinite(v).all() for v in saved['model_state_dict'].values())
report['formal_resume'] = str(source)
(root/'allocation_validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
