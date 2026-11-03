import json
from pathlib import Path
import torch

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
report = {}
for name in ('individual_pit_smoke_20260919', 'individual_pit_inherit_20260919'):
    p = root/'runs'/name
    assert json.loads((p/'SUCCESS.json').read_text())['state'] == 'complete'
    run = json.loads((p/'run.json').read_text())
    assert all(run['resume_checks'].values())
    assert run['pit_course_profile'] == 'individual_speed_v1'
    assert run['allocation_profile'] == 'balanced_adaptive_v1'
    assert run['mixed_checks']['pit_rewards_zero_on_old']
    rows = [json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
    assert all(r['pit']['frontier_height_m'] is None for r in rows)
    assert all(sum(j['torque_violation_samples'] for j in r['actuator_guard']['joints'].values()) == 0 for r in rows)
    result = rows[-1]['pit']['course']
    assert sum(v['promotions'] for v in result['by_learner'].values()) > 0
    report[name] = dict(start=run['starting_completed_updates'], end=rows[-1]['completed_updates'],
                       checks=run['resume_checks'], individual=result,
                       mean_seconds=sum(r['seconds'] for r in rows)/len(rows))
source = root/'runs/individual_pit_inherit_20260919/model_34447.pt'
saved = torch.load(source, weights_only=False, map_location='cpu')
assert saved['completed_updates'] == 34448
assert saved['pit_individual']['levels'].max() > 9
assert saved['pit_individual']['levels'].max() <= 21
assert all(torch.isfinite(v).all() for v in saved['model_state_dict'].values())
report['formal_resume'] = str(source)
(root/'individual_validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
