"""Verify completed isolated checks and summarize guarded frozen-policy trials."""
import json
from pathlib import Path

import torch


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/actuator_guard_20260918'
result = {}
for folder, count in (('smoke64', 80), ('resume64', 82)):
    path = audit / folder
    assert json.loads((path / 'SUCCESS.json').read_text())['completed_updates'] == count
    run = json.loads((path / 'run.json').read_text())
    assert run['actuator_profile'] == 'rs0402_guard_v1'
    assert all(run['resume_checks'].values())
    ckpt = torch.load(path / f'model_{count-1}.pt', map_location='cpu', weights_only=False)
    assert ckpt['actuator_profile'] == 'rs0402_guard_v1'
    assert ckpt['completed_updates'] == count
    rows = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
    assert len(rows) == (8 if folder == 'smoke64' else 2)
    for row in rows:
        stats = row['actuator_guard']
        assert stats['samples_per_joint'] == 64 * 24 * 4
        assert all(j['torque_violation_samples'] == 0 for j in stats['joints'].values())
        assert max(j['max_target_position_excess'] for j in stats['joints'].values()) < 1e-6
    result[folder] = {'updates': count, 'resume_checks': run['resume_checks'],
                      'profile': run['actuator_profile'], 'last_seconds': rows[-1]['seconds']}

data = json.loads((audit / 'heading19001_guard/results.json').read_text())
assert data['actuator_profile'] == 'rs0402_guard_v1'
assert data['completed_updates'] == 19001
assert len(data['cases']) == 162
result['frozen_student'] = {}
names = data['joint_names']
for terrain in ('wave', 'stairs_up', 'stairs_down'):
    rows = [c for c in data['cases'] if c['learner'] == 'student' and c['terrain'] == terrain]
    summary = {'trials': len(rows), 'forward_exits': sum(c['forward_exit'] for c in rows),
               'physical_failures': sum(c['physical_failure'] for c in rows), 'by_level': {}}
    for level in (3, 6, 9):
        cs = [c for c in rows if c['level'] == level]
        summary['by_level'][level] = {'exits': sum(c['forward_exit'] for c in cs),
                                     'failures': sum(c['physical_failure'] for c in cs)}
    for kind in ('hip', 'thigh', 'calf', 'foot'):
        ids = [i for i, name in enumerate(names) if name.endswith(kind + '_joint')]
        summary[kind] = {key: max(c['joints'][key][i] for c in rows for i in ids) for key in (
            'max_applied_torque_nm', 'max_speed_rad_s', 'max_hard_excess_rad', 'max_target_hard_excess_rad')}
    result['frozen_student'][terrain] = summary
for case in data['cases']:
    assert sum(case['joints']['torque_excess_samples']) == 0
    assert max(case['joints']['max_target_hard_excess_rad']) < 1e-6
result['implementation_checks_passed'] = True
result['motion_acceptance_passed'] = all(
    case['forward_exit'] and not case['physical_failure']
    and not sum(case['joints']['speed_excess_samples'])
    and not sum(case['joints']['hard_excess_samples'])
    for case in data['cases'] if case['learner'] == 'student'
)
result['scope'] = 'implementation and budget checks; frozen pass counts are not training improvement'
(audit / 'validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
