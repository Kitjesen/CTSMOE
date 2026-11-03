"""Match pre-fix and fixed wave trials, retaining physical failures and slew checks."""
import json
from pathlib import Path

import numpy as np

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/recording')
folders = {'original_rate12':root/'wave_diagnosis_20260919/rate12',
           'surface_rate12':root/'wave_reset_20260919/surface_rate12'}
output, starts, records = {}, {}, {}
for name, folder in folders.items():
    result = json.loads((folder/'results.json').read_text())
    assert result['completed_updates'] == 25501 and result['ablation']['target_rate'] == 12
    assert not result['diagnostic_no_actuator_termination']
    records[name] = result
    starts[name] = np.array(json.loads((folder/'initial_geometry.json').read_text())['points_w'])
    with np.load(folder/'stair_failure_traces.npz') as trace:
        ids = [result['joint_names'].index(n) for n in trace['leg_action_joint_names']]
        soft = np.array(result['soft_limits_rad'])[ids]
        previous = np.clip(trace['previous_leg_target_rad'], soft[:, 0], soft[:, 1])
        requested = np.clip(trace['requested_leg_target_rad'], soft[:, 0], soft[:, 1])
        expected = np.clip(previous + np.clip(requested-previous, -.24, .24), soft[:, 0], soft[:, 1])
        active = trace['active_before_step'].astype(bool)
        np.testing.assert_allclose(trace['joint_pos_target_rad'][:, :, ids][active], expected[active], atol=2e-6, rtol=0)
    output[name] = {}
    for learner in ('teacher', 'student'):
        rows = [c for c in result['cases'] if c['learner'] == learner]
        output[name][learner] = dict(count=len(rows), passes=sum(c['forward_exit'] for c in rows),
             early_failures=sum(not c['forward_exit'] and c['duration_s'] < .35 for c in rows),
             actuator=sum(c['termination_reasons']['actuator_violation'] for c in rows),
             base_contact=sum(c['termination_reasons']['illegal_contact'] for c in rows),
             timeouts=sum(c['termination_reasons']['time_out'] for c in rows),
             by_level={str(level):sum(c['forward_exit'] for c in rows if c['level']==level) for level in (0, 3, 6, 9)})
for before, after in zip(records['original_rate12']['cases'], records['surface_rate12']['cases']):
    for field in ('learner', 'terrain', 'level', 'speed', 'repeat', 'column'):
        assert before[field] == after[field], field
np.testing.assert_allclose(starts['original_rate12'][..., :2], starts['surface_rate12'][..., :2], atol=2e-5, rtol=0)
for points in starts.values():
    points[:, :, 2] -= points[:, :1, 2].copy()
np.testing.assert_allclose(starts['original_rate12'], starts['surface_rate12'], atol=2e-5, rtol=0)
comparison = dict(checkpoint=25501, controls_matched=True, actual_slew_verified=True, groups=output)
(root/'wave_reset_20260919/motion_comparison.json').write_text(json.dumps(comparison, indent=2))
print(json.dumps(comparison, indent=2))
