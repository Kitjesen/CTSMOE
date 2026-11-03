"""Check the randomized one-second start window separately from traversal results."""
import json
from pathlib import Path

import numpy as np

folder = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/recording/wave_reset_20260919')
output, positions = {}, {}
for name in ('random_original', 'random_surface'):
    result = json.loads((folder/name/'results.json').read_text())
    assert 'First one second' in result['scope']
    assert result['completed_updates'] == 25501 and result['ablation']['target_rate'] == 12
    assert not result['diagnostic_no_actuator_termination']
    positions[name] = np.array(json.loads((folder/name/'initial_geometry.json').read_text())['points_w'])
    output[name] = {}
    for learner in ('teacher', 'student'):
        rows = [c for c in result['cases'] if c['learner'] == learner]
        output[name][learner] = dict(count=len(rows), failures_first_second=sum(c['physical_failure'] for c in rows),
             failures_first_035s=sum(c['physical_failure'] and c['duration_s'] < .35 for c in rows),
             actuator=sum(c['termination_reasons']['actuator_violation'] for c in rows),
             base_contact=sum(c['termination_reasons']['illegal_contact'] for c in rows))
np.testing.assert_allclose(positions['random_original'][..., :2], positions['random_surface'][..., :2], atol=2e-5, rtol=0)
for points in positions.values():
    points[:, :, 2] -= points[:, :1, 2].copy()
np.testing.assert_allclose(positions['random_original'], positions['random_surface'], atol=2e-5, rtol=0)
not_worse = all(output['random_surface'][name]['failures_first_second'] <= output['random_original'][name]['failures_first_second']
                for name in ('teacher', 'student'))
report = dict(controls_matched=True, not_worse=not_worse, groups=output,
              scope='One second, randomized initial pose/joints/root velocities; no observation/parameter/push randomization; not a traversal success rate')
(folder/'random_comparison.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
