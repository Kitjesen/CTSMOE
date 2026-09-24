"""Verify added telemetry and compare unchanged action traces with the prior probe."""
import json
from pathlib import Path

import numpy as np


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
output = root / 'recording/all_scene_phase_smoke_20260921'
manifest = json.loads((output / 'manifest.json').read_text())
assert manifest['smoke_only'] is True
assert len(manifest['cases']) == 213
pit = np.array([c['terrain'] == 'pit' for c in manifest['cases']])
result = {'cases': 213, 'smoke_only': True, 'telemetry': {}, 'comparison': {}}
with np.load(output / 'traces.npz') as current:
    for key, shape in [('ground_relief', (3, 213)), ('ground_scan_valid', (3, 213)),
                       ('wheel_clearance', (3, 213, 4)), ('wheel_ground_force', (3, 213, 4)),
                       ('pit_front_gap', (3, 213)), ('platform_stop', (3, 213))]:
        value = current[key]
        assert value.shape == shape, (key, value.shape)
        if key == 'pit_front_gap':
            assert np.isfinite(value[:, pit]).all()
            assert np.isnan(value[:, ~pit]).all()
        elif key == 'ground_relief':
            assert np.isfinite(value[current['ground_scan_valid']]).all()
        else:
            assert np.isfinite(value).all(), key
        result['telemetry'][key] = {'shape': list(shape), 'finite': int(np.isfinite(value).sum())}
    assert (current['wheel_ground_force'] >= 0).all()
    with np.load(root / 'recording/all_scene_v6_20260920/traces.npz') as previous:
        for key in ('initial_q', 'initial_root', 'raw_action', 'joint_pos', 'target'):
            actual = current[key]
            expected = previous[key] if key.startswith('initial_') else previous[key][:3]
            result['comparison'][key] = float(np.max(np.abs(actual - expected)))
print(json.dumps(result, indent=2))
(output / 'verification.json').write_text(json.dumps(result, indent=2))
