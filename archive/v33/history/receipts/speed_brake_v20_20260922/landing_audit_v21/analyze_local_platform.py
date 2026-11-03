"""Separate normal stair ascent from rebound after actual platform entry."""
import json
from pathlib import Path
import sys

import numpy as np

folder = Path(sys.argv[1])
manifest = json.loads((folder/'manifest.json').read_text())
arrays = np.load(folder/'traces.npz')
rows = []
for case in manifest['cases']:
    i, n = case['id'], case['end_step']+1
    valid = arrays['active'][:n, i].astype(bool)
    if case.get('reset_metadata', {}).get('reset_occurred_after_capture', False):
        valid[-1] = False
    phase = arrays['landing_phase'][:n, i]
    vz = arrays['world_velocity'][:n, i, 2]
    q = arrays['joint_pos'][:n, i, :12] - np.asarray(manifest['default_q'])[:12]
    by_phase = {}
    for code, label in enumerate(('climb', 'last_step', 'platform_unstable', 'platform_supported', 'support_dwell_complete')):
        mask = valid & (phase == code)
        if mask.any():
            by_phase[label] = dict(steps=int(mask.sum()), first_s=float(np.flatnonzero(mask)[0]*manifest['dt']),
                positive_vz_peak=float(np.maximum(vz[mask], 0).max()), negative_vz_min=float(np.minimum(vz[mask], 0).min()),
                q_rms_peak=float(np.sqrt((q[mask]**2).mean(-1)).max()))
    row = {key: case[key] for key in ('id','policy','terrain','level','velocity','landing_mode','physical','end_reason')}
    row.update(phases=by_phase, post_platform_positive_vz_peak=float(np.maximum(vz[valid & (phase>=2)], 0).max()) if (valid & (phase>=2)).any() else None,
               support_dwell_s=float(arrays['local_hold_steps'][:n, i][valid].max()*manifest['dt']),
               last_velocity=arrays['velocity'][:n, i][valid][-1].tolist(),
               last_yaw=float(arrays['angular_velocity'][:n, i, 2][valid][-1]))
    rows.append(row)
(folder/'LOCAL_PLATFORM_METRICS.json').write_text(json.dumps(dict(cases=rows, random_success_rate=False,
    note='Two-second geometric contact support; velocity/posture reported independently, not assumed stationary.'),indent=2))
print('ANALYZED', len(rows))
