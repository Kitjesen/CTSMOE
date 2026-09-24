"""Measure obstacle approach timing; reward gates are not policy control switches."""
import json
from pathlib import Path

import numpy as np

b = Path(__file__).parent
rows = []
for folder, policies in (('all_scene_v6', (0, 1)), ('all_scene_v7', (1, 2))):
    m = json.loads((b / folder / 'manifest.json').read_text(encoding='utf-8'))
    with np.load(b / folder / 'traces.npz') as z:
        positions = z['body_pos']
        q = z['joint_pos']
        gravity = z['gravity']
        requested = z['requested']
        stop = z['platform_stop'] if 'platform_stop' in z.files else None
    wheel_ids = [i for i, name in enumerate(m['body_names']) if name.lower().endswith('_foot_link')]
    leg_ids = [i for i, name in enumerate(m['joint_names']) if not name.endswith('_foot_joint')]
    for policy in policies:
        c = next(c for c in m['cases'] if c['policy']==policy and c['terrain']=='pit'
                 and c['level']==21 and c['velocity']==[.6, 0., 0.])
        i, n = c['id'], c['end_step']+1
        t = (np.arange(n)+1)*m['dt']
        gap = 1.5-(positions[:n, i][:, wheel_ids, 0].max(-1)+.093)
        # Same approach branch as pit_posture.phase_weight. Ignore the later
        # platform branch here: this diagnostic ends at the first wall arrival.
        gate = np.clip((gap-.25)/.5, 0., 1.)
        rms = np.sqrt(np.mean((q[:n, i][:, leg_ids]-np.array(m['default_q'])[leg_ids])**2, axis=1))
        tilt = np.degrees(np.arccos(np.clip(-gravity[:n, i, 2], -1., 1.)))
        arrival = np.flatnonzero(gap<=.01)
        end = int(arrival[0]) if len(arrival) else n-1
        free = np.flatnonzero(gate[:end+1]==0.)
        first_free = int(free[0]) if len(free) else None
        request = requested[:n, i]-np.array(m['default_q'])[m['request_joint_ids']]
        rows.append(dict(label=m['history'][policy]['label'], first_wall_arrival_s=float(t[end]),
            posture_gate_zero_s=float(t[first_free]) if first_free is not None else None,
            available_time_after_release_s=float(t[end]-t[first_free]) if first_free is not None else None,
            max_tilt_before_wall_deg=float(tilt[:end+1].max()),
            max_actual_rms_before_wall_rad=float(rms[:end+1].max()),
            max_request_offset_after_release_rad=float(np.abs(request[first_free:end+1]).max()) if first_free is not None else None,
            gate_at_first_wall=float(gate[end]), final_time_s=float(t[-1]), terms=c['terms'],
            stop_on_platform_ever=bool(stop[:n,i].any()) if stop is not None else None))
(b / 'all_scene_v7/pit_transition_timing.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
print(json.dumps(rows, indent=2))
