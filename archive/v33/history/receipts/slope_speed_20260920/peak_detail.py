"""Relate measured wheel speed peaks to targets and applied torque."""
import json
from pathlib import Path

import numpy as np

base = Path(__file__).parent
out = []
for mode in ('controlled', 'training'):
    folder = base/mode
    d = json.loads((folder/'results.json').read_text())
    z = np.load(folder/'traces.npz')
    control, physics = z['control'], z['physics']
    fields, offset = {}, 0
    for name, size in d['fields']:
        fields[name] = slice(offset, offset+size)
        offset += size
    for i, case in enumerate(d['cases']):
        if case['level'] != 9 or not case['causes']['speed']:
            continue
        p = physics[:(case['step']+1)*4, i]
        step, joint = np.unravel_index(np.abs(p[:, :16]).argmax(), p[:, :16].shape)
        cs = min(step//4, len(control)-1)
        row = dict(mode=mode, index=i, speed=case['speed'],
                   joint=d['joints'][joint], time=(step+1)*.005,
                   before=float(p[max(0, step-1), joint]), peak=float(p[step, joint]),
                   torque=float(p[step, 16+joint]),
                   target_previous_control=float(control[max(0, cs-1), i, fields['velocity_target']][joint]),
                   target_control=float(control[cs, i, fields['velocity_target']][joint]))
        out.append(row)
(base/'peak_detail.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
