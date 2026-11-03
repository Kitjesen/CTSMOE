import json
from pathlib import Path
import numpy as np

p=Path(__file__).parent
rows=[]
for directory in ('baseline','candidate96','repeat96'):
    m=json.loads((p/directory/'manifest.json').read_text())
    with np.load(p/directory/'traces.npz') as f:
        action,velocity=f['raw_action'],f['joint_vel']
    ids=[m['joint_names'].index(side+'_foot_joint') for side in ('fr','fl','rr','rl')]
    for c in m['cases']:
        if c['direction']!='forward' or c['mode']!='direct':
            continue
        a=action[:50,c['id'],12:]*5.
        v=velocity[:50,c['id']][:,ids]
        rows.append(dict(policy=m['history'][c['policy']]['label'],dataset=directory,
            request_abs_max_rad_s=float(abs(a).max()),
            first_request_rad_s=a[0].tolist(),actual_abs_max_rad_s=float(abs(v).max()),
            requested_over30_fraction=float((abs(a)>30.).mean())))
(p/'wheel_start_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
