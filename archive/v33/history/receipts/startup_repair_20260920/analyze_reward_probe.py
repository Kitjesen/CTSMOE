"""Compare actual dt-weighted returns, activation gates and physical response."""
import json
from pathlib import Path
import numpy as np

p=Path(__file__).parent
d=p/'reward_probe'
m=json.loads((d/'manifest.json').read_text())
with np.load(d/'traces.npz') as f:
    a={k:f[k] for k in f.files}
names=m['reward_names']
np.testing.assert_allclose(a['weighted_reward'].sum(-1),a['reward_total'],atol=1e-5)
np.testing.assert_allclose(a['command'],a['reward_command'],atol=1e-6)
assert np.isfinite(a['weighted_reward']).all()
# Reward-only instrumentation must not materially change the measured behavior.
with np.load(p/'transition_probe'/'traces.npz') as f:
    parity={k:float(np.max(np.abs(a[k]-f[k]))) for k in ('joint_pos','raw_action','root_clearance','command')}
rows=[]
for c in m['cases']:
    n,i=c['end_step']+1,c['id']
    start=100 if c['mode']=='settle' else 0
    stages={'start_1s':(start,min(start+50,n)), 'move_2s':(start,min(start+100,n)),
            'stop_last_1s':(max(start+100,n-50),n)}
    for stage,(lo,hi) in stages.items():
        costs=a['weighted_reward'][lo:hi,i].sum(0)
        rows.append(dict(label=m['history'][c['policy']]['label'],mode=c['mode'],direction=c['direction'],
            stage=stage,total=float(costs.sum()),contributions=dict(zip(names,map(float,costs))),
            posture_gate_min=a['posture_gates'][lo:hi,i].min(0).tolist(),
            posture_gate_max=a['posture_gates'][lo:hi,i].max(0).tolist(),
            height_min=float(a['reward_height'][lo:hi,i].min()),
            height_max=float(a['reward_height'][lo:hi,i].max())))
(d/'analysis.json').write_text(json.dumps(dict(parity=parity,weights=m['reward_weights'],rows=rows),indent=2))
print('TRAJECTORY_PARITY',parity)
for row in rows:
    if row['direction']=='forward' and row['mode']=='direct' and 'student' in row['label'] and row['stage']=='move_2s':
        print(json.dumps(row,indent=2))
