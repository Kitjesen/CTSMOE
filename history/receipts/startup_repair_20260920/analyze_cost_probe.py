import json
from pathlib import Path
import numpy as np

p=Path(__file__).parent/'cost_probe'
m=json.loads((p/'manifest.json').read_text())
f=np.load(p/'traces.npz')
tol=np.array([.25,.45,.55]*4)
base=m['body_names'].index('base_link')
rows=[]
for c in m['cases']:
    if c['terrain']!='pit':
        continue
    n=c['end_step']+1
    i=c['id']
    cost=np.maximum(np.abs(.25*f['raw_action'][:n,i,:12])/tol-1.,0.)**2
    cost=cost.mean(-1)*f['pit_posture_gate'][:n,i]
    approach=f['body_pos'][:n,i,base,0]<1.5
    row=dict(policy=c['policy'],speed=c['speed'],climb=c['climb'],stable=c['stable'],
        integrated_request_penalty=float((cost*m['dt']).sum()),
        approach_request_penalty=float((cost[approach]*m['dt']).sum()),
        after_climb_seen_penalty=float((cost[f['climb'][:n,i].astype(bool)]*m['dt']).sum()),
        peak_penalty_rate=float(cost.max()),active_seconds=float((cost>0).sum()*m['dt']),
        climb_event_bonus=5.,stable_success_bonus=10.)
    rows.append(row)
(p/'cost_summary.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
