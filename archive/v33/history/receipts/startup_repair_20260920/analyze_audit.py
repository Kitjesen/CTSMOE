"""Whole-episode and startup metrics; early physical failures never count as pass."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
a=parser.parse_args()
m=json.loads((a.directory/'manifest.json').read_text())
with np.load(a.directory/'traces.npz') as f:
    d={k:f[k] for k in ('joint_pos','root_clearance','velocity','angular_velocity','command')}
legs=[j for j,n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
default=np.array(m['default_q'])[legs]
rows=[]
for c in m['cases']:
    i=c['id']
    end=c['end_step']+1
    start=100 if c['mode']=='settle' else 0
    q=d['joint_pos'][:end,i][:,legs]-default
    rms=np.sqrt(np.mean(q*q,axis=1))
    h=d['root_clearance'][:end,i]
    velocity=d['velocity'][:end,i,:2]
    cmd=d['command'][:end,i]
    first=rms[start:min(start+50,end)]
    last=slice(max(start,end-50),end)
    row=dict(**c,label=m['history'][c['policy']]['label'],
        first_rms=float(first.mean()) if len(first) else None,
        max_rms=float(rms.max()),height_min=float(h.min()),height_max=float(h.max()),
        max_xy_speed=float(np.linalg.norm(velocity,axis=-1).max()),
        last_xy_error=float(np.linalg.norm(velocity[last]-cmd[last,:2],axis=-1).mean()),
        last_yaw_error=float(np.abs(d['angular_velocity'][last,i,2]-cmd[last,2]).mean()))
    row['early_failure']=any(term not in ('time_out','stair_route_exit','budget') for term in c['terms'])
    row['startup_pass']=bool(not c['physical'] and not row['early_failure'] and len(first)==50 and row['first_rms']<.25
        and row['max_rms']<.35
        and row['height_min']>.34 and row['height_max']<.60
        and row['max_xy_speed']<max(.5,np.linalg.norm(c['velocity'][:2])+.5))
    # An immobile policy can satisfy posture criteria; require command tracking
    # independently, over the final second, before accepting a repair.
    row['tracking_pass']=bool(end-start>=100 and row['last_xy_error']<.15
                              and row['last_yaw_error']<.2)
    row['combined_pass']=row['startup_pass'] and row['tracking_pass']
    rows.append(row)
(a.directory/'summary.json').write_text(json.dumps(rows,indent=2))
for label in dict.fromkeys(r['label'] for r in rows):
    for mode in ('direct','settle'):
        rr=[r for r in rows if r['label']==label and r['mode']==mode]
        print(label,mode,'pass',sum(r['startup_pass'] for r in rr),'/',len(rr),
              'max_h',round(max(r['height_max'] for r in rr),3),
              'first_rms',[(r['direction'],round(r['first_rms'],3) if r['first_rms'] is not None else None) for r in rr])
