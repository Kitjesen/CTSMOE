"""Summarize identical flat-ground first episodes across saved policies."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=Path(__file__).parent
m=json.loads((p/'manifest.json').read_text())
with np.load(p/'traces.npz') as archive:
    d={k:archive[k] for k in ('joint_pos','target','root_clearance','velocity')}
legs=[i for i,n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
default=np.array(m['default_q'])[legs]
rows=[]
for case in m['cases']:
    i=case['id']
    end=case['end_step']+1
    q=d['joint_pos'][:end,i][:,legs]-default
    target=d['target'][:end,i][:,legs]-default
    rms=np.sqrt(np.mean(q*q,axis=1))
    height=d['root_clearance'][:end,i]
    start=100 if case['mode']=='settle' else 0
    first=slice(start,min(start+50,end))
    row=dict(checkpoint=m['history'][case['policy']]['label'],mode=case['mode'],
        seconds=case['seconds'],physical=case['physical'],
        first_second_rms=float(rms[first].mean()) if end>start else None,
        first_second_peak_joint=float(np.abs(q[first]).max()) if end>start else None,
        first_second_target_peak=float(np.abs(target[first]).max()) if end>start else None,
        first_second_min_height=float(height[first].min()) if end>start else None,
        final_second_rms=float(rms[-50:].mean()),final_second_height=float(height[-50:].mean()),
        final_second_vx=float(d['velocity'][max(0,end-50):end,i,0].mean()),
        peak_vx=float(d['velocity'][:end,i,0].max()),
        first_target_rms=float(np.sqrt(np.mean(target[0]**2))),
        path=m['history'][case['policy']]['path'])
    rows.append(row)
(p/'summary.json').write_text(json.dumps(rows,indent=2))
with (p/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
fig,axes=plt.subplots(3,1,figsize=(13,11),sharex=True)
for mode in ('direct','settle','ramp'):
    rr=[r for r in rows if r['mode']==mode]
    x=[int(r['checkpoint']) for r in rr]
    axes[0].plot(x,[r['first_second_rms'] for r in rr],'.-',label=mode)
    axes[1].plot(x,[r['final_second_rms'] for r in rr],'.-',label=mode)
rr=[r for r in rows if r['mode']=='stand']
axes[2].plot([int(r['checkpoint']) for r in rr],[r['final_second_height'] for r in rr],'.-',label='stand height')
axes[2].axhline(.46,color='gray',linestyle='--',label='nominal 0.46 m')
for ax in axes:
    for x,label in ((14535,'height off / stairs80'),(30000,'mixed pit'),(41647,'surface posture'),(41751,'strong stance')):
        ax.axvline(x,color='gray',alpha=.3)
    ax.grid(alpha=.2)
    ax.legend()
axes[0].set_ylabel('First moving second joint RMS (rad)')
axes[1].set_ylabel('Last second joint RMS (rad)')
axes[2].set_ylabel('Standing body clearance (m)')
axes[2].set_xlabel('Completed training updates')
fig.suptitle('Matched historical student policies: flat ground, +0.6 m/s\nCurrent identical physics / 10 ms delay / no noise; one first episode per mode')
fig.tight_layout()
fig.savefig(p/'history.png',dpi=150)
for n in sorted(set(r['checkpoint'] for r in rows),key=int):
    rr={r['mode']:r for r in rows if r['checkpoint']==n}
    print(n,'direct/settle/ramp first RMS',*[round(rr[k]['first_second_rms'] or 0,3) for k in ('direct','settle','ramp')],
          'stand h/q',round(rr['stand']['final_second_height'],3),round(rr['stand']['final_second_rms'],3),
          'move q',round(rr['direct']['final_second_rms'],3),'fail',sum(r['physical'] for r in rr.values()))
