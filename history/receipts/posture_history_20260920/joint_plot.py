"""Plot policy-issued targets alongside measured startup posture."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=Path(__file__).parent
m=json.loads((p/'manifest.json').read_text())
with np.load(p/'traces.npz') as archive:
    d={k:archive[k] for k in ('joint_pos','target','root_clearance','velocity','command')}
legs=[i for i,n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
default=np.array(m['default_q'])
fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
details=[]
for checkpoint,mode in [('31001','direct'),('33001','direct'),('42001','direct'),('42001','settle'),('42001','ramp')]:
    case=next(c for c in m['cases'] if m['history'][c['policy']]['label']==checkpoint and c['mode']==mode)
    i=case['id']
    end=min(case['end_step']+1,200)
    t=(np.arange(end)+1)*.02
    q=d['joint_pos'][:end,i]-default
    target=d['target'][:end,i]-default
    label=checkpoint+' '+mode
    axes[0].plot(t,np.sqrt(np.mean(q[:,legs]**2,axis=1)),label=label)
    axes[1].plot(t,np.sqrt(np.mean(target[:,legs]**2,axis=1)),label=label)
    axes[2].plot(t,d['root_clearance'][:end,i],label=label)
    axes[3].plot(t,d['velocity'][:end,i,0],label=label)
    if mode=='direct':
        ti,ji=np.unravel_index(np.abs(q[:50,legs]).argmax(),q[:50,legs].shape)
        j=legs[ji]
        details.append(dict(checkpoint=checkpoint,time_s=float(t[ti]),joint=m['joint_names'][j],
            default_rad=float(default[j]),actual_rad=float(d['joint_pos'][ti,i,j]),
            target_rad=float(d['target'][ti,i,j]),deviation_rad=float(q[ti,j])))
axes[0].set_ylabel('Actual leg RMS (rad)')
axes[1].set_ylabel('Target leg RMS (rad)')
axes[2].set_ylabel('Body height (m)')
axes[2].axhline(.46,color='gray',linestyle='--')
axes[3].set_ylabel('Forward velocity (m/s)')
axes[3].axhline(.6,color='gray',linestyle='--')
axes[3].set_xlabel('Time after reset (s)')
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(fontsize=8,ncol=3)
fig.suptitle('Flat-ground startup: actual joints follow large policy-issued targets\nSame current physics; raw first episodes, no imposed pose correction')
fig.tight_layout()
fig.savefig(p/'startup_timeseries.png',dpi=150)
(p/'joint_examples.json').write_text(json.dumps(details,indent=2))
print(json.dumps(details))
