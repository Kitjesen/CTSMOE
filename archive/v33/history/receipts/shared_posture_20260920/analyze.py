"""First-episode, phase-specific posture audit; no surviving-episode pooling."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = Path(__file__).parent
m = json.loads((p/'manifest.json').read_text())
with np.load(p/'traces.npz') as archive:
    z = {k: archive[k] for k in archive.files}
names = m['joints']
leg = [i for i,n in enumerate(names) if not n.endswith('_foot_joint')]
nominal = np.asarray(m['default_q'])[leg]
tol = np.asarray([{'hip': .25, 'thigh': .45, 'calf': .55}[names[i].split('_')[1]] for i in leg])
rows = []
for i,c in enumerate(m['cases']):
    n = c['end_step']+1
    t = (np.arange(n)+1)*.02
    q = z['q'][:n,i][:,leg]
    target = z['target'][:n,i][:,leg]
    error = q-nominal
    phases = {'first_second': (0,1), 'early': (1,2), 'middle': (3,5), 'final': (10,12)}
    for phase,(start,end) in phases.items():
        mask = (t>start)&(t<=end)
        if not mask.any():
            continue
        rows.append(dict(case=i,**c,phase=phase,observed_seconds=round(float(mask.sum()*.02),2),
            joint_rms_rad=float(np.sqrt((error[mask]**2).mean())),
            target_rms_rad=float(np.sqrt(((target[mask]-nominal)**2).mean())),
            tracking_mae_rad=float(abs(target[mask]-q[mask]).mean()),
            stance_band_fraction=float((abs(error[mask])<=tol).all(-1).mean()),
            root_clearance_m=float(z['root_clearance'][:n,i][mask].mean()),
            vx_mps=float(z['velocity'][:n,i,0][mask].mean()),
            four_contacts_fraction=float((z['wheel_force'][:n,i][mask]>5).all(-1).mean()),
            max_leg_deviation_rad=float(abs(error[mask]).max()),
            x_end_m=float(z['xyz'][n-1,i,0])))
(p/'summary.json').write_text(json.dumps(rows,indent=2))
with (p/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
fig,axes = plt.subplots(3,3,figsize=(16,11),sharex=True,sharey=True)
for ax,terrain in zip(axes.flat,dict.fromkeys(c['terrain'] for c in m['cases'])):
    for i,c in enumerate(m['cases']):
        if c['terrain']!=terrain or c['repeat']!=0:
            continue
        n=c['end_step']+1
        error=z['q'][:n,i][:,leg]-nominal
        ax.plot((np.arange(n)+1)*.02,np.sqrt((error**2).mean(-1)),
                label=f"L{c['level']} {c['mode']}",alpha=.8)
    ax.set_title(terrain)
    ax.set_ylabel('12 leg joints RMS from nominal (rad)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=7,ncol=2)
for ax in axes[-1]:
    ax.set_xlabel('seconds')
fig.suptitle(f"Shared student {m['completed_updates']}: first episodes, nominal physics, no observation noise")
fig.tight_layout()
fig.savefig(p/'all_terrain_posture.png',dpi=150)
plt.close(fig)
fig,axes=plt.subplots(4,3,figsize=(15,11),sharex=True)
case_id=next(i for i,c in enumerate(m['cases']) if c['terrain']=='flat' and c['mode']=='start_stop' and c['repeat']==0)
n=m['cases'][case_id]['end_step']+1
t=(np.arange(n)+1)*.02
for ax,j in zip(axes.flat,leg):
    ax.plot(t,z['q'][:n,case_id,j],label='actual')
    ax.plot(t,z['target'][:n,case_id,j],label='policy target',alpha=.65)
    ax.axhline(m['default_q'][j],color='black',ls='--',label='nominal')
    ax.axvspan(2,8,color='green',alpha=.06)
    ax.set_title(names[j])
    ax.set_ylabel('rad')
    ax.grid(alpha=.2)
axes[0,0].legend()
for ax in axes[-1]:
    ax.set_xlabel('seconds')
fig.suptitle('Flat: stand 0-2s, move +0.6 m/s 2-8s, stop 8-12s')
fig.tight_layout()
fig.savefig(p/'flat_joint_timeseries.png',dpi=150)
plt.close(fig)
fig,axes=plt.subplots(4,3,figsize=(15,11),sharex=True)
case_id=next(i for i,c in enumerate(m['cases']) if c['terrain']=='flat' and c['mode']=='move' and c['repeat']==0)
n=m['cases'][case_id]['end_step']+1
t=(np.arange(n)+1)*.02
for ax,j in zip(axes.flat,leg):
    ax.plot(t,z['q'][:n,case_id,j],label='actual')
    ax.plot(t,z['target'][:n,case_id,j],label='policy target',alpha=.7)
    ax.axhline(m['default_q'][j],color='black',ls='--',label='nominal')
    ax.set_title(names[j])
    ax.set_ylabel('rad')
    ax.grid(alpha=.2)
axes[0,0].legend()
for ax in axes[-1]:
    ax.set_xlabel('seconds')
fig.suptitle('Flat direct +0.6 m/s from reset: large startup targets, then recovery')
fig.tight_layout()
fig.savefig(p/'flat_direct_start_timeseries.png',dpi=150)
plt.close(fig)
for terrain in dict.fromkeys(c['terrain'] for c in m['cases']):
    selected=[r for r in rows if r['terrain']==terrain and r['phase']=='early']
    print(terrain, 'early RMS',round(np.mean([r['joint_rms_rad'] for r in selected]),3),
          'physical',sum(c['physical'] for c in m['cases'] if c['terrain']==terrain))
print('FLAT',json.dumps([r for r in rows if r['terrain']=='flat' and r['repeat']==0],indent=2))
