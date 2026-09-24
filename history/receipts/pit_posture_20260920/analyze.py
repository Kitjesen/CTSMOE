import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=Path(__file__).parent
d=json.loads((p/'frozen/manifest.json').read_text())
with np.load(p/'frozen/traces.npz') as archive:
    z={key:archive[key] for key in archive.files}
names=d['joint_names']
default=np.asarray(d['default_joint_pos'])
case=d['cases'][1]
n=case['end_step']+1
t=(np.arange(n)+1)*.02
q=z['joint_pos'][:n,1]
target=z['joint_target'][:n,1]
fig,axes=plt.subplots(4,3,figsize=(14,10),sharex=True)
result={}
for i,side in enumerate(('fr','fl','rr','rl')):
    for j,kind in enumerate(('hip','thigh','calf')):
        name=f'{side}_{kind}_joint'
        k=names.index(name)
        ax=axes[i,j]
        ax.plot(t,q[:,k],label='measured')
        ax.plot(t,target[:,k],alpha=.7,label='policy target')
        ax.axhline(default[k],color='black',ls='--',label='nominal stance')
        ax.set_title(name)
        ax.set_ylabel('rad')
        ax.grid(alpha=.3)
        result[name]=dict(default=float(default[k]),final_actual=float(q[-1,k]),final_target=float(target[-1,k]),
            last_second_target_tracking_mae=float(abs(q[-50:,k]-target[-50:,k]).mean()),
            last_second_mean_torque=float(abs(z['torque'][n-50:n,1,k]).mean()))
axes[0,0].legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel('seconds')
fig.suptitle('Frozen student 41001: measured joints follow bad policy targets after climbing')
fig.tight_layout()
fig.savefig(p/'joint_targets_and_angles.png',dpi=140)
plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
for side in ('fr','fl','rr','rl'):
    k=names.index(side+'_calf_joint')
    axes[0].plot(t,z['joint_vel'][:n,1,k],label=side+' calf')
axes[0].set_ylabel('rad/s')
axes[0].legend(ncol=4)
axes[1].plot(t,z['height'][:n,1],label='height above active support plane')
axes[1].axhline(.46,color='black',ls='--',label='stance target')
axes[1].set_ylabel('m')
axes[1].legend()
axes[2].plot(t,z['posture_weight'][:n,1],label='stance shaping activation')
axes[2].plot(t,z['posture_ok'][:n,1],label='stance acceptable',alpha=.6)
axes[2].set_ylabel('0..1')
axes[2].set_xlabel('seconds')
axes[2].legend()
for ax in axes:
    ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(p/'phase_and_velocity.png',dpi=140)
plt.close(fig)
with (p/'joint_timeseries.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f)
    w.writerow(['case','command_speed_mps','time_s','joint','angle_rad','target_rad','velocity_rad_s','applied_torque_nm'])
    for c in d['cases']:
        for step in range(c['end_step']+1):
            for k,name in enumerate(names):
                w.writerow([c['id'],c['speed'],(step+1)*.02,name,
                    *[float(z[key][step,c['id'],k]) for key in ('joint_pos','joint_target','joint_vel','torque')]])
(p/'target_tracking.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
