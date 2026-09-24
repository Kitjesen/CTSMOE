"""Per-joint first-episode timing; requested and realized states stay distinct."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

P = Path(__file__).parent
NAMES = [f'{side}_{joint}_joint' for side in ('fr','fl','rr','rl') for joint in ('hip','thigh','calf')]
TOL = np.array([.25,.45,.55]*4)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})


def first(mask, dt, offset=0.):
    hit = np.flatnonzero(mask)
    return float(hit[0]*dt+offset) if len(hit) else None


rows = []
for directory in ('repeat96','request96','protected96','cost_probe'):
    m = json.loads((P/directory/'manifest.json').read_text())
    with np.load(P/directory/'traces.npz') as source:
        a = {k: source[k] for k in source.files}
    ids = [m['joint_names'].index(n) for n in NAMES]
    nominal = np.array(m['default_q'])[ids]
    for c in m['cases']:
        flat = c['terrain']=='flat'
        if flat and (c['policy']==0 or c['direction']!='forward'):
            continue
        if not flat and (c['terrain']!='pit' or c['speed']!=.6):
            continue
        n,i,dt = c['end_step']+1,c['id'],m['dt']
        q = a['joint_pos'][:n,i][:,ids]-nominal
        target = a['target'][:n,i][:,ids]-nominal
        request = .25*a['raw_action'][:n,i,:12]
        assert np.isfinite(q).all() and np.isfinite(request).all()
        start = 100 if c.get('mode')=='settle' else 0
        label = m['history'][c['policy']]['label'] if flat else c['policy']
        for j,name in enumerate(NAMES):
            row = dict(dataset=directory,policy=label,mode=c.get('mode','pit'),joint=name,
                       request_first_excess_s=first(abs(request[start:,j])>TOL[j],dt),
                       target_first_excess_s=first(abs(target[start:,j])>TOL[j],dt),
                       actual_first_excess_s=first(abs(q[start:,j])>TOL[j],dt,dt),
                       request_peak_offset_rad=float(abs(request[:,j]).max()),
                       target_peak_offset_rad=float(abs(target[:,j]).max()),
                       actual_peak_offset_rad=float(abs(q[:,j]).max()),
                       request_limited_gap_mean_rad=float(abs(request[:,j]-target[:,j]).mean()))
            if not flat:
                after = a['climb'][:n,i].astype(bool)
                row['after_climb_actual_peak_offset_rad'] = float(abs(q[after,j]).max()) if after.any() else None
                row['after_climb_request_peak_offset_rad'] = float(abs(request[after,j]).max()) if after.any() else None
            rows.append(row)
        if (flat and c['policy']==1 and c['mode']=='direct') or not flat:
            fig, axes = plt.subplots(4,3,figsize=(14,10),sharex=True)
            t = np.arange(n)*dt
            for j,ax in enumerate(axes.flat):
                ax.plot(t,request[:,j],color='#c23b35',label='限位前请求',lw=1)
                ax.plot(t,target[:,j],color='#d4922d',label='限位后目标',lw=1)
                ax.plot(t+dt,q[:,j],color='#1676a5',label='实际角度',lw=1.4)
                ax.axhspan(-TOL[j],TOL[j],color='gray',alpha=.1)
                ax.axhline(0,color='gray',lw=.5)
                if not flat and a['climb'][:n,i].any():
                    ax.axvline(first(a['climb'][:n,i],dt,dt),color='green',ls=':',label='检测到爬上')
                ax.set_title(NAMES[j])
                ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=8)
            for ax in axes[:,0]:
                ax.set_ylabel('相对默认角度（rad）')
            for ax in axes[-1]:
                ax.set_xlabel('时间（秒）')
            fig.suptitle(f'{directory} / {label}：请求、控制目标、实际角度\n请求和目标为步开始；实际角度为步结束（20 ms）；阴影为姿态容许区间')
            fig.tight_layout(rect=(0,0,1,.94))
            fig.savefig(P/f'joint_phase_{directory}_{label}.png',dpi=130)
            plt.close(fig)
(P/'joint_phase_summary.json').write_text(json.dumps(rows,indent=2))
fields = list(dict.fromkeys(k for row in rows for k in row))
with (P/'joint_phase_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print('Audited joint traces:',len(rows))
print(json.dumps([r for r in rows if r['dataset']=='protected96' and r['mode']=='direct' and 'student' not in r['policy'] and 'teacher' not in r['policy'] and 'hip' in r['joint']],indent=2))
