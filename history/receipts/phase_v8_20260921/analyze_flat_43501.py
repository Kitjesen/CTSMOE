"""Summarize matched first episodes; preserve failures and expose sample counts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
m = json.loads((a.directory/'manifest.json').read_text(encoding='utf-8'))
if m.get('smoke_only', False):
    raise ValueError('Smoke traces are not a behavior evaluation')
version = 'V8'
with np.load(a.directory/'traces.npz') as f:
    d = {k:f[k] for k in f.files}
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                     'axes.unicode_minus':False})
names = dict(flat='平地',stairs_up='上楼梯',stairs_down='下楼梯',slope_up='上坡',
             slope_down='下坡',wave='波浪',rough_slope='粗糙坡',obstacles='障碍块',pit='高台')
legs = [i for i,n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
wheels = [i for i in range(len(m['joint_names'])) if i not in legs]
base_id = m['body_names'].index('base_link')
default = np.array(m['default_q'])
rows = []
for c in m['cases']:
    i,n = c['id'],c['end_step']+1
    for key in ('joint_pos','root_clearance','torque_peak','speed_peak','velocity','command'):
        assert np.isfinite(d[key][:n,i]).all(),(key,i)
    q = d['joint_pos'][:n,i][:,legs]-default[legs]
    rms = np.sqrt(np.mean(q*q,axis=-1))
    start = 100 if c['mode']=='settle' else 0
    first = rms[start:min(n,start+50)]
    last = slice(max(start,n-50),n)
    xy = np.linalg.norm(d['velocity'][:n,i,:2]-d['command'][:n,i,:2],axis=-1)
    yaw = np.abs(d['angular_velocity'][:n,i,2]-d['command'][:n,i,2])
    torque = d['torque_peak'][:n,i]
    speed = d['speed_peak'][:n,i]
    tilt = np.degrees(np.arccos(np.clip(-d['gravity'][:n,i,2],-1,1)))
    physical = c['physical']
    early = any(t not in ('time_out','stair_route_exit','budget') for t in c['terms'])
    row = dict(**c,label=m['history'][c['policy']]['label'],
        first_rms=float(first.mean()) if len(first) else None,
        max_rms=float(rms.max()),height_min=float(d['root_clearance'][:n,i].min()),
        height_max=float(d['root_clearance'][:n,i].max()),
        xy_mae=float(xy[start:].mean()) if n>start else None,
        last_xy_error=float(xy[last].mean()) if n>start else None,
        last_yaw_error=float(yaw[last].mean()) if n>start else None,
        tilt_max_deg=float(tilt.max()),
        leg_torque_peak_nm=float(torque[:,legs].max()),wheel_torque_peak_nm=float(torque[:,wheels].max()),
        leg_speed_peak_rad_s=float(speed[:,legs].max()),wheel_speed_peak_rad_s=float(speed[:,wheels].max()),
        hard_position_excess_rad=float(d['hard_excess'][:n,i][:,legs].max()),
        leg_saturation_fraction=float(d['saturation'][:n,i][:,legs].mean()),
        base_contact_peak_n=float(d['base_contact'][:n,i].max()))
    if c['terrain']=='flat':
        row['pose_pass']=bool(not physical and not early and len(first)==50 and row['first_rms']<.25
            and row['max_rms']<.35 and row['height_min']>.34 and row['height_max']<.60
            and np.linalg.norm(d['velocity'][:n,i,:2],axis=-1).max()<max(.5,np.linalg.norm(c['velocity'][:2])+.5))
        row['tracking_pass']=bool(n-start>=100 and row['last_xy_error']<.15 and row['last_yaw_error']<.2)
        row['pass']=row['pose_pass'] and row['tracking_pass']
    elif c['terrain']=='pit':
        row['pass']=bool(c['climb'] and c['stable'] and not physical)
    else:
        row['pass']=bool(c['passed_4m'] and not physical)
    rows.append(row)

(a.directory/'analysis.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
show=[r for r in rows if r['mode']=='direct' and r['direction'] in ('forward','reverse','left','forward_turn')]
fig,axes=plt.subplots(4,3,figsize=(13,11),sharex=True)
labels=dict(forward='前进',reverse='后退',left='侧移',forward_turn='边走边转')
for rr,aa in zip(show,axes):
    i,n=rr['id'],rr['end_step']+1
    t=(np.arange(n)+1)*m['dt']
    q=d['joint_pos'][:n,i][:,legs]-default[legs]
    aa[0].plot(t,np.sqrt((q*q).mean(-1)),label='腿姿态偏差RMS')
    aa[1].plot(t,d['root_clearance'][:n,i],label='实际高度')
    aa[1].axhline(.46,color='gray',ls='--',label='默认高度0.46m')
    for j,label in enumerate(('前向','侧向')):
        aa[2].plot(t,d['velocity'][:n,i,j],label=label+'实际')
        aa[2].plot(t,d['command'][:n,i,j],ls='--',alpha=.6,label=label+'指令')
    aa[0].set_ylabel(labels[rr['direction']])
    for ax in aa:
        ax.grid(alpha=.25)
for ax,title in zip(axes[0],('相对默认角 RMS (rad)','机身高度 (m)','速度 (m/s)')):
    ax.set_title(title)
    ax.legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel('起步后时间 (s)')
fig.suptitle('V8学生43,501轮：默认站姿直接起步；无噪声/推扰，固定10ms延迟')
fig.tight_layout()
fig.savefig(a.directory.parent/'flat_timeseries.png',dpi=140)
plt.close(fig)
print(json.dumps(dict(count=len(rows),pose_pass=sum(r['pose_pass'] for r in rows),tracking_pass=sum(r['tracking_pass'] for r in rows),passed=sum(r['pass'] for r in rows),physical=sum(r['physical'] for r in rows),shown=[{k:r[k] for k in ('direction','first_rms','max_rms','height_min','height_max','last_xy_error','last_yaw_error','tilt_max_deg','pass')} for r in show],failures=[{k:r[k] for k in ('direction','mode','pose_pass','tracking_pass','last_xy_error','last_yaw_error','max_rms')} for r in rows if not r['pass']]),indent=2))
