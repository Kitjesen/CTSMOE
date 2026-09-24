"""Summarize matched first episodes; preserve failures and expose sample counts."""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
m = json.loads((a.directory/'manifest.json').read_text(encoding='utf-8'))
if m.get('smoke_only', False):
    raise ValueError('Smoke traces are not a behavior evaluation')
version='V10'
with np.load(a.directory/'traces.npz') as f:
    d = {k:f[k] for k in f.files}
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
        first_peak_rms=float(first.max()) if len(first) else None,
        first_peak_joint_rad=float(np.abs(q[start:min(n,start+50)]).max()) if len(first) else None,
        first_peak_time_s=float((np.argmax(first)+1)*m['dt']) if len(first) else None,
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
        row['pose_pass']=bool(not physical and not early and len(first)==50 and row['first_peak_rms']<.25 and row['first_peak_joint_rad']<.60
            and row['max_rms']<.35 and row['height_min']>.34 and row['height_max']<.60
            and np.linalg.norm(d['velocity'][:n,i,:2],axis=-1).max()<max(.5,np.linalg.norm(c['velocity'][:2])+.5))
        row['tracking_pass']=bool(n-start>=100 and row['last_xy_error']<.15 and row['last_yaw_error']<.2)
        row['pass']=row['pose_pass'] and row['tracking_pass']
    elif c['terrain']=='pit':
        row['pass']=bool(c['climb'] and c['stable'] and not physical)
    else:
        row['pass']=bool(c['passed_4m'] and not physical)
    row['last_height']=float(d['root_clearance'][last,i].mean())
    row['last_rms']=float(rms[last].mean())
    if c['terrain'].startswith('stairs'):
        quat=d['body_quat'][:n,i,base_id]
        vel=d['velocity'][:n,i]
        vector=quat[:,1:]
        world=vel+2*(quat[:,:1]*np.cross(vector,vel)+np.cross(vector,np.cross(vector,vel)))
        row['end_vz_peak']=float(world[-100:,2].max())
        row['end_phase_stable_s']=float(d['phase_stable_s'][n-1,i])
        row['end_phase_weight']=float(d['phase_weight'][n-1,i])
    rows.append(row)

groups=[]
for policy in range(len(m['history'])):
    for terrain in names:
        rr=[r for r in rows if r['policy']==policy and r['terrain']==terrain]
        groups.append(dict(policy=policy,terrain=terrain,count=len(rr),passed=sum(r['pass'] for r in rr),
            straight=sum(r.get('straight_4m',False) for r in rr),physical=sum(r['physical'] for r in rr),
            mean_xy_mae=float(np.mean([r['xy_mae'] for r in rr if r['xy_mae'] is not None])),
            peak_leg_torque=max(r['leg_torque_peak_nm'] for r in rr),
            peak_wheel_torque=max(r['wheel_torque_peak_nm'] for r in rr),
            peak_leg_speed=max(r['leg_speed_peak_rad_s'] for r in rr),
            peak_wheel_speed=max(r['wheel_speed_peak_rad_s'] for r in rr),
            max_hard_excess=max(r['hard_position_excess_rad'] for r in rr)))
(a.directory/'analysis.json').write_text(json.dumps(dict(cases=rows,groups=groups),indent=2),encoding='utf-8')


forward=[r for r in rows if r['terrain']=='flat' and r['mode']=='direct' and r['direction']=='forward']
ids=[r['id'] for r in forward]
np.savez_compressed(a.directory/'forward_traces.npz',ids=np.asarray(ids),
    **{k:d[k][:300,ids] for k in ('joint_pos','target','requested','root_clearance','velocity','command')})
summary=dict(forward=[{k:r[k] for k in ('label','first_rms','first_peak_rms','first_peak_joint_rad','first_peak_time_s','max_rms','height_min','height_max','last_height','last_rms','last_xy_error','last_yaw_error','pass')} for r in forward],
    terrains=[dict(label=m['history'][g['policy']]['label'],terrain=g['terrain'],passed=g['passed'],count=g['count'],physical=g['physical'],straight=g['straight']) for g in groups],
    pit=[{k:r[k] for k in ('label','level','velocity','climb','stable','physical')} for r in rows if r['terrain']=='pit'],
    flat_failures=[{k:r[k] for k in ('label','direction','mode','pose_pass','tracking_pass','last_xy_error','last_yaw_error')} for r in rows if r['terrain']=='flat' and not r['pass']])
(a.directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(dict(forward=summary['forward'],terrains=summary['terrains']),indent=2))

joint_rows=[]
requests=m['request_joint_ids']
for c in m['cases']:
    if c['terrain']!='flat':
        continue
    start=100 if c['mode']=='settle' else 0
    n=min(c['end_step']+1,start+50)
    if n<=start:
        continue
    for j in legs:
        actual=d['joint_pos'][start:n,c['id'],j]-default[j]
        target=d['target'][start:n,c['id'],j]-default[j]
        requested=d['requested'][start:n,c['id'],requests.index(j)]-default[j]
        k=int(np.argmax(abs(actual)))
        target_delta=np.diff(d['target'][max(0,start-1):n,c['id'],j])
        joint_rows.append(dict(label=m['history'][c['policy']]['label'],case=c['id'],
            direction=c['direction'],mode=c['mode'],joint=m['joint_names'][j],
            actual_peak_rad=float(abs(actual).max()),target_peak_rad=float(abs(target).max()),
            request_peak_rad=float(abs(requested).max()),peak_time_s=float((k+1)*m['dt']),
            target_at_actual_peak_rad=float(target[k]),request_at_actual_peak_rad=float(requested[k]),
            target_step_peak_rad=float(abs(target_delta).max()) if len(target_delta) else None))
(a.directory/'startup_joints.json').write_text(json.dumps(joint_rows,indent=2),encoding='utf-8')
