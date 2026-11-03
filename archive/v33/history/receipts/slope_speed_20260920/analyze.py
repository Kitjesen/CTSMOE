import json
from pathlib import Path
import sys

import numpy as np

folder = Path(sys.argv[1])
d = json.loads((folder/'results.json').read_text())
z = np.load(folder/'traces.npz')
t = z['control']
physics_all = z['physics']
fields = {}
offset = 0
for name, size in d['fields']:
    fields[name] = slice(offset, offset+size)
    offset += size
wheel = np.array(d['wheel_joints'])
leg = np.array([i for i in range(16) if i not in wheel])
records = []
for i, case in enumerate(d['cases']):
    n = case['step']+1
    c = t[:n,i]
    last = c[max(0,n-250):]
    get = lambda name: last[:,fields[name]]
    contact = get('wheel_force') > 5
    slip = get('slip_proxy')
    physics = physics_all[:n*4,i]
    speeds = physics[:,:16]
    torque = physics[:,16:32]
    peak_index = np.unravel_index(np.abs(speeds).argmax(),speeds.shape)
    peak_time = (peak_index[0]+1)*.005
    push_times = [p['step']*.02 for p in d['pushes'] if i in p['ids'] and p['step'] <= n]
    recent_push = max((x for x in push_times if x <= peak_time),default=None)
    r = dict(case, index=i,
        last5_vx=float(get('vel_b')[:,0].mean()),
        last5_world_vx=float(get('vel_w')[:,0].mean()),
        last5_vx_mae=float(np.abs(get('vel_b')[:,0]-case['speed']).mean()),
        max_yaw_deg=float(np.rad2deg(np.abs(c[:,fields['rpy']][:,2])).max()),
        last5_pitch_deg=float(np.rad2deg(get('rpy')[:,1]).mean()),
        last5_contact_wheels=float(contact.sum(1).mean()),
        last5_slip_proxy=float(slip[contact].mean()) if contact.any() else None,
        last5_leg_clip=float(get('clip_fraction')[:,leg].mean()),
        last5_wheel_clip=float(get('clip_fraction')[:,wheel].mean()),
        last5_wheel_abs_speed=float(np.abs(get('qvel')[:,wheel]).mean()),
        last5_wheel_abs_torque=float(np.abs(get('torque')[:,wheel]).mean()),
        leg_peak_torque=float(np.abs(torque[:,leg]).max()),
        wheel_peak_torque=float(np.abs(torque[:,wheel]).max()),
        peak_speed=float(np.abs(speeds).max()),peak_joint=d['joints'][peak_index[1]],
        peak_time=peak_time, peak_after_push=peak_time-recent_push if recent_push is not None else None)
    records.append(r)
groups = {}
for r in records:
    key = str((r['level'],r['speed'],r['mu'],r['heading'],r['push']))
    groups.setdefault(key,[]).append(r)
for key, rows in groups.items():
    print(key, dict(n=len(rows),passed=sum(r['passed'] for r in rows),
                   speed=sum(r['causes']['speed'] for r in rows),contact=sum('illegal_contact' in r['terms'] for r in rows),
                   x=round(float(np.mean([r['xyz'][0] for r in rows])),2),
                   vx=round(float(np.mean([r['last5_vx'] for r in rows])),2),
                   slip=round(float(np.mean([r['last5_slip_proxy'] or 0 for r in rows])),2)))
(folder/'analysis.json').write_text(json.dumps(records,indent=2))
print('SPEED CASES')
for r in records:
    if r['causes']['speed']:
        print({k:r[k] for k in ('index','level','mu','heading','push','seconds','xyz','peak_joint','peak_speed','peak_time','peak_after_push')})
