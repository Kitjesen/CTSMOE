"""Summarize the frozen matched startup interventions, excluding post-failure states."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
run = sys.argv[1] if len(sys.argv) > 1 else "results_r1"
p = HERE / run
m = json.loads((p / "manifest.json").read_text())
z = np.load(p / "traces.npz")
physics = np.load(p / "physics.npz")
names = m["joint_names"]
legs = [i for i, n in enumerate(names) if not n.endswith("_foot_joint")]
default = np.array(m["default_q"])
dt = m["dt"]
rows = []
for c in m["cases"]:
    ci = c["id"]
    begin = round(c["wait"] / dt)
    end = min(len(z["joint_pos"]), c.get("physical_end_step", 100000) + 1)
    q = z["joint_pos"][:end, ci][:, legs] - default[legs]
    rms = np.sqrt(np.mean(q**2, axis=1))
    target = z["target"][:end, ci][:, legs] - default[legs]
    request = z["requested"][:end, ci] - default[m["request_joint_ids"]]
    height = z["height"][:end, ci]
    vel = z["velocity"][:end, ci]
    cmd = z["command"][:end, ci]
    first = slice(begin, min(begin + 50, end))
    first2 = slice(begin, min(begin + 100, end))
    row = dict(policy=m["history"][c["policy"]]["label"], case=c["name"], id=ci,
        first1_rms_peak=float(rms[first].max()), first2_rms_peak=float(rms[first2].max()),
        first1_single_peak=float(np.abs(q[first]).max()),
        first2_height_min=float(height[first2].min()), first2_vx_max=float(vel[first2, 0].max()),
        late_rms=float(rms[-100:].mean()), late_height=float(height[-100:].mean()),
        late_xy_error=float(np.linalg.norm(vel[-100:, :2]-cmd[-100:, :2], axis=-1).mean()),
        first2_request_peak=float(np.abs(request[first2]).max()),
        first2_target_peak=float(np.abs(target[first2]).max()),
        first2_target_variation_mean=float(np.abs(np.diff(target[first2], axis=0)).sum(0).mean()),
        physical_end=c.get("physical_end_step"))
    row["reward_first2_sum"] = {k:float(z["rewards"][first2,ci,j].sum()*dt) for j,k in enumerate(m['reward_terms'])}
    row["first2_phase_weight_range"] = [float(z['phase_weight'][first2,ci].min()),float(z['phase_weight'][first2,ci].max())]
    peaks=[]
    speed=[]
    for group in ("hip", "thigh", "calf"):
        use=(physics[group+'_step']>=begin)&(physics[group+'_step']<min(begin+100,end))
        peaks.append(np.abs(physics[group+'_applied'][use,ci]).max())
        speed.append(np.abs(physics[group+'_qd'][use,ci]).max())
    row['first2_leg_torque_peak']=float(max(peaks))
    row['first2_leg_speed_peak']=float(max(speed))
    rows.append(row)
(HERE/f'{run}_summary.json').write_text(json.dumps(rows,indent=2))
for r in rows:
    print(r['id'],r['policy'],r['case'], 'rms1',round(r['first1_rms_peak'],4),
          'single',round(r['first1_single_peak'],4), 'hmin',round(r['first2_height_min'],4),
          'vmax',round(r['first2_vx_max'],3),'late',round(r['late_rms'],4),
          'err',round(r['late_xy_error'],3),'tau',round(r['first2_leg_torque_peak'],2),
          'fail',r['physical_end'])
ci=2
for group in ('hip','thigh','calf'):
    ids=m['actuator_init'][group]['joint_indices']
    print('PHYSICS',group,[names[i] for i in ids])
    for j in range(min(9,len(physics[group+'_step']))):
        print(int(physics[group+'_step'][j]),
            'in',np.round(physics[group+'_incoming'][j,ci],3).tolist(),
            'delay',np.round(physics[group+'_delayed'][j,ci],3).tolist(),
            'q',np.round(physics[group+'_q'][j,ci],3).tolist(),
            'applied',np.round(physics[group+'_applied'][j,ci],1).tolist())
