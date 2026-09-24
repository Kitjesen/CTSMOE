"""Measure recovery after intervention; report every command, not just winners."""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
m = json.loads((a.directory/'manifest.json').read_text())
if m['smoke_only']:
    raise ValueError('Smoke traces are not a recovery assessment')
with np.load(a.directory/'traces.npz') as f:
    d = {k: f[k] for k in f.files}
legs = [i for i, n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
default = np.asarray(m['default_q'])[legs]
rows = []
for c in m['cases']:
    i, n = c['id'], c['end_step']+1
    t = (np.arange(n)+1)*m['dt']
    q = d['joint_pos'][:n, i][:, legs]-default
    rms = np.sqrt(np.mean(q*q, axis=-1))
    intervention = c['intervention_time']
    start = 0 if intervention is None else round(intervention/m['dt'])
    baseline = next(x for x in m['cases'] if x['direction'] == c['direction'] and x['policy_source'] == 'current')
    before = min(start, n, baseline['end_step']+1)
    last = slice(max(0, n-50), n)
    speed_error = np.linalg.norm(d['velocity'][:n, i, :2]-d['command'][:n, i, :2], axis=-1)
    row = dict(direction=c['direction'], command=c['velocity'], policy_source=c['policy_source'],
               intervention_time=intervention, physical=c['physical'], terms=c['terms'],
               duration_s=n*m['dt'], first_peak_rms=float(rms[t<=1].max()),
               last_rms=float(rms[last].mean()), max_joint_rad=float(abs(q).max()),
               last_xy_error=float(speed_error[last].mean()),
               last_yaw_error=float(abs(d['angular_velocity'][last,i,2]-d['command'][last,i,2]).mean()),
               min_height=float(d['root_clearance'][:n,i].min()), max_height=float(d['root_clearance'][:n,i].max()),
               leg_speed_peak=float(d['speed_peak'][:n,i][:,legs].max()),
               pre_switch_leg_parity_max=float(abs(d['joint_pos'][:before,i][:,legs]-d['joint_pos'][:before,baseline['id']][:,legs]).max()) if before else None)
    after = rms[start:min(n, start+50)]
    row['post_switch_1s_mean_rms'] = float(after.mean()) if len(after) else None
    row['post_switch_1s_peak_rms'] = float(after.max()) if len(after) else None
    rows.append(row)
target = [r for r in rows if r['direction']=='vx_0.6']
switches = [r for r in rows if r['policy_source']=='switch_current_to_v6']
new_failures = [r for r in switches if r['physical'] and not next(
    b['physical'] for b in rows if b['direction']==r['direction'] and b['policy_source']=='current')]
critical = [r for r in target if r['intervention_time'] in (.1,.2,.4)]
gate = dict(ready_for_bounded_anchor_experiment=bool(not new_failures and all(
    r['last_rms']<.2 and r['last_xy_error']<.15 and r['duration_s']>=4 for r in critical)),
    critical_command_mps=.6, criterion='0.1/0.2/0.4s interventions recover to last-second RMS<.2rad, XY error<.15m/s and survive4s; no new physical failure at other commands',
    new_physical_failures=new_failures, not_a_training_success=True)
(a.directory/'recovery_analysis.json').write_text(json.dumps(dict(gate=gate,cases=rows),indent=2))
print(json.dumps(dict(gate=gate,forward_point_six=target),indent=2))
