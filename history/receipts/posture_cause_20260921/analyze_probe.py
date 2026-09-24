"""Matched trajectory parity, terrain-specific startup, and real reward ledger."""
import json
from pathlib import Path

import numpy as np

root=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
directory=root/'recording/posture_cause_reward_20260921'
baseline=root/'recording/flat_v9_landing_20260921'
m=json.loads((directory/'manifest.json').read_text())
with np.load(directory/'traces.npz') as f:
    d={k:f[k] for k in f.files}
assert not m['smoke_only']
np.testing.assert_allclose(d['weighted_reward'].sum(-1),d['reward_total'],atol=1e-5,rtol=1e-5)
np.testing.assert_allclose(d['command'],d['reward_command'],atol=1e-6)
assert np.isfinite(d['weighted_reward']).all()
bm=json.loads((baseline/'manifest.json').read_text())
parity={}
flat_start_parity={}
with np.load(baseline/'traces.npz') as old:
    for key in ('joint_pos','raw_action','root_clearance','command'):
        old_values=old[key]
        errors=[]
        flat_errors=[]
        for c in m['cases']:
            bc=bm['cases'][c['id']]
            assert (c['policy'],c['terrain'],c['level'],c['velocity']) == (bc['policy'],bc['terrain'],bc['level'],bc['velocity'])
            n=min(c['end_step']+1,bc['end_step']+1)
            errors.append(float(abs(d[key][:n,c['id']]-old_values[:n,c['id']]).max()))
            if c['terrain']=='flat' and c['mode']=='direct':
                flat_errors.append(float(abs(d[key][:min(n,100),c['id']]-old_values[:min(n,100),c['id']]).max()))
        parity[key]=max(errors)
        flat_start_parity[key]=max(flat_errors)
legs=[i for i,n in enumerate(m['joint_names']) if not n.endswith('_foot_joint')]
wheel_bodies=[i for i,n in enumerate(m['body_names']) if '_foot_' in n.lower()]
base_id=m['body_names'].index('base_link')
q0=np.asarray(m['default_q'])[legs]
forward_replay_parity=[]
with np.load(baseline/'traces.npz') as old:
    old_q=old['joint_pos']
    for c in m['cases']:
        if c['terrain']=='flat' and c['direction']=='forward' and c['mode']=='direct':
            i=c['id']
            n=min(50,c['end_step']+1,bm['cases'][i]['end_step']+1)
            current=d['joint_pos'][:n,i][:,legs]
            previous=old_q[:n,i][:,legs]
            forward_replay_parity.append(dict(label=m['history'][c['policy']]['label'],
                first_second_leg_max_difference_rad=float(abs(current-previous).max()),
                current_peak_rms=float(np.sqrt(((current-q0)**2).mean(-1)).max()),
                previous_peak_rms=float(np.sqrt(((previous-q0)**2).mean(-1)).max())))
rows=[]
startup=[]
for c in m['cases']:
    n,i=c['end_step']+1,c['id']
    start=100 if c['mode']=='settle' else 0
    for name,length in [('first_1s',50),('first_2s',100)]:
        hi=min(start+length,n)
        if hi<=start:
            continue
        score=d['weighted_reward'][start:hi,i].sum(0)
        rows.append(dict(label=m['history'][c['policy']]['label'],terrain=c['terrain'],
            level=c['level'],direction=c['direction'],mode=c['mode'],velocity=c['velocity'],stage=name,
            seconds=(hi-start)*m['dt'],total=float(score.sum()),
            contributions=dict(zip(m['reward_names'],map(float,score)))))
    hi=min(start+50,n)
    if hi<=start:
        continue
    q=d['joint_pos'][start:hi,i][:,legs]-q0
    rms=np.sqrt((q*q).mean(-1))
    k=int(rms.argmax())+start
    gap=d['pit_front_gap'][start:hi,i]
    startup.append(dict(label=m['history'][c['policy']]['label'],terrain=c['terrain'],level=c['level'],
        direction=c['direction'],mode=c['mode'],velocity=c['velocity'],
        rms_peak=float(rms.max()),joint_peak=float(abs(q).max()),
        at_peak=dict(time_s=(k-start+1)*m['dt'],joint_deviation=q[k-start].tolist(),
            root_x=float(d['body_pos'][k,i,base_id,0]),
            front_wheel_x=float(d['body_pos'][k,i,wheel_bodies,0].max()+.093),
            scanner_relief=float(d['ground_relief'][k,i]),phase_weight=float(d['phase_weight'][k,i]),
            climbed=bool(d['climb'][k,i])),
        pit_gap_min=float(gap.min()) if np.isfinite(gap).all() else None))
result=dict(parity=parity,flat_start_parity=flat_start_parity,forward_replay_parity=forward_replay_parity,
            reward_weights=m['reward_weights'],rows=rows,startup=startup)
(directory/'reward_analysis.json').write_text(json.dumps(result,indent=2))
summary=dict(parity=parity,flat_start_parity=flat_start_parity,forward_replay_parity=forward_replay_parity,
             forward=[r for r in rows if r['terrain']=='flat' and r['direction']=='forward'
                                  and r['mode']=='direct'],
             terrain_06=[r for r in startup if r['label']=='V9_44711' and r['velocity']==[.6,0.,0.]
                         and r['mode']=='direct'])
(directory/'reward_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(dict(parity=parity,flat_forward=[dict(label=r['label'],stage=r['stage'],total=r['total'])
                                             for r in summary['forward']]),indent=2))
