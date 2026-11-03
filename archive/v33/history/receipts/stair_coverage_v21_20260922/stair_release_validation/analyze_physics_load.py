"""5ms torque/acceleration audit with geometric climbing windows and per-joint metrics."""
import csv
import json
from pathlib import Path
import sys
import zipfile

import numpy as np


def longest(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges==1), np.flatnonzero(edges==-1)
    return int((ends-starts).max()) if len(starts) else 0


def stats(acc, torque, velocity, computed, names, dt):
    if len(acc)==0:
        return None, []
    loc = np.unravel_index(abs(acc).argmax(), acc.shape)
    near = abs(torque)>=114.
    clipped = abs(computed-torque)>.001
    perjoint = []
    for j,name in enumerate(names):
        perjoint.append(dict(joint=name, acceleration_peak_rad_s2=float(abs(acc[:,j]).max()),
            acceleration_p99_rad_s2=float(np.quantile(abs(acc[:,j]),.99)),
            acceleration_rms_rad_s2=float(np.sqrt(np.mean(acc[:,j]**2))),
            torque_peak_nm=float(abs(torque[:,j]).max()),
            torque_rms_nm=float(np.sqrt(np.mean(torque[:,j]**2))),
            torque_ge114_fraction=float(near[:,j].mean()),
            torque_ge114_longest_s=longest(near[:,j])*dt,
            torque_clipped_fraction=float(clipped[:,j].mean()),
            torque_clipped_longest_s=longest(clipped[:,j])*dt,
            speed_peak_rad_s=float(abs(velocity[:,j]).max())))
    summary=dict(samples=len(acc), seconds=len(acc)*dt,
        acceleration_peak_rad_s2=float(abs(acc).max()),
        acceleration_p99_rad_s2=float(np.quantile(abs(acc),.99)),
        acceleration_rms_rad_s2=float(np.sqrt(np.mean(acc**2))),
        acceleration_peak_joint=names[loc[1]], acceleration_peak_index=int(loc[0]),
        acceleration_peak_sample_speed_before=float(velocity[max(0,loc[0]-1),loc[1]]),
        acceleration_peak_sample_speed_after=float(velocity[loc[0],loc[1]]),
        torque_peak_nm=float(abs(torque).max()),
        torque_max_joint_rms_nm=max(x['torque_rms_nm'] for x in perjoint),
        torque_ge114_any_joint_time_fraction=float(near.any(1).mean()),
        torque_ge114_longest_single_joint_s=max(x['torque_ge114_longest_s'] for x in perjoint),
        torque_over120_samples=int((abs(torque)>120.001).sum()),
        torque_clipped_any_joint_time_fraction=float(clipped.any(1).mean()),
        torque_clipped_longest_single_joint_s=max(x['torque_clipped_longest_s'] for x in perjoint),
        speed_peak_rad_s=float(abs(velocity).max()),
        leg_speed_over15_longest_single_joint_s=max(longest(abs(velocity[:,j])>15.001)*dt for j in range(12)))
    return summary, perjoint


def main(root):
    rows, joint_rows, capture = [], [], {}
    selected = {}
    selected_meta = []
    source = root.parent/'stair_release_validation_attempt02_20260922'
    for condition in ('nominal','perturbed'):
        folder = root/condition
        manifest = json.loads((folder/'manifest.json').read_text())
        capture[condition] = json.loads((folder/'PHYSICS_CAPTURE.json').read_text())
        with np.load(folder/'PHYSICS_5MS.npz') as z:
            physics = dict(z)
        with np.load(folder/'traces.npz') as z:
            control = dict(z)
        with np.load(folder/'stair_profiles.npz') as z:
            profile = dict(z)
        names=manifest['joint_names']
        legs=[i for i,n in enumerate(names) if not n.endswith('foot_joint')]
        wheels=[i for i,n in enumerate(names) if n.endswith('foot_joint')]
        feet=[i for i,n in enumerate(manifest['body_names']) if n.lower().endswith('_foot_link')]
        leg_names=[names[i] for i in legs]
        assert len(feet)==4 and len(legs)==12
        acc_diff = np.diff(physics['velocity'],axis=0)/.005
        max_acc_diff = float(abs(acc_diff-physics['acceleration'][1:]).max())
        capture[condition]['acceleration_vs_5ms_velocity_difference_max_error']=max_acc_diff
        np.testing.assert_allclose(acc_diff,physics['acceleration'][1:],atol=.01,rtol=1e-5)
        for case in manifest['cases']:
            i=case['id']
            count=case['end_step']+1
            n=count*4
            time=(np.arange(n)+1)*.005
            z=profile['z'][i]
            changes=np.flatnonzero(abs(z-z[0])>.005)
            if not len(changes):
                raise RuntimeError('No first stair edge')
            edge_x=float(profile['x'][changes[0]])
            front=control['body_pos'][:count,i][:,feet,0].max(-1)+.093
            first=np.flatnonzero(front>=edge_x)
            climb_start=int(first[0])*4 if len(first) else n
            full=np.flatnonzero(control['stair_platform_hull'][:count,i])
            climb_end=(int(full[0])+1)*4 if len(full) else n
            ranges={'whole':(0,n),'climb':(climb_start,climb_end),
                    'after_platform':(climb_end,n)}
            row=dict(condition=condition,id=i,terrain=case['terrain'],level=case['level'],
                speed=case['velocity'][0],mode=case['mode'],landing_mode=case['landing_mode'],
                physical=case['physical'],end_reason=case['end_reason'],
                first_step_edge_x_m=edge_x,climb_start_s=climb_start*.005,climb_end_s=climb_end*.005,
                phases={})
            for phase,(start,end) in ranges.items():
                a=physics['acceleration'][start:end,i][:,legs]
                torque=physics['torque'][start:end,i][:,legs]
                v=physics['velocity'][start:end,i][:,legs]
                computed=physics['computed_torque'][start:end,i][:,legs]
                values, joint=stats(a,torque,v,computed,leg_names,.005)
                if values:
                    values['acceleration_peak_time_s']=(start+values['acceleration_peak_index']+1)*.005
                row['phases'][phase]=values
                for j in joint:
                    joint_rows.append(dict(condition=condition,case_id=i,terrain=case['terrain'],
                        level=case['level'],speed=case['velocity'][0],phase=phase,**j))
            row['wheel_torque_peak_nm']=float(abs(physics['torque'][:n,i][:,wheels]).max())
            row['wheel_speed_peak_rad_s']=float(abs(physics['velocity'][:n,i][:,wheels]).max())
            rows.append(row)
            pick=(condition=='nominal' and i in (28,64)) or case['physical']
            if pick:
                key=condition+'_'+str(i)+'__'
                for field in ('acceleration','torque','velocity','computed_torque'):
                    selected[key+field]=physics[field][:n,i]
                selected[key+'time']=time
                selected[key+'wheel_force_z_20ms']=control['wheel_ground_force_z'][:count,i]
                selected[key+'root_velocity_20ms']=control['world_velocity'][:count,i]
                selected_meta.append(dict(**row,joint_names=names,leg_ids=legs,wheel_ids=wheels))
    groups=[]
    for condition in ('nominal','perturbed'):
        for terrain in ('stairs_up','stairs_down'):
            for speed in (.3,.6,1.2):
                subset=[r for r in rows if (r['condition'],r['terrain'],r['speed'])==(condition,terrain,speed)]
                climb=[r['phases']['climb'] for r in subset if r['phases']['climb']]
                groups.append(dict(condition=condition,terrain=terrain,speed=speed,cases=len(subset),
                    acceleration_peak_rad_s2=max(r['acceleration_peak_rad_s2'] for r in climb),
                    acceleration_p99_worst_case=max(r['acceleration_p99_rad_s2'] for r in climb),
                    torque_peak_nm=max(r['torque_peak_nm'] for r in climb),
                    torque_max_joint_rms_nm=max(r['torque_max_joint_rms_nm'] for r in climb),
                    near_limit_longest_single_joint_s=max(r['torque_ge114_longest_single_joint_s'] for r in climb),
                    near_limit_case_count=sum(r['torque_ge114_any_joint_time_fraction']>0 for r in climb),
                    torque_limit_violated_cases=sum(r['torque_over120_samples']>0 for r in climb),
                    physical_failures=sum(r['physical'] for r in subset)))
    report=dict(optimizer_updates=0,cases=len(rows),physics_dt=.005,capture=capture,groups=groups,
        interpretation='Acceleration is simulator joint velocity change over5ms; no validated motor/structure acceleration allowable is available.',
        climb_window='First wheel horizontal envelope reaches first step edge until full collision hull reaches terminal platform; geometric window, not a contact classifier.',
        rows=rows,prior_source=str(source))
    (root/'LOAD_AUDIT.json').write_text(json.dumps(report,indent=2))
    (root/'SELECTED_LOAD_CASES.json').write_text(json.dumps(selected_meta,indent=2))
    np.savez_compressed(root/'SELECTED_LOAD_TRACES.npz',**selected)
    with (root/'LOAD_PER_JOINT.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(joint_rows[0]))
        writer.writeheader()
        writer.writerows(joint_rows)
    with zipfile.ZipFile(root/'LOAD_REVIEW.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('LOAD_AUDIT.json','SELECTED_LOAD_CASES.json','SELECTED_LOAD_TRACES.npz','LOAD_PER_JOINT.csv','FINALIZED.json'):
            archive.write(root/name,name)
    print(json.dumps(dict(capture=capture,groups=groups,selected=selected_meta),indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]))
