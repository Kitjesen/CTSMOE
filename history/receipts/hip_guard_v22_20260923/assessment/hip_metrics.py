"""Per-joint and per-case evidence; fixed cases are not random success rates."""
import csv
import json
from pathlib import Path
import sys

import numpy as np


def analyze(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    arrays = np.load(folder/'traces.npz')
    names, dt = manifest['joint_names'], manifest['dt']
    legs = [i for i,n in enumerate(names) if not n.endswith('foot_joint')]
    defaults = np.asarray(manifest['default_q'])
    request_ids = manifest['request_joint_ids']
    signs = {'fr_hip_joint': -1., 'fl_hip_joint': 1., 'rr_hip_joint': 1., 'rl_hip_joint': -1.}
    cases, joints = [], []
    totals = {}
    for case in manifest['cases']:
        i, count = case['id'], case['end_step']+1
        error = arrays['joint_pos'][:count,i] - defaults
        pose_rms = np.sqrt((error[:,legs]**2).mean(-1))
        group = manifest['history'][case['policy']]['label']+'/'+case['terrain']
        total = totals.setdefault(group, dict(cases=0, route_4m=0, straight_4m=0, physical=0, platform_dwell=0, pit_success=0))
        total['cases'] += 1
        total['route_4m'] += int(case.get('passed_4m', False))
        total['straight_4m'] += int(case.get('straight_4m', False))
        total['physical'] += int(case.get('physical', False))
        total['platform_dwell'] += int(case.get('end_reason') == 'supported_platform_dwell')
        total['pit_success'] += int(case.get('stable', False))
        velerr = arrays['velocity'][:count,i,:2]-arrays['command'][:count,i,:2]
        yawerr = arrays['angular_velocity'][:count,i,2]-arrays['command'][:count,i,2]
        row = dict(id=i, policy=case['policy'], terrain=case['terrain'], level=case['level'],
            mode=case['mode'], command=case['velocity'], seconds=count*dt,
            physical=case.get('physical', False), terms=case.get('terms'), end_reason=case.get('end_reason'),
            startup_pose_rms_peak=float(pose_rms[:min(count,round(1/dt))].max()),
            terminal_pose_rms_mean=float(pose_rms[-min(count,round(1/dt)):].mean()),
            xy_rmse=float(np.sqrt((velerr**2).sum(-1).mean())), yaw_rmse=float(np.sqrt((yawerr**2).mean())),
            calf_contact_peak_N=float(arrays['calf_contact_peak'][:count,i].max()),
            max_side_m=case.get('max_side_m'), straight_4m=case.get('straight_4m'),
            pit_success=case.get('stable'), platform_hold_s=case.get('platform_hold_s'))
        cases.append(row)
        for j in legs:
            req_j = request_ids.index(j)
            joint = dict(case=i, policy=case['policy'], terrain=case['terrain'], level=case['level'], joint=names[j],
                actual_abs_deviation_peak=float(np.abs(error[:,j]).max()),
                startup_abs_deviation_peak=float(np.abs(error[:min(count,round(1/dt)),j]).max()),
                target_abs_deviation_peak=float(np.abs(arrays['target'][:count,i,j]-defaults[j]).max()),
                request_abs_deviation_peak=float(np.abs(arrays['requested'][:count,i,req_j]-defaults[j]).max()),
                torque_peak_Nm=float(arrays['torque_peak'][:count,i,j].max()),
                speed_peak_rad_s=float(arrays['speed_peak'][:count,i,j].max()),
                acceleration_peak_rad_s2=float(arrays['acceleration_peak'][:count,i,j].max()))
            if names[j] in signs:
                inward = arrays['joint_pos'][:count,i,j] * signs[names[j]]
                target = arrays['target'][:count,i,j] * signs[names[j]]
                joint.update(actual_inward_peak=float(inward.max()), target_inward_peak=float(target.max()),
                             seconds_inward_gt0121=float((inward>.121).sum()*dt))
            joints.append(joint)
    keys = list(dict.fromkeys(k for row in joints for k in row))
    with (folder/'HIP_PER_JOINT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(joints)
    result = dict(fixed_cases_only=True, optimizer_updates=0, totals=totals, cases=cases,
                  joints=joints, default_q=manifest['default_q'], hip_contract=manifest['hip_contract'])
    (folder/'HIP_REVIEW.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(totals))


if __name__ == '__main__':
    analyze(sys.argv[1])
