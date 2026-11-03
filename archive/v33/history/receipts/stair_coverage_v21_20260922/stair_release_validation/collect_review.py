"""Collect compact evidence without altering full simulator traces."""
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

root = Path(sys.argv[1])
review = json.loads((root/'BEHAVIOR_REVIEW.json').read_text())
rows = review['cases']
summary = dict(
    failure_reasons=dict(Counter(x for r in rows if not r['complete'] for x in r['reasons'])),
    physical_cases=[r for r in rows if r['physical']],
    all_stop_cases=sum(r['landing_mode']=='stop' for r in rows),
    quiet_stop_complete=sum(r['complete'] and r['landing_mode']=='stop' for r in rows),
    quiet_hold_achieved=sum(r['quiet_hold_max_s']>=2.-1e-6 for r in rows if r['landing_mode']=='stop'),
    all_continue_cases=sum(r['landing_mode']=='continue' for r in rows),
    continue_complete=sum(r['complete'] and r['landing_mode']=='continue' for r in rows),
    over_nominal_leg_speed_cases=sum(r['leg_over15_control_steps']>0 for r in rows),
    over_nominal_wheel_speed_cases=sum(r['wheel_over30_control_steps']>0 for r in rows),
    peaks={k:max(r[k] for r in rows if r[k] is not None) for k in (
        'leg_torque_peak_nm','wheel_torque_peak_nm','leg_speed_peak_rad_s',
        'wheel_speed_peak_rad_s','hard_position_excess_rad','startup_rms_peak_rad',
        'startup_joint_abs_peak_rad','post_full_hull_positive_vz_peak_m_s')},
    geometry=[{k:r[k] for k in ('terrain','level','step_rise_min_m','step_rise_max_m',
                               'measured_plateau_count','profile_total_elevation_m')}
              for r in rows if r['condition']=='nominal' and r['speed']==.6 and
              r['mode']=='direct' and r['landing_mode']=='continue'])
(root/'SUMMARY.json').write_text(json.dumps(summary, indent=2))
selected = [next(r for r in rows if r['condition']=='nominal' and r['terrain']==terrain and
                 r['level']==9 and r['speed']==.6 and r['mode']=='direct' and r['landing_mode']=='continue')
            for terrain in ('stairs_up','stairs_down')]
selected.extend(r for r in rows if r['physical'])
selected.append(max([r for r in rows if r['condition']=='perturbed' and r['terrain']=='stairs_down' and r['speed']==.3],
                    key=lambda r:r['max_side_m']))
selected.append(next(r for r in rows if r['condition']=='nominal' and r['terrain']=='stairs_down' and
                     r['landing_mode']=='stop' and not r['complete']))
trace_output = {}
for condition in ('nominal','perturbed'):
    folder = root/'results_verified'/condition
    manifest = json.loads((folder/'manifest.json').read_text())
    with np.load(folder/'traces.npz') as archive:
        arrays = dict(archive)
    for row in selected:
        if row['condition'] != condition:
            continue
        key = condition+'_'+str(row['id'])
        case = manifest['cases'][row['id']]
        row['manifest_case'] = case
        row['default_q'] = manifest['default_q']
        row['joint_names'] = manifest['joint_names']
        row['body_names'] = manifest['body_names']
        row['request_joint_ids'] = manifest['request_joint_ids']
        row['dt'] = manifest['dt']
        for name in ('joint_pos','target','requested','velocity','world_velocity','body_pos',
                     'gravity','speed_peak','torque_peak','angular_velocity','quiet_hold_steps',
                     'stair_platform_hull','stair_platform_supported','root_clearance','wheel_ground_force_z'):
            trace_output[key+'__'+name] = arrays[name][:case['end_step']+1, row['id']]
np.savez_compressed(root/'SELECTED_TRACES.npz', **trace_output)
(root/'SELECTED_CASES.json').write_text(json.dumps(selected, indent=2))
with zipfile.ZipFile(root/'review_bundle.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
    for file in [root/'BEHAVIOR_REVIEW.json', root/'SUMMARY.json', root/'PER_JOINT.csv',
                 root/'IO_VERIFICATION.json',root/'SELECTED_TRACES.npz',root/'SELECTED_CASES.json',
                 *list((root/'results_verified').rglob('*.json'))]:
        archive.write(file, file.relative_to(root))
print(json.dumps(summary, indent=2))
