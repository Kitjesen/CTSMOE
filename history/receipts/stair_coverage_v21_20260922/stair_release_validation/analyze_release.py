"""Independent complete-flight and quiet-stop review of the ONNX action loop."""
import csv
import json
from pathlib import Path
import sys

import numpy as np


def analyze(root):
    rows, joint_rows = [], []
    for condition in ('nominal', 'perturbed'):
        folder = root/'results_verified'/condition
        manifest = json.loads((folder/'manifest.json').read_text())
        with np.load(folder/'traces.npz') as archive:
            arrays = dict(archive)
        with np.load(folder/'stair_profiles.npz') as archive:
            profiles = dict(archive)
        names = manifest['joint_names']
        leg = np.array([i for i, n in enumerate(names) if not n.endswith('foot_joint')])
        wheel = np.array([i for i, n in enumerate(names) if n.endswith('foot_joint')])
        default = np.array(manifest['default_q'])
        dt = manifest['dt']
        for c in manifest['cases']:
            i, count = c['id'], c['end_step']+1
            q = arrays['joint_pos'][:count, i]
            error = q[:, leg]-default[leg]
            nstart = min(count, int((2. if c['mode']=='settle' else 0.)/dt))
            startup = error[nstart:min(nstart+50, count)]
            support = arrays['stair_platform_supported'][:count, i]
            on_platform = arrays['stair_platform_hull'][:count, i]
            side = c['max_side_m']
            quiet_s = float(arrays['quiet_hold_steps'][:count, i].max()*dt)
            completed = bool(support.any()) and not c['physical'] and side <= .5
            if c['landing_mode'] == 'stop':
                completed &= c['end_reason'] == 'quiet_supported_platform_dwell' and quiet_s >= 2.-1e-6
            else:
                completed &= c['end_reason'] == 'safe_tile_endpoint'
            speed = arrays['speed_peak'][:count, i]
            torque = arrays['torque_peak'][:count, i]
            profile = profiles['z'][i]
            # Fine ray sampling resolves both vertical faces and sloped triangle
            # edges. Quantize plateau levels at 1 mm, not each derivative sample.
            unique, counts = np.unique(np.round(profile, 3), return_counts=True)
            plateaus = sorted(unique[counts >= 10].tolist())
            rises = np.diff(plateaus)
            hull_indices = np.flatnonzero(on_platform)
            window = slice(int(hull_indices[0]), count) if len(hull_indices) else slice(0, 0)
            vz = arrays['world_velocity'][window, i, 2]
            reasons = []
            if c['physical']:
                reasons += c['terms']
            if not support.any():
                reasons.append('never_full_hull_and_three_wheel_platform_support')
            if side > .5:
                reasons.append('lateral_excursion_gt_0.5m')
            if c['landing_mode']=='stop' and quiet_s < 2.-1e-6:
                reasons.append('quiet_supported_hold_lt_2s')
            r = dict(condition=condition, id=i, terrain=c['terrain'], level=c['level'],
                speed=c['velocity'][0], mode=c['mode'], landing_mode=c['landing_mode'],
                complete=completed, physical=c['physical'], end_reason=c['end_reason'],
                reasons=reasons, seconds=c['seconds'], max_side_m=side,
                platform_supported=bool(support.any()), quiet_hold_max_s=quiet_s,
                step_rise_min_m=float(rises.min()) if len(rises) else None,
                step_rise_max_m=float(rises.max()) if len(rises) else None,
                measured_plateau_count=len(plateaus), profile_total_elevation_m=float(abs(profile[-1]-profile[0])),
                startup_rms_peak_rad=float(np.sqrt(np.mean(startup**2, axis=1)).max()) if len(startup) else None,
                startup_joint_abs_peak_rad=float(abs(startup).max()) if len(startup) else None,
                leg_torque_peak_nm=float(torque[:, leg].max()), wheel_torque_peak_nm=float(torque[:, wheel].max()),
                leg_speed_peak_rad_s=float(speed[:, leg].max()), wheel_speed_peak_rad_s=float(speed[:, wheel].max()),
                leg_over15_control_steps=int((speed[:, leg].max(-1)>15.001).sum()),
                wheel_over30_control_steps=int((speed[:, wheel].max(-1)>30.001).sum()),
                hard_position_excess_rad=float(arrays['hard_excess'][:count, i].max()),
                post_full_hull_positive_vz_peak_m_s=float(np.maximum(vz, 0).max()) if len(vz) else None)
            rows.append(r)
            request_ids = manifest['request_joint_ids']
            for request_index, joint_id in enumerate(request_ids):
                joint_rows.append(dict(condition=condition, case_id=i, joint=names[joint_id],
                    requested_peak_abs_offset_rad=float(abs(arrays['requested'][:count, i, request_index]-default[joint_id]).max()),
                    limited_target_peak_abs_offset_rad=float(abs(arrays['target'][:count, i, joint_id]-default[joint_id]).max()),
                    actual_peak_abs_offset_rad=float(abs(q[:, joint_id]-default[joint_id]).max()),
                    actual_rms_offset_rad=float(np.sqrt(np.mean((q[:, joint_id]-default[joint_id])**2))),
                    target_rate_peak_rad_s=float(abs(np.diff(arrays['target'][:count, i, joint_id])/dt).max()) if count>1 else 0.,
                    torque_peak_nm=float(torque[:, joint_id].max()), speed_peak_rad_s=float(speed[:, joint_id].max())))
    groups = []
    for condition in ('nominal', 'perturbed'):
        for terrain in ('stairs_up', 'stairs_down'):
            for speed in (.3, .6, 1.2):
                subset = [r for r in rows if (r['condition'],r['terrain'],r['speed'])==(condition,terrain,speed)]
                groups.append(dict(condition=condition, terrain=terrain, speed=speed,
                    complete=sum(r['complete'] for r in subset), total=len(subset),
                    platform_reached=sum(r['platform_supported'] for r in subset),
                    physical_failures=sum(r['physical'] for r in subset)))
    result = dict(fixed_cases=True, randomized_success_rate=False, optimizer_updates=0,
        complete=sum(r['complete'] for r in rows), total=len(rows),
        release_accepted=all(r['complete'] for r in rows), groups=groups, cases=rows)
    (root/'BEHAVIOR_REVIEW.json').write_text(json.dumps(result, indent=2))
    with (root/'PER_JOINT.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(joint_rows[0]))
        writer.writeheader()
        writer.writerows(joint_rows)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}, indent=2))


if __name__ == '__main__':
    analyze(Path(sys.argv[1]))
