"""Measure high-step failure symptoms from existing Isaac traces; no training changes."""
import json
from pathlib import Path

import numpy as np


def first_time(times, mask):
    selected = np.flatnonzero(mask)
    return float(times[selected[0]]) if len(selected) else None


root = Path(__file__).resolve().parents[2]
folder = root / 'recordings/straight_high13501_20260918/trace'
manifest = json.loads((folder / 'manifest.json').read_text())
legs = [i for i, name in enumerate(manifest['joint_names']) if '_foot_' not in name]
wheels = [i for i, name in enumerate(manifest['joint_names']) if '_foot_' in name]
contact_names = manifest['probe']['contact_body_names']
calves = [i for i, name in enumerate(contact_names) if 'calf' in name.lower()]
wheel_bodies = [i for i, name in enumerate(contact_names) if 'foot' in name.lower()]
base = contact_names.index('base_link')
results = []
for clip in manifest['clips']:
    records = json.loads((folder / (clip['name'] + '.json')).read_text())
    times = np.array([r['time'] for r in records])
    cmd = np.array([r['command'] for r in records])
    velocity = np.array([r['velocity'] for r in records])
    position = np.array([r['root_pos'] for r in records])
    quat = np.array([r['root_quat'] for r in records])
    w, x, y, z = quat.T
    yaw = np.unwrap(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
    tilt = np.rad2deg(np.arccos(np.clip([r['upright'] for r in records], -1, 1)))
    computed = np.array([r['computed_torque'] for r in records])
    applied = np.array([r['applied_torque'] for r in records])
    force = np.array([r['contact_forces_w'] for r in records])
    force_norm = np.linalg.norm(force, axis=2)
    clipped = np.abs(computed-applied) > 0.001
    moving = cmd[:, 0] > 0
    good = moving & (tilt < 60)
    intervals = []
    for start, end in ((2, 6), (6, 12), (12, 18), (18, 25.01)):
        mask = moving & (times >= start) & (times < end)
        if not mask.any():
            continue
        intervals.append({
            'from_s': start, 'to_s': float(times[mask][-1]),
            'mean_vx': float(velocity[mask, 0].mean()),
            'mean_vy': float(velocity[mask, 1].mean()),
            'max_tilt_deg': float(tilt[mask].max()),
            'calf_contact_over5N_frames_percent': float((force_norm[mask][:, calves] > 5).any(axis=1).mean()*100),
            'wheel_torque_clipping_joint_samples_percent': float(clipped[mask][:, wheels].mean()*100),
            'leg_torque_clipping_joint_samples_percent': float(clipped[mask][:, legs].mean()*100),
        })
    body_contacts = {}
    for i, name in enumerate(contact_names):
        if 'foot' not in name.lower():
            body_contacts[name] = {'first_over5N_s': first_time(times, moving & (force_norm[:, i] > 5)),
                                   'frames_over5N_percent': float((force_norm[moving, i] > 5).mean()*100),
                                   'peak_N': float(force_norm[moving, i].max())}
    results.append({
        'level': clip['terrain']['level'], 'riser_cm': clip['terrain']['actual_step_height_m']*100,
        'mean_vx_before_tilt60': float(velocity[good, 0].mean()),
        'fraction_forward_speed_below_0p05_before_tilt60': float((velocity[good, 0] < .05).mean()),
        'first_tilt_over60_s': first_time(times, tilt > 60),
        'first_tilt_over90_s': first_time(times, tilt > 90),
        'first_base_contact_over5N_s': first_time(times, force_norm[:, base] > 5),
        'max_abs_yaw_drift_before_tilt60_deg': float(np.rad2deg(np.abs(yaw[good] - yaw[0])).max()),
        'leg_clipping_before_tilt60_joint_samples_percent': float(clipped[good][:, legs].mean()*100),
        'wheel_clipping_before_tilt60_joint_samples_percent': float(clipped[good][:, wheels].mean()*100),
        'leg_peak_torque_before_tilt60_Nm': float(np.abs(applied[good][:, legs]).max()),
        'wheel_peak_torque_before_tilt60_Nm': float(np.abs(applied[good][:, wheels]).max()),
        'delta_xy_m': (position[-1, :2] - position[0, :2]).tolist(),
        'intervals': intervals, 'nonwheel_contacts': body_contacts,
    })
output = {'checkpoint_updates': 13501, 'clips': results,
          'limits': 'One nominal rollout per level. Torque and contact sample only the last physics substep at 50 Hz. These are symptoms, not intervention-based proof of a mechanical cause.'}
Path(__file__).with_name('high_stairs_symptoms.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
