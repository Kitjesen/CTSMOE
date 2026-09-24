"""Numpy-only measured braking events from evaluate_measured manifest/NPZ files.

Commands are issued before a step; all kinematic traces are captured afterwards.
No protocol-clock event is fabricated, and all cases stop at their valid end_step.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np


THRESHOLDS = dict(xy_stop_m_s=.1, yaw_stop_rad_s=.15, stable_s=.5,
                  entry_window_s=1., entry_relative_tolerance=.2,
                  entry_tracking_fraction=.8, zero_command_atol=1e-8)
TRACE_FIELDS = ('command', 'velocity', 'angular_velocity', 'gravity', 'body_pos', 'joint_pos',
                'requested', 'target', 'torque', 'torque_peak', 'speed_peak', 'root_clearance',
                'base_contact', 'active', 'world_velocity')


def _peak(values):
    return float(np.max(np.abs(values)))


def _integral(values, dt):
    return float(np.sum((values[:-1] + values[1:]) * .5) * dt)


def _first_stable(quiet, dt):
    # Require .5 seconds between measured samples, not just 25 samples at 50 Hz.
    span = int(math.ceil(THRESHOLDS['stable_s'] / dt - 1e-10))
    run_start = None
    for index, value in enumerate(quiet):
        run_start = index if value and run_start is None else run_start
        if not value:
            run_start = None
        elif index - run_start >= span:
            return run_start, index
    return None, None


def _entry(command, actual, entry, dt):
    start = max(0, entry + 1 - int(math.ceil(THRESHOLDS['entry_window_s'] / dt)))
    window = actual[start:entry + 1]
    expected = command[entry]
    tolerances = np.maximum(np.abs(expected) * THRESHOLDS['entry_relative_tolerance'], 1e-8)
    zero = np.abs(expected) <= THRESHOLDS['zero_command_atol']
    tolerances[zero] = np.array([.1, .1, .15])[zero]
    tracked = (np.abs(window - expected) <= tolerances).all(-1)
    # Do not qualify a window belonging to a preceding different command segment.
    tracked &= np.isclose(command[start:entry + 1], expected, atol=1e-8, rtol=0).all(-1)
    ratios = {}
    xy_norm = float(np.linalg.norm(expected[:2]))
    if xy_norm > THRESHOLDS['zero_command_atol']:
        ratios['xy_signed_projection'] = float(np.dot(actual[entry, :2], expected[:2]) / xy_norm**2)
    if abs(expected[2]) > THRESHOLDS['zero_command_atol']:
        ratios['yaw_signed'] = float(actual[entry, 2] / expected[2])
    return dict(
        commanded_velocity_before_stop=expected.tolist(),
        actual_body_velocity_before_stop=actual[entry].tolist(),
        entry_xy_speed_m_s=float(np.linalg.norm(actual[entry, :2])),
        entry_yaw_speed_abs_rad_s=float(abs(actual[entry, 2])),
        entry_speed_ratio=min(ratios.values()), entry_speed_ratio_components=ratios,
        pre_stop_tracking_window_s=len(window) * dt,
        pre_stop_tracking_fraction=float(tracked.mean()),
        reached_command_before_stop=bool(tracked[-1] and tracked.mean() >= THRESHOLDS['entry_tracking_fraction']),
    )


def _event(manifest, data, case, start, end, next_motion, count, leg_ids, wheel_ids, base_id):
    dt, index = float(manifest['dt']), case['id']
    command = data['command'][:count, index]
    actual = np.column_stack((data['velocity'][:count, index, :2], data['angular_velocity'][:count, index, 2]))
    entry = start - 1
    sl = slice(entry, end + 1)
    velocity = actual[sl]
    quiet = (np.linalg.norm(velocity[:, :2], axis=-1) < THRESHOLDS['xy_stop_m_s'])
    quiet &= np.abs(velocity[:, 2]) < THRESHOLDS['yaw_stop_rad_s']
    stable_start, stable_confirm = _first_stable(quiet, dt)
    position = data['body_pos'][sl, index, base_id]
    delta = np.diff(position[:, :2], axis=0)
    distances = np.linalg.norm(delta, axis=-1)
    displacement = position[-1, :2] - position[0, :2]
    gravity = data['gravity'][sl, index]
    gravity_norm = np.linalg.norm(gravity, axis=-1)
    if np.any(gravity_norm <= 0):
        raise ValueError(f'case {index} has zero gravity vector')
    tilt = np.arccos(np.clip(-gravity[:, 2] / gravity_norm, -1., 1.))
    default = np.asarray(manifest['default_q'])[leg_ids]
    actual_offset = data['joint_pos'][sl, index][:, leg_ids] - default
    control_sl = slice(start, end + 1)
    target_offset = data['target'][control_sl, index][:, leg_ids] - default
    requested_offset = data['requested'][control_sl, index] - default
    torque_key = 'torque_peak' if 'torque_peak' in data else 'torque'
    torque, speed = data[torque_key][control_sl, index], data['speed_peak'][control_sl, index]
    entry_norm = float(np.linalg.norm(velocity[0, :2]))
    rebound = (float(np.maximum(-(velocity[:, :2] @ (velocity[0, :2] / entry_norm)), 0.).max())
               if entry_norm > 1e-8 else None)
    yaw_rebound = (float(np.maximum(-velocity[:, 2] * np.sign(velocity[0, 2]), 0.).max())
                   if abs(velocity[0, 2]) > 1e-8 else None)
    entry_values = _entry(command, actual, entry, dt)
    ends_at_case_end = not next_motion
    physical_during_stop = bool(ends_at_case_end and case.get('physical', False))
    failure_during_stop = ends_at_case_end and (physical_during_stop or any(
        term not in ('time_out', 'budget') for term in case.get('terms', [])))
    inactive_censored = count < int(case['end_step']) + 1
    return dict(
        stop_start_step=start, stop_command_time_s=start * dt,
        entry_state_time_s=start * dt, zero_command_end_step=end,
        zero_window_observed_s=(end - entry) * dt,
        zero_window_censored=ends_at_case_end,
        physical_failure_during_stop=physical_during_stop,
        terminal_failure_during_stop=bool(failure_during_stop),
        stable_stop_time_s=None if stable_start is None else stable_start * dt,
        stable_stop_confirmation_s=None if stable_confirm is None else stable_confirm * dt,
        qualified_stop=bool(stable_start is not None and entry_values['reached_command_before_stop']
                            and not failure_during_stop and not (ends_at_case_end and inactive_censored)),
        path_distance_xy_m=float(distances.sum()),
        path_until_stable_stop_m=None if stable_start is None else float(distances[:stable_start].sum()),
        displacement_xy_m=displacement.tolist(), displacement_xy_norm_m=float(np.linalg.norm(displacement)),
        body_z_angular_velocity_integral_rad=_integral(velocity[:, 2], dt),
        body_z_abs_angular_velocity_integral_rad=_integral(np.abs(velocity[:, 2]), dt),
        reverse_rebound_body_projection_peak_m_s=rebound,
        reverse_yaw_body_z_peak_rad_s=yaw_rebound,
        posture_tilt_rms_rad=float(np.sqrt(np.mean(tilt**2))), posture_tilt_peak_rad=_peak(tilt),
        joint_actual_offset_rms_rad=float(np.sqrt(np.mean(actual_offset**2))),
        joint_actual_offset_peak_abs_rad=_peak(actual_offset),
        joint_actual_offset_peak_abs_rad_by_joint=np.max(np.abs(actual_offset), axis=0).tolist(),
        joint_requested_offset_peak_abs_rad_by_joint=np.max(np.abs(requested_offset), axis=0).tolist(),
        joint_target_offset_peak_abs_rad_by_joint=np.max(np.abs(target_offset), axis=0).tolist(),
        root_clearance_range_m=[float(data['root_clearance'][sl, index].min()),
                                float(data['root_clearance'][sl, index].max())],
        base_height_relative_origin_range_m=[float(position[:, 2].min()), float(position[:, 2].max())],
        vertical_velocity_body_z_peak_abs_m_s=_peak(data['velocity'][sl, index, 2]),
        vertical_velocity_world_z_peak_abs_m_s=(_peak(data['world_velocity'][sl, index, 2])
                                               if 'world_velocity' in data else None),
        leg_torque_abs_peak_nm=_peak(torque[:, leg_ids]), wheel_torque_abs_peak_nm=_peak(torque[:, wheel_ids]),
        leg_speed_abs_peak_rad_s=_peak(speed[:, leg_ids]), wheel_speed_abs_peak_rad_s=_peak(speed[:, wheel_ids]),
        base_contact_peak_n=_peak(data['base_contact'][sl, index]),
        **entry_values,
    )


def analyze_braking(manifest, traces):
    """Return one JSON-ready report per policy label, preserving manifest IDs."""
    data = {key: np.asarray(traces[key]) for key in TRACE_FIELDS if key in traces}
    command = data['command']
    if command.ndim != 3 or command.shape[-1] != 3:
        raise ValueError('command must have shape (T,N,3)')
    steps, count, _ = command.shape
    dt = float(manifest['dt'])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    names, bodies = manifest['joint_names'], manifest['body_names']
    leg_ids = list(manifest['request_joint_ids'])
    wheel_ids = [i for i, name in enumerate(names) if name.endswith('_foot_joint')]
    if len(leg_ids) != 12 or len(set(leg_ids)) != 12 or len(wheel_ids) != 4:
        raise ValueError('expected 12 unique requested leg IDs and four *_foot_joint wheels')
    if set(leg_ids) & set(wheel_ids) or set(leg_ids + wheel_ids) != set(range(len(names))):
        raise ValueError('joint IDs must partition the manifest joint names')
    base_id = bodies.index('base_link')
    expected = dict(command=(3,), velocity=(3,), angular_velocity=(3,), gravity=(3,),
                    body_pos=(len(bodies), 3), joint_pos=(len(names),), requested=(12,),
                    target=(len(names),), torque=(len(names),), speed_peak=(len(names),),
                    root_clearance=(), base_contact=(), active=())
    if 'torque_peak' in data:
        expected['torque_peak'] = (len(names),)
    if 'world_velocity' in data:
        expected['world_velocity'] = (3,)
    for key, tail in expected.items():
        if key not in data or data[key].shape != (steps, count, *tail):
            raise ValueError(f'{key} must have shape {(steps, count, *tail)}')
    if np.asarray(manifest['default_q']).shape != (len(names),):
        raise ValueError('default_q must match joint_names')
    reports, seen = {}, set()
    for case in manifest['cases']:
        index, end = int(case['id']), int(case['end_step'])
        if index in seen or not 0 <= index < count or not 0 <= end < steps:
            raise ValueError('case ID/end_step is duplicate or outside the recorded trace')
        seen.add(index)
        inactive = np.flatnonzero(~data['active'][:end + 1, index].astype(bool))
        valid = min(end + 1, int(inactive[0])) if len(inactive) else end + 1
        for key in expected:
            if not np.isfinite(data[key][:valid, index]).all():
                raise ValueError(f'case {index} has nonfinite {key} within its valid trajectory')
        history = manifest['history'][int(case['policy'])]
        label = history['label']
        if label not in reports:
            reports[label] = dict(
                schema_version=1, kind='measured_braking_events', label=label, checkpoint=history['path'],
                teacher=bool(history.get('teacher', False)), protocol_id=manifest['protocol_id'], dt=dt,
                thresholds=dict(THRESHOLDS), requested_joint_names=[names[i] for i in leg_ids],
                coordinate_notes=dict(
                    path='base_link world-axis XY relative to its fixed terrain origin',
                    angular_integral='integral of body-frame omega_z, not world yaw angle',
                    reverse_rebound='approximate body-frame XY projection onto the pre-stop body velocity; frames rotate',
                    vertical_velocity='body-frame z component, not world vertical velocity',
                    world_vertical_velocity='world_velocity z when recorded; null when unavailable',
                    sample_timing='issued command at k*dt; kinematics at (k+1)*dt; pre-stop state from k-1',
                    metric_windows='kinematics include the pre-stop state; requested/target/torque/speed peaks use zero-command steps only',
                    torque_source='torque_peak substep maximum' if 'torque_peak' in data else 'torque sampled applied torque'),
                cases=[])
        zero = (np.abs(command[:valid, index]) <= THRESHOLDS['zero_command_atol']).all(-1)
        starts = np.flatnonzero(zero[1:] & ~zero[:-1]) + 1
        events = []
        for start in starts:
            moving = np.flatnonzero(~zero[start:])
            stop_end = int(start + moving[0] - 1) if len(moving) else valid - 1
            events.append(_event(manifest, data, case, int(start), stop_end, bool(len(moving)),
                                 valid, leg_ids, wheel_ids, base_id))
        row = {key: case[key] for key in ('id', 'policy', 'terrain', 'direction', 'mode', 'velocity',
                                           'end_step', 'physical', 'terms') if key in case}
        row.update(valid_steps=valid, inactive_truncated=valid < end + 1, stop_events=events)
        reports[label]['cases'].append(row)
    return reports


def write_reports(directory, output_directory=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    with np.load(directory / 'traces.npz', allow_pickle=False) as data:
        reports = analyze_braking(manifest, data)
    output = directory if output_directory is None else Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for label, report in reports.items():
        safe_label = re.sub(r'[^A-Za-z0-9_.-]', '_', label)
        path = output / f'BRAKE_METRICS_{safe_label}.json'
        if path in paths:
            raise ValueError('policy labels collide after filename sanitization')
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        paths.append(path)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output-directory', type=Path)
    args = parser.parse_args()
    for written in write_reports(args.directory, args.output_directory):
        print(written)
