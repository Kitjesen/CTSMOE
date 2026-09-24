"""Explicit braking and stair-landing diagnostic cases; no training mutations."""
import numpy as np


def build_focus_cases(selection, suite, original_builder):
    if suite == 'stairend':
        rows = [dict(c, landing_mode=mode) for c in original_builder(selection, 'terrain83')
                if c['terrain'] in ('stairs_up', 'stairs_down') for mode in ('continue', 'stop')]
        for i, row in enumerate(rows):
            row['id'] = i
        return rows
    if suite != 'brake25':
        return original_builder(selection, suite)
    prototypes = original_builder(selection, 'flat25')
    commands = []
    for axis, name, values in [(0, 'longitudinal', (.1, 1., 2.)),
                               (1, 'lateral', (.2, .6, 1.)),
                               (2, 'yaw', (.2, .8, 1.5))]:
        for sign in (1., -1.):
            for speed in values:
                value = [0., 0., 0.]
                value[axis] = sign*speed
                commands.append((f'{name}_{sign:+g}_{speed:g}', value))
    rows = []
    for policy in range(len(selection)):
        for mode in ('direct', 'settle'):
            template = next(c for c in prototypes if c['policy'] == policy and c['mode'] == mode)
            for label, velocity in commands:
                row = dict(template, id=len(rows), direction=label, velocity=velocity)
                rows.append(row)
    return rows


def brake_factor(elapsed):
    """Five full 5-second segments: move, stop, reverse, stop, move."""
    segment = (elapsed / 5.).floor().long() % 5
    return ((segment == 0) | (segment == 4)).float() - (segment == 2).float()


def stair_window_metrics(manifest, arrays):
    rows = []
    dt = manifest['dt']
    default = np.asarray(manifest['default_q'])
    ids = [i for i, name in enumerate(manifest['joint_names'])
           if name.endswith(('_hip_joint', '_thigh_joint', '_calf_joint'))]
    for case in manifest['cases']:
        i, count = case['id'], case['end_step'] + 1
        entered = np.flatnonzero(arrays['stair_platform_hull'][:count, i])
        approaching = np.flatnonzero(arrays['stair_edge_window'][:count, i])
        start = max(0, int(approaching[0])-int(.5/dt)) if len(approaching) else 0
        end = min(count, int(entered[0])+int(2./dt)) if len(entered) else count
        q_error = arrays['joint_pos'][start:end, i][:, ids] - default[ids]
        rows.append(dict(id=i, policy=case['policy'], terrain=case['terrain'], level=case['level'],
            landing_mode=case['landing_mode'],
            physical=case['physical'], window_start_s=start*dt, window_end_s=end*dt,
            terms=case['terms'], end_reason=case['end_reason'], platform_hold_s=case['platform_hold_s'],
            supported_platform_reached=case['supported_platform_reached'],
            actual_hull_reached_platform=bool(len(entered)),
            platform_entry_s=float(entered[0]*dt) if len(entered) else None,
            window_positive_vz_peak=float(np.maximum(arrays['world_velocity'][start:end, i, 2], 0).max()),
            window_q_rms_peak=float(np.sqrt((q_error*q_error).mean(-1)).max()),
            window_per_joint_abs_peak=np.abs(q_error).max(0).tolist(),
            true_phase_weight_max=float(arrays['phase_weight'][start:end, i].max()),
            true_phase_stable_s_max=float(arrays['phase_stable_s'][start:end, i].max()),
            supported_platform_steps=int(arrays['stair_platform_supported'][:count, i].sum()),
            diagnostic_geometry='same capped outer landing for all policies; distinct from legacy 4m-route test'))
    return dict(cases=rows, dt=dt, learned_success_rate=False)
