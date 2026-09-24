"""Summarize frozen-policy retest and pre-projection requests on the server."""
import json
from pathlib import Path

import numpy as np


audit = Path(__file__).resolve().parent
summary = {}
for label in ('old19001', 'current21001'):
    folder = audit / label
    result = json.loads((folder / 'results.json').read_text())
    traces = np.load(folder / 'stair_failure_traces.npz')
    names = list(traces['joint_names'])
    leg_names = list(traces['leg_action_joint_names'])
    leg_ids = [names.index(name) for name in leg_names]
    soft = np.asarray(result['soft_limits_rad'])[leg_ids]
    rates = list(traces['reward_names'])
    groups = {}
    for learner in ('teacher', 'student'):
        for terrain in ('wave', 'stairs_up', 'stairs_down'):
            cases = [c for c in result['cases'] if c['learner'] == learner and c['terrain'] == terrain]
            groups[f'{learner}_{terrain}'] = {
                'trials': len(cases), 'passes': sum(bool(c['forward_exit']) for c in cases),
                'failures': sum(bool(c['physical_failure']) for c in cases),
                'by_level_speed': {
                    f'L{level}_v{speed}': {
                        'passes': sum(bool(c['forward_exit']) for c in cases if c['level'] == level and c['speed'] == speed),
                        'trials': sum(c['level'] == level and c['speed'] == speed for c in cases),
                    } for level in (3, 6, 9) for speed in (0.3, 0.6, 0.9)
                },
            }
    details = []
    for column, index in enumerate(traces['case_indices']):
        case = result['cases'][int(index)]
        # Report all failures and each repeat of the user's specific high/slow case.
        if case['forward_exit'] and not (case['terrain'] == 'stairs_up' and case['level'] == 9 and case['speed'] == 0.3):
            continue
        active = traces['active_before_step'][:, column]
        last_time = traces['time_s'][active][-1]
        keep = active & (traces['time_s'] >= last_time - 5.0 - 1e-8)
        requested = traces['requested_leg_target_rad'][keep, column]
        processed = traces['joint_pos_target_rad'][keep, column][:, leg_ids]
        position = traces['joint_pos_rad'][keep, column][:, leg_ids]
        root = traces['root_pos_relative_w_m'][keep, column]
        per_leg = {}
        for j, name in enumerate(leg_names):
            beyond = np.maximum(soft[j, 0] - requested[:, j], requested[:, j] - soft[j, 1]).clip(0)
            per_leg[name] = {
                'requested_min_max_rad': [float(requested[:, j].min()), float(requested[:, j].max())],
                'requested_mean_rad': float(requested[:, j].mean()),
                'soft_bound_rad': soft[j].tolist(),
                'requested_outside_soft_fraction': float((beyond > 1e-5).mean()),
                'requested_outside_soft_max_rad': float(beyond.max()),
                'processed_min_max_rad': [float(processed[:, j].min()), float(processed[:, j].max())],
                'processed_mean_rad': float(processed[:, j].mean()),
                'actual_mean_rad': float(position[:, j].mean()),
            }
        details.append({
            'case_index': int(index),
            'case': {k: v for k, v in case.items() if k not in ('joints', 'decomposition_final', 'decomposition_at_peak')},
            'window_s': [float(traces['time_s'][keep][0]), float(last_time)],
            'forward_delta_m': float(root[-1, 0] - root[0, 0]),
            'leg_requests': per_leg,
            'weighted_reward_rate_mean': dict(zip(rates, traces['reward_weighted_rate'][keep, column].mean(0).astype(float).tolist())),
            'weighted_total_reward_rate_mean': float(traces['reward_weighted_rate'][keep, column].sum(-1).mean()),
        })
    summary[label] = {'completed_updates': result['completed_updates'], 'checkpoint_sha256': result['checkpoint_sha256'],
                      'groups': groups, 'details': details, 'reward_weights': result['reward_weights']}
    if label == 'old19001':
        previous = json.loads((audit.parent / 'stair_failure_peak_20260918/guarded/results.json').read_text())
        summary[label]['all_162_cases_exact_replay'] = result['cases'] == previous['cases']
(audit / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({label: {'groups': row['groups'], 'exact_replay': row.get('all_162_cases_exact_replay')} for label, row in summary.items()}))
