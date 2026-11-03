"""Summarize applied target slew, reward contributions, and frozen trial outcomes."""
import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent


def describe(values):
    values = np.asarray(values)
    return {'mean': float(values.mean()), 'p95': float(np.quantile(values, .95)),
            'p99': float(np.quantile(values, .99)), 'max': float(values.max())}


output = {}
for mode in ('baseline', 'heading'):
    folder = root / mode
    result = json.loads((folder / 'results.json').read_text())
    z = np.load(folder / 'stair_failure_traces.npz')
    names = list(z['joint_names'])
    leg_names = list(z['leg_action_joint_names'])
    ids = [names.index(name) for name in leg_names]
    soft = np.asarray(result['soft_limits_rad'])[ids]
    requested = z['requested_leg_target_rad']
    previous = z['previous_leg_target_rad'].clip(soft[:, 0], soft[:, 1])
    bounded_request = requested.clip(soft[:, 0], soft[:, 1])
    target = z['joint_pos_target_rad'][:, :, ids]
    expected = (previous + (bounded_request - previous).clip(-.12, .12)).clip(soft[:, 0], soft[:, 1])
    active = z['active_before_step']
    np.testing.assert_allclose(target[active], expected[active], atol=2e-6, rtol=0)
    rate = np.abs(target - previous) / .02
    assert rate[active].max() <= 6.0001
    limited = np.abs(bounded_request - previous) > .12001
    outside = np.maximum(soft[:, 0] - requested, requested - soft[:, 1]) > 1e-5
    reward_names = list(z['reward_names'])
    groups = {}
    for terrain in ('wave', 'stairs_up', 'stairs_down'):
        columns = [i for i, case_id in enumerate(z['case_indices']) if result['cases'][int(case_id)]['terrain'] == terrain]
        cases = [result['cases'][int(z['case_indices'][i])] for i in columns]
        valid = active[:, columns]
        values = rate[:, columns][valid]
        rewards = z['reward_weighted_rate'][:, columns][valid]
        # Rates are weighted per-joint/control-step samples, rewards per active control-step.
        groups[terrain] = {
            'trials': len(cases), 'passes_4m': sum(bool(c['forward_exit']) for c in cases),
            'failures': [{k: c[k] for k in ('level', 'speed', 'repeat', 'duration_s', 'final_forward_m', 'termination_reasons')}
                         for c in cases if not c['forward_exit']],
            'target_rate_rad_s': describe(values),
            'target_rate_limit_fraction': float(limited[:, columns][valid].mean()),
            'initial_03s_target_rate_rad_s': describe(rate[:, columns][valid & (z['time_s'][:, None] <= .300001)]),
            'initial_03s_rate_limit_fraction': float(limited[:, columns][valid & (z['time_s'][:, None] <= .300001)].mean()),
            'per_leg_target_rate': {name: {**describe(values[:, j]),
                                         'limited_fraction': float(limited[:, columns][valid][:, j].mean())}
                                    for j, name in enumerate(leg_names)},
            'requested_outside_soft_fraction': float(outside[:, columns][valid].mean()),
            'joint_speed_50hz_rad_s': describe(np.abs(z['joint_vel_rad_s'][:, columns][:, :, ids][valid])),
            'weighted_reward_rates_mean': dict(zip(reward_names, rewards.mean(0).astype(float).tolist())),
            'total_reward_rate_mean': float(rewards.sum(-1).mean()),
            'vx_mae_case_mean_mps': float(np.mean([c['vx_mae_mps'] for c in cases])),
            'passed_cases_max_cross_m': describe([c['max_cross_m'] for c in cases if c['forward_exit']]),
            'passed_cases_cross_over_05m': sum(c['max_cross_m'] > .5 for c in cases if c['forward_exit']),
            'level9_by_speed': {str(speed): {
                'passes': sum(bool(c['forward_exit']) for c in cases if c['level'] == 9 and c['speed'] == speed),
                'cases': [{k: c[k] for k in ('repeat', 'forward_exit', 'duration_s', 'vx_mae_mps', 'max_cross_m')}
                          for c in cases if c['level'] == 9 and c['speed'] == speed],
            } for speed in (.3, .6, .9)},
        }
    output[mode] = {'checkpoint_updates': result['completed_updates'], 'checkpoint_sha256': result['checkpoint_sha256'],
                    'target_projection_verified': True, 'groups': groups,
                    'reward_weights': result['reward_weights'],
                    'scope': 'Nominal dynamics, no noise/randomization, first 4 m exit or termination. 27 student trials per terrain. All active samples including first terminal step retained.'}
(root / 'analysis.json').write_text(json.dumps(output, indent=2))
print(json.dumps({mode: {name: {k: row[k] for k in ('passes_4m', 'target_rate_rad_s', 'target_rate_limit_fraction', 'vx_mae_case_mean_mps')}
                          for name, row in data['groups'].items()} for mode, data in output.items()}))
