"""Summarize recorded reward contributions and saved stair torque costs."""
import json
from pathlib import Path
import re
import statistics

import numpy as np
import yaml


here = Path(__file__).resolve().parent
config = yaml.load((here / 'env.yaml').read_text(), Loader=yaml.BaseLoader)
log = (here / 'training_snapshot.log').read_text()
blocks = re.split(r'Learning iteration\s+(\d+)/\d+', log)
rows = []
for i in range(1, len(blocks)-1, 2):
    values = {name: float(value) for name, value in re.findall(
        r'Episode_Reward/(\w+):\s+(-?[\d.eE+]+)', blocks[i+1])}
    if len(values) == len(config['rewards']):
        rows.append({'completed_updates': int(blocks[i])+1, 'rewards': values})
assert len(rows) >= 100
selected = rows[-100:]
averages = {name: statistics.mean(row['rewards'][name] for row in selected)
            for name in selected[0]['rewards']}
weights = {name: float(term['weight']) for name, term in config['rewards'].items()}
for term in config['curriculum'].values():
    params = term['params']
    if 'term_name' in params and selected[-1]['completed_updates'] >= int(params['end_it']):
        weights[params['term_name']] = float(params['final_weight'])

folder = here.parents[1] / 'recordings/straight_high13501_20260918/trace'
manifest = json.loads((folder / 'manifest.json').read_text())
leg_ids = [i for i, name in enumerate(manifest['joint_names']) if '_foot_' not in name]
wheel_ids = [i for i, name in enumerate(manifest['joint_names']) if '_foot_' in name]
trace_costs = []
for level in (7, 9):
    records = json.loads((folder / f'stairs_up_L{level}.json').read_text())
    times = np.array([r['time'] for r in records])
    torques = np.array([r['applied_torque'] for r in records])
    velocity = np.array([r['velocity'] for r in records])
    command = np.array([r['command'] for r in records])
    costs = []
    for start, end in ((2, 5), (6, 10), (12, 16)):
        mask = (times >= start) & (times < end)
        total_torque_cost = weights['joint_torques_l2'] * np.square(torques[mask]).sum(axis=1)
        wheel_cost = weights['joint_torques_l2'] * np.square(torques[mask][:, wheel_ids]).sum(axis=1)
        leg_cost = weights['joint_torques_l2'] * np.square(torques[mask][:, leg_ids]).sum(axis=1)
        tracking = weights['track_lin_vel_xy_exp'] * np.exp(
            -np.square(command[mask, :2]-velocity[mask, :2]).sum(axis=1)/.5**2)
        costs.append({'interval_s': [start, end], 'torque_cost_before_dt': float(total_torque_cost.mean()),
                      'leg_torque_cost_before_dt': float(leg_cost.mean()),
                      'wheel_torque_cost_before_dt': float(wheel_cost.mean()),
                      'xy_tracking_reward_before_dt': float(tracking.mean())})
    trace_costs.append({'level': level, 'intervals': costs})

result = {'completed_update_window': [selected[0]['completed_updates'], selected[-1]['completed_updates']],
          'weights_current': weights, 'mean_episode_reward_log_last100': averages,
          'log_definition': 'Mean of 100 logger rows. Each logged term is weighted episode integral divided by fixed max_episode_length_s=25, averaged across completed episodes; all terrains and teacher/student mixed. Not a per-stair measurement.',
          'trace_policy_completed_updates': 13501, 'trace_costs': trace_costs,
          'trace_limitations': 'One nominal rollout per level. Torque sampled at final physics substep at 50 Hz. Costs recomputed before the common dt factor 0.02; these are not comparable as identical estimators to episode logs. No causal intervention.'}
(here / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
