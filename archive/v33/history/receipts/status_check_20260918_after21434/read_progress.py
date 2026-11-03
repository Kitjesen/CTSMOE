"""Read existing formal metrics without changing the training process."""
from collections import deque
import json
from pathlib import Path
import statistics


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
run = root / 'runs/official4096_30000_rs0402peak_20260918'
with (run / 'metrics.jsonl').open() as stream:
    lines = list(deque(stream, maxlen=1001))
rows = [json.loads(line) for line in lines]
assert len(rows) == 1001


def window(first, last):
    result = {'from_updates': first['completed_updates'], 'to_updates': last['completed_updates']}
    for terrain in ('stairs_up', 'stairs_down'):
        a = first['stair_speed']['episode_statistics']['student'][terrain]
        b = last['stair_speed']['episode_statistics']['student'][terrain]
        row = {key: sum(x[key] for x in b) - sum(x[key] for x in a)
               for key in ('episodes', 'grade_eligible', 'radius_passes', 'physical_failures')}
        row['physical_failure_percent'] = 100 * row['physical_failures'] / row['episodes']
        mae_delta = sum(x['vx_mae_mps'] * x['episodes'] for x in b) - sum(x['vx_mae_mps'] * x['episodes'] for x in a)
        row['episode_mean_vx_mae_mps'] = mae_delta / row['episodes']
        result[terrain] = row
    return result


last = rows[-1]
speed = statistics.mean(row['seconds'] for row in rows[-100:])
result = {
    'completed_updates': last['completed_updates'],
    'terrain': last['terrain'],
    'terrain_by_type': last['terrain_by_type'],
    'seconds_mean_last100': speed,
    'remaining_hours_at_recent_speed': (30000 - last['completed_updates']) * speed / 3600,
    'previous500': window(rows[0], rows[500]),
    'recent500': window(rows[500], rows[-1]),
    'max_speed_last100': {
        group: max(joint['peak_speed'] for row in rows[-100:] for name, joint in row['actuator_guard']['joints'].items()
                   if name.endswith('foot_joint') == (group == 'wheel')) for group in ('leg', 'wheel')
    },
    'torque_violation_samples_last100': sum(joint['torque_violation_samples'] for row in rows[-100:]
                                          for joint in row['actuator_guard']['joints'].values()),
    'scope': 'Existing stochastic training trajectories, mixed levels and randomization. Not a frozen-policy evaluation.',
}
Path(__file__).with_name('progress.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
