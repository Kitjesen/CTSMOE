"""Read formal reward telemetry and recent episode statistics."""
from collections import deque
import json
from pathlib import Path
import statistics

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
run = root / 'runs/official4096_30000_rs0402peak_20260918'
with (run / 'metrics.jsonl').open() as stream:
    lines = list(deque(stream, maxlen=501))
rows = [json.loads(line) for line in lines]
first, last = rows[0], rows[-1]
episodes = {}
for terrain in ('stairs_up', 'stairs_down'):
    a = first['stair_speed']['episode_statistics']['student'][terrain]
    b = last['stair_speed']['episode_statistics']['student'][terrain]
    n = sum(x['episodes'] for x in b) - sum(x['episodes'] for x in a)
    failures = sum(x['physical_failures'] for x in b) - sum(x['physical_failures'] for x in a)
    episodes[terrain] = {'episodes': n, 'physical_failures': failures, 'failure_percent': 100 * failures / n,
                         'episode_vx_mae_mps': (sum(x['vx_mae_mps'] * x['episodes'] for x in b)
                                               - sum(x['vx_mae_mps'] * x['episodes'] for x in a)) / n}
events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
tags = events.Tags()['scalars']
reward_tags = [tag for tag in tags if 'Episode_Reward/' in tag]
rewards = {}
for tag in reward_tags:
    values = events.Scalars(tag)[-100:]
    rewards[tag] = {'mean_last100': statistics.mean(x.value for x in values),
                    'last_step': values[-1].step, 'samples': len(values)}
result = {'from_updates': first['completed_updates'], 'completed_updates': last['completed_updates'],
          'state': last['state'], 'seconds_mean100': statistics.mean(r['seconds'] for r in rows[-100:]),
          'terrain': last['terrain'], 'terrain_by_type': last['terrain_by_type'], 'student_last500': episodes,
          'reward_telemetry': rewards, 'all_scalar_tags': tags,
          'reward_scope': 'Existing Episode_Reward logging; normalized episode sums, not identical to instantaneous reward rates in the frozen probe.'}
Path(__file__).with_name('training.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
