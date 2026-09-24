import json
from pathlib import Path
import statistics

p = Path(__file__).parent
rows = [json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
a, b = rows[:100], rows[-100:]
result = dict(first=rows[0]['completed_updates'], last=rows[-1]['completed_updates'],
              last100_seconds=statistics.mean(x['seconds'] for x in b),
              terrain={k: dict(first100=statistics.mean(x['terrain_by_type'][k]['mean'] for x in a),
                              last100=statistics.mean(x['terrain_by_type'][k]['mean'] for x in b))
                       for k in rows[-1]['terrain_by_type']})
start, end = rows[-501], rows[-1]
result['window'] = [start['completed_updates'], end['completed_updates']]
result['pit_1p2m'] = {}
for i, name in enumerate(('teacher', 'student')):
    before = start['pit']['climb_by_height_speed'][i][21]
    after = end['pit']['climb_by_height_speed'][i][21]
    delta = [sum(x[j]-y[j] for x, y in zip(after, before)) for j in range(4)]
    result['pit_1p2m'][name] = dict(episodes=delta[0], climb=delta[1],
        climb_rate=delta[1]/delta[0], failed=delta[2], stable=delta[3])
result['stairs'] = {}
for name in ('teacher', 'student'):
    result['stairs'][name] = {}
    for terrain in ('stairs_up', 'stairs_down'):
        prev = start['stair_speed']['episode_statistics'][name][terrain]
        curr = end['stair_speed']['episode_statistics'][name][terrain]
        totals = {k: sum(x[k]-y[k] for x,y in zip(curr,prev))
                  for k in ('episodes','grade_eligible','radius_passes','physical_failures')}
        totals['physical_rate'] = totals['physical_failures']/totals['episodes']
        result['stairs'][name][terrain] = totals
result['finite_losses'] = {k: [min(x[k] for x in b),max(x[k] for x in b)]
                           for k in ('value','surrogate','mean_latent_loss')}
(p/'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
