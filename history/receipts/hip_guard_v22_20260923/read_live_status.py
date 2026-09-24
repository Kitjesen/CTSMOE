"""Read the bounded V22 run without changing process or checkpoint state."""
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/cts-omni-repair-v19-20260922/hip_guard_v22_attempt03')
result = {'pipeline': json.loads((ROOT / 'pipeline_status.json').read_text()), 'segments': []}
for folder in sorted(ROOT.glob('training_*')):
    item = {'path': str(folder)}
    for name in ('status.json', 'SUCCESS.json'):
        path = folder / name
        if path.exists():
            item[name] = json.loads(path.read_text())
    path = folder / 'run.json'
    if path.exists():
        run = json.loads(path.read_text())
        item['restore'] = {key: run.get(key) for key in (
            'hip_guard_profile', 'resume_checks', 'additional_updates',
            'target_completed_updates', 'resume_sha256', 'num_envs')}
    path = folder / 'metrics.jsonl'
    if path.exists():
        lines = path.read_text().splitlines()
        if lines:
            last = json.loads(lines[-1])
            item['latest_metric_keys'] = list(last)
            item['latest'] = {key: value for key, value in last.items()
                              if key in ('completed_updates', 'iteration', 'collection_time',
                                         'learn_time', 'mean_reward', 'hip_guard')}
    item['checkpoints'] = [p.name for p in sorted(folder.glob('model_*.pt'))]
    result['segments'].append(item)
print(json.dumps(result, indent=2))
