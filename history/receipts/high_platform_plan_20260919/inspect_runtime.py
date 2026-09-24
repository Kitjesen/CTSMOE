"""Compact read-only runtime evidence for the two authorized training processes."""
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
result = {'queried_at': datetime.now().astimezone().isoformat(), 'runs': {}}
for name, directory, pid in (
    ('original', 'official4096_30000_wave_surface_20260919', 107477),
    ('pit', 'pit_formal1024_v3_20260919', 115374),
):
    path = root/'runs'/directory
    os.kill(pid, 0)
    row = json.loads((path/'status.json').read_text())
    info = {k: row[k] for k in ('state', 'completed_updates', 'seconds') if k in row}
    info.update(pid=pid, directory=str(path))
    if name == 'pit':
        run = json.loads((path/'run.json').read_text())
        with (path/'metrics.jsonl').open() as stream:
            recent = [json.loads(s) for s in deque(stream, maxlen=10)]
        info.update(num_envs=run['num_envs'], teacher_envs=run['teacher_envs'], student_envs=run['student_envs'],
                    target_updates=run['starting_updates']+run['additional_updates'],
                    seconds_last10_mean=sum(v['seconds'] for v in recent)/len(recent),
                    frontier_height_m=row['pit']['frontier_height_m'], max_height_m=row['pit']['max_height_m'],
                    learning_rate=row['learning_rate'], losses=row['losses'],
                    learner_totals={k: v['totals'] for k,v in row['pit']['by_learner'].items()},
                    checkpoint_names=sorted(p.name for p in path.glob('model_*.pt')))
    result['runs'][name] = info
result['gpu'] = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name,memory.used,memory.free,utilization.gpu', '--format=csv'], text=True).strip()
(root/'runs/pit_formal1024_v3_20260919/runtime_start.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
