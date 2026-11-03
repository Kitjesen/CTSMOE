"""Read-only live evidence for the authorized mixed terrain continuation."""
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
run = root/'runs/mixed9_4096_60000_20260919'
os.kill(126943, 0)
status = json.loads((run/'status.json').read_text())
receipt = json.loads((run/'run.json').read_text())
rows = [json.loads(line) for line in deque((run/'metrics.jsonl').open(), maxlen=10)]
result = dict(queried_at=datetime.now().astimezone().isoformat(), pid=126943,
              state=status['state'], completed_updates=status['completed_updates'],
              target_updates=receipt['target_completed_updates'],
              seconds_last10_mean=sum(row['seconds'] for row in rows)/len(rows),
              terrain_original_mean=status['terrain'],
              terrains={name: {'count': data['count'], 'mean': data['mean']}
                        for name, data in status['terrain_by_type'].items()},
              pit_frontier_m=status['pit']['frontier_height_m'],
              pit_totals={name: data['totals'] for name, data in status['pit']['by_learner'].items()},
              resume_checks=receipt['resume_checks'], mixed_checks=receipt['mixed_checks'],
              checkpoint_names=sorted(p.name for p in run.glob('model_*.pt')),
              stopped_pit=json.loads((root/'runs/pit_formal1024_v3_20260919/status.json').read_text()))
(run/'runtime_verified.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
